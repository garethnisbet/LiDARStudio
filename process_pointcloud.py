#!/usr/bin/env python3
"""
Colour Point Cloud Generator

Reads a LIDAR bag and an IMAGE bag, projects the 3-D LiDAR points onto
the closest camera frame using the calibration extrinsics, assigns RGB
colours, optionally voxel-downsamples, and writes a PLY file.

Usage (called automatically by lidar_server.py):
    python process_pointcloud.py
        --lidar-bag  LIDAR_20260615182333.bag
        --image-bag  IMAGE_20260615182333.bag
        --output     /project/pointclouds/pointcloud_20260615182333.ply
        [--params    project_parameters.json]
        [--calibration  calibration/]
        [--voxel-size 0.05]

Progress protocol:
    Lines that begin with "PROGRESS:<percent>:<message>" are parsed by
    lidar_server.py and shown in the progress bar.  All other printed
    lines appear in the log view.

Required packages:
    pip install rosbags open3d numpy opencv-python-headless

Calibration format (calibration/calib.json):
    camera_info.<name>.K     — 9-element flat row-major 3×3 intrinsic matrix
    camera_info.<name>.coeff — 4-element distortion coefficients (k1,k2,p1,p2
                                or k1,k2,k3,k4 for fisheye)
    out_put.<name>.transform_matrix — 4×4 LiDAR-to-camera extrinsic (row-major)
"""

import argparse
import json
import sys
from pathlib import Path


# ── Dependency check ─────────────────────────────────────────────────

def _require(*packages):
    missing = []
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"ERROR: Missing packages: {', '.join(missing)}", flush=True)
        print(f"Install with:  pip install {' '.join(missing)}", flush=True)
        sys.exit(1)


def progress(pct: int, msg: str):
    print(f"PROGRESS:{pct}:{msg}", flush=True)


# ── Calibration loader ───────────────────────────────────────────────

def load_calibration(calib_dir: Path, camera_name: str = "front") -> dict:
    """Return intrinsics and extrinsics for the named camera."""
    calib_file = calib_dir / "calib.json"
    if not calib_file.exists():
        raise FileNotFoundError(f"calib.json not found in {calib_dir}")

    cal = json.loads(calib_file.read_text())

    cam_info = cal["camera_info"][camera_name]
    out_put   = cal["out_put"][camera_name]

    K_flat = cam_info["K"]          # 9-element flat 3×3 row-major
    coeff  = cam_info["coeff"]      # distortion coefficients
    T_mat  = out_put["transform_matrix"]  # 4×4 list-of-lists (LiDAR → camera)

    import numpy as np
    K = np.array(K_flat, dtype=float).reshape(3, 3)
    T = np.array(T_mat, dtype=float)          # shape (4, 4)

    return {"K": K, "D": np.array(coeff, dtype=float), "T": T}


# ── Bag readers ──────────────────────────────────────────────────────

def read_lidar_points(bag_path: Path):
    """
    Yield (timestamp_ns, xyz_array) from PointCloud2 messages in a ROS1 bag.

    xyz_array is shape (N, 3) float32.
    """
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    import numpy as np

    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Reader(str(bag_path)) as bag:
        # Find a topic that looks like a point cloud
        pc_topic = None
        for topic, info in bag.topics.items():
            if "PointCloud2" in info.msgtype:
                pc_topic = topic
                break
        if pc_topic is None:
            raise RuntimeError("No PointCloud2 topic found in LIDAR bag")
        print(f"  LiDAR topic: {pc_topic}", flush=True)

        conn = [c for c in bag.connections if c.topic == pc_topic]
        for connection, timestamp, rawdata in bag.messages(connections=conn):
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            # Extract XYZ from structured point cloud
            data = np.frombuffer(msg.data, dtype=np.uint8)
            # Determine offsets for x, y, z fields
            fields = {f.name: f for f in msg.fields}
            offsets = [fields[n].offset for n in ("x", "y", "z")]
            row_step = msg.point_step
            n_pts = msg.width * msg.height
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n_pts, row_step)
            xyz = np.stack([
                np.frombuffer(raw[:, off:off+4].tobytes(), dtype=np.float32)
                for off in offsets
            ], axis=1)
            yield timestamp, xyz


def rotate_to_portrait(img, rot: str):
    """
    Rotate a decoded camera frame so it matches the (portrait) calibrated
    intrinsics.  The JMK7 front camera records 4000×3000 landscape JPEGs but the
    calibration K is portrait (cx≈1586, cy≈2099 → centred only on a 3000×4000
    canvas), so every frame must be turned 90° before projection/training.
    `rot` is one of "cw", "ccw", "180", "none".
    """
    import cv2
    if rot == "cw":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rot == "ccw":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rot == "180":
        return cv2.rotate(img, cv2.ROTATE_180)
    return img


def read_images(bag_path: Path, camera_name: str = "front", rot: str = "ccw"):
    """
    Yield (timestamp_ns, bgr_image) from the camera bag.

    Tries both raw Image and CompressedImage message types.  Each frame is
    rotated to portrait (`rot`) so it lines up with the calibrated intrinsics.
    """
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    import numpy as np
    import cv2

    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Reader(str(bag_path)) as bag:
        # Find a topic matching the camera name
        img_topic = None
        for topic, info in bag.topics.items():
            if camera_name in topic and ("Image" in info.msgtype or "image" in topic):
                img_topic = topic
                break
        if img_topic is None:
            # Fall back to first image-like topic
            for topic, info in bag.topics.items():
                if "Image" in info.msgtype:
                    img_topic = topic
                    break
        if img_topic is None:
            raise RuntimeError("No image topic found in IMAGE bag")
        print(f"  Image topic: {img_topic}", flush=True)

        conn = [c for c in bag.connections if c.topic == img_topic]
        for connection, timestamp, rawdata in bag.messages(connections=conn):
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            if "Compressed" in connection.msgtype:
                buf = np.frombuffer(msg.data, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            else:
                h, w = msg.height, msg.width
                enc = msg.encoding
                buf = np.frombuffer(msg.data, dtype=np.uint8)
                if enc in ("rgb8", "bgr8"):
                    img = buf.reshape(h, w, 3)
                    if enc == "rgb8":
                        img = img[:, :, ::-1]  # RGB → BGR
                else:
                    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            yield timestamp, rotate_to_portrait(img, rot)


# ── IMU reader & orientation integration ─────────────────────────────
#
# The Vanjee 722z (and similar handheld scanners) emit each PointCloud2 in
# the *instantaneous sensor frame*.  While scanning, the unit is panned and
# tilted, so stacking the raw frames piles every laser beam onto a fixed
# direction at a varying range — the cloud collapses into lines radiating
# from the sensor origin.  To rebuild a coherent scene we must rotate each
# frame into a common world frame using the sensor orientation at that
# moment.  The bag carries no absolute pose, but it does carry a gyro, so we
# integrate angular velocity to recover the (relative) orientation track.
#
# This corrects rotation only.  Translation is assumed negligible (true for
# a tripod/pan-tilt scan); a moving platform would need full LiDAR-inertial
# odometry, which is out of scope here.

def read_imu(bag_path: Path):
    """
    Return (timestamps_ns, angular_velocity, linear_acceleration) from the
    first Imu topic.

    timestamps_ns : (M,) int64 ndarray
    angular_velocity : (M, 3) float64 ndarray, rad/s in the sensor frame.
    linear_acceleration : (M, 3) float64 ndarray, m/s² in the sensor frame
        (specific force — at rest this points along world-up).
    Returns (None, None, None) if the bag has no Imu topic.
    """
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    import numpy as np

    typestore = get_typestore(Stores.ROS1_NOETIC)
    with Reader(str(bag_path)) as bag:
        imu_topic = None
        for topic, info in bag.topics.items():
            if "Imu" in info.msgtype:
                imu_topic = topic
                break
        if imu_topic is None:
            return None, None, None
        print(f"  IMU topic: {imu_topic}", flush=True)

        ts_list, w_list, a_list = [], [], []
        conn = [c for c in bag.connections if c.topic == imu_topic]
        for connection, timestamp, rawdata in bag.messages(connections=conn):
            msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
            w = msg.angular_velocity
            a = msg.linear_acceleration
            ts_list.append(timestamp)
            w_list.append((w.x, w.y, w.z))
            a_list.append((a.x, a.y, a.z))

    if not ts_list:
        return None, None, None
    ts = np.asarray(ts_list, dtype=np.int64)
    w  = np.asarray(w_list,  dtype=np.float64)
    a  = np.asarray(a_list,  dtype=np.float64)
    order = np.argsort(ts)
    return ts[order], w[order], a[order]


def _quat_mul(a, b):
    import numpy as np
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ])


def _quat_from_rotvec(v):
    import numpy as np
    ang = float(np.linalg.norm(v))
    if ang < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = v / ang
    s = np.sin(ang / 2.0)
    return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(ang / 2.0)])


def _quat_to_R(q):
    import numpy as np
    x, y, z, w = q
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s*(y*y + z*z), s*(x*y - z*w),     s*(x*z + y*w)],
        [s*(x*y + z*w),     1 - s*(x*x + z*z), s*(y*z - x*w)],
        [s*(x*z - y*w),     s*(y*z + x*w),     1 - s*(x*x + y*y)],
    ])


def _level_quat(acc):
    """
    Quaternion [x, y, z, w] that rotates the measured gravity direction `acc`
    (specific force in the sensor frame, ≈ world-up at rest) onto +Z, so the
    integrated cloud comes out level.  Heading (rotation about Z) is left
    undetermined — there is no magnetometer to fix it.
    """
    import numpy as np
    g = np.asarray(acc, dtype=np.float64)
    norm = np.linalg.norm(g)
    if norm < 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0])
    g = g / norm
    up = np.array([0.0, 0.0, 1.0])
    axis = np.cross(g, up)
    s = np.linalg.norm(axis)
    c = float(np.dot(g, up))
    if s < 1e-8:                       # already (anti)parallel to up
        if c > 0:
            return np.array([0.0, 0.0, 0.0, 1.0])
        return np.array([1.0, 0.0, 0.0, 0.0])   # 180° flip about X
    angle = np.arctan2(s, c)
    return _quat_from_rotvec(axis / s * angle)


def integrate_orientation(imu_ts, imu_w, imu_a=None):
    """
    Integrate gyro angular velocity into an orientation track (sensor→world).

    Returns (imu_ts, quats) where quats is (M, 4) [x, y, z, w].  When `imu_a`
    is supplied the track is initialised from the gravity direction (averaged
    over the opening samples, assumed roughly static) so the cloud comes out
    level; otherwise it starts from identity.  Heading remains arbitrary — only
    tilt is fixed — but that is all the accelerometer can resolve.
    """
    import numpy as np
    m = len(imu_ts)
    quats = np.zeros((m, 4), dtype=np.float64)
    if imu_a is not None and len(imu_a):
        n0 = min(100, len(imu_a))      # ~0.5 s at 200 Hz — opening static window
        q = _level_quat(imu_a[:n0].mean(axis=0))
    else:
        q = np.array([0.0, 0.0, 0.0, 1.0])
    quats[0] = q
    for i in range(1, m):
        dt = (imu_ts[i] - imu_ts[i-1]) / 1e9
        if dt <= 0:
            quats[i] = q
            continue
        q = _quat_mul(q, _quat_from_rotvec(imu_w[i] * dt))
        q /= np.linalg.norm(q)
        quats[i] = q
    return imu_ts, quats


def frame_rotations(lidar_ts, imu_ts, quats):
    """Return a list of 3×3 sensor→world rotation matrices, one per LiDAR ts."""
    import numpy as np
    Rs = []
    for ts in lidar_ts:
        j = int(np.searchsorted(imu_ts, ts))
        j = min(max(j, 0), len(imu_ts) - 1)
        Rs.append(_quat_to_R(quats[j]))
    return Rs


# ── Full registration (KISS-ICP LiDAR odometry) ──────────────────────
#
# Gyro integration (above) recovers rotation only and assumes the scanner
# stays put — fine for a tripod pan, useless for a handheld walk-through,
# where the frames must also be *translated* into place.  KISS-ICP runs
# scan-to-map ICP odometry over the sequence and returns each scan's full
# 4×4 sensor→world pose (rotation AND translation), which is what makes the
# accumulated cloud resemble the device's own SLAM-fused output.

def register_with_kiss(all_xyz, max_range=50.0, voxel_size=None,
                       deskew=False, progress_cb=None):
    """
    Run KISS-ICP over the LiDAR scans and return a list of 4×4 sensor→world
    pose matrices, one per scan.  Raises ImportError if kiss-icp is absent so
    the caller can fall back to gyro-only stacking.
    """
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
    import numpy as np

    cfg = KISSConfig()
    cfg.data.deskew = deskew
    cfg.data.max_range = float(max_range)
    cfg.data.min_range = 0.0
    # This kiss-icp build does not auto-derive the map voxel size; supply one
    # (the usual heuristic is ~max_range/100).
    cfg.mapping.voxel_size = float(voxel_size) if voxel_size else float(max_range) / 100.0

    kiss = KissICP(cfg)
    empty = np.array([], dtype=np.float64)
    poses = []
    n = len(all_xyz)
    for i, xyz in enumerate(all_xyz):
        frame = np.ascontiguousarray(
            xyz[np.isfinite(xyz).all(axis=1)], dtype=np.float64)
        kiss.register_frame(frame, empty)
        poses.append(np.asarray(kiss.last_pose, dtype=np.float64).copy())
        if progress_cb and (i % 20 == 0 or i == n - 1):
            progress_cb(i + 1, n)
    return poses


def _level_transform(imu_a, n0=100):
    """4×4 transform that rotates the integrated map so gravity points down."""
    import numpy as np
    L = np.eye(4)
    if imu_a is not None and len(imu_a):
        k = min(n0, len(imu_a))
        L[:3, :3] = _quat_to_R(_level_quat(imu_a[:k].mean(axis=0)))
    return L


def apply_pose(pose, xyz):
    """Transform an (N,3) scan by a 4×4 sensor→world pose."""
    return (pose[:3, :3] @ xyz.T).T + pose[:3, 3]


def save_trajectory(cloud_path: Path, lidar_ts, poses):
    """
    Save the per-scan trajectory next to the cloud as ``<cloud>.traj.npz`` so the
    splat stage can derive camera poses (lidar pose ∘ lidar→camera extrinsic).
    Poses are already gravity-levelled, matching the saved cloud.
    """
    import numpy as np
    p = Path(str(cloud_path) + ".traj.npz")
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p,
             ts=np.asarray(lidar_ts, dtype=np.int64),
             poses=np.asarray(poses, dtype=np.float64))
    return p


# ── Colour projection ────────────────────────────────────────────────

def project_and_colour(xyz, image, K, D, T):
    """
    Project LiDAR points (N×3) onto an image and return RGB colours (N×3 uint8).

    Points behind the camera or outside the image boundary are masked out.
    Returns (valid_mask, rgb_array).
    """
    import numpy as np
    import cv2

    n = len(xyz)
    ones = np.ones((n, 1), dtype=float)
    xyz_hom = np.hstack([xyz.astype(float), ones])   # (N, 4)

    # Transform to camera frame
    pts_cam = (T @ xyz_hom.T).T[:, :3]               # (N, 3)

    # Keep only points in front of the camera
    in_front = pts_cam[:, 2] > 0

    # Project with the OpenCV *fisheye* model (the JMK7 lens is ~180°; the 4
    # calib coeffs are equidistant k1..k4, not Brown k1,k2,p1,p2 — the fisheye
    # model lands ~97% of front-facing points in-frame vs ~60% for Brown).
    h, w = image.shape[:2]
    pts_2d, _ = cv2.fisheye.projectPoints(
        pts_cam[in_front].reshape(-1, 1, 3).astype(np.float64),
        np.zeros((3, 1)), np.zeros((3, 1)), K, D.reshape(4, 1)
    )
    pts_2d = pts_2d.squeeze(1)  # (M, 2)

    # Pixel bounds
    ix = np.round(pts_2d[:, 0]).astype(int)
    iy = np.round(pts_2d[:, 1]).astype(int)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)

    rgb = np.zeros((n, 3), dtype=np.uint8)
    in_front_idx = np.where(in_front)[0]
    valid_local  = in_front_idx[inside]

    bgr_pix = image[iy[inside], ix[inside]]          # (K, 3) BGR
    rgb[valid_local] = bgr_pix[:, ::-1]              # BGR → RGB

    mask = np.zeros(n, dtype=bool)
    mask[valid_local] = True
    return mask, rgb


def colour_points_multiview(xyz_world, poses, lidar_ts, image_bag, calib,
                            camera="front", image_rot="ccw", max_range=20.0,
                            occl_tol=0.03, depth_ds=4):
    """Colour world points from *every* camera frame, not just the time-nearest.

    The camera is rigidly mounted to the LiDAR and only sees the front
    hemisphere of each scan, so single-frame colouring leaves ~40% of the fused
    cloud black (and those points get dropped from the splat).  Here each world
    point is projected into all frames using the recovered trajectory; it takes
    the colour from the *closest* view that actually sees it.  A per-frame depth
    buffer rejects points occluded by nearer geometry (the fisheye otherwise
    paints background points with foreground colour).

    Returns (rgb uint8 N×3, n_coloured).
    """
    import numpy as np
    import cv2
    K = calib["K"].astype(np.float64)
    D = calib["D"].reshape(4, 1).astype(np.float64)
    T = calib["T"].astype(np.float64)

    N = len(xyz_world)
    rgb = np.zeros((N, 3), np.uint8)
    best = np.full(N, np.inf, np.float32)             # depth of best view per point
    world_h = np.hstack([xyz_world.astype(np.float32),
                         np.ones((N, 1), np.float32)])  # (N,4)

    ts = np.asarray(lidar_ts, np.int64)
    pa = np.asarray(poses, np.float64)                # (F,4,4) sensor→world
    o = np.argsort(ts); ts, pa = ts[o], pa[o]

    n_img = 0
    for ts_img, img in read_images(image_bag, camera, rot=image_rot):
        n_img += 1
        j = int(np.searchsorted(ts, ts_img)); j = min(max(j, 0), len(pa) - 1)
        if j > 0 and abs(ts[j - 1] - ts_img) < abs(ts[j] - ts_img):
            j -= 1
        P = pa[j]
        cam_c = P[:3, 3]                              # sensor origin in world (≈cam)
        near = ((xyz_world - cam_c) ** 2).sum(1) < max_range ** 2
        idxN = np.where(near)[0]
        if idxN.size == 0:
            continue

        # world → camera (= T ∘ inv(P)); keep points in front, then fisheye-project.
        cam = (T @ np.linalg.inv(P) @ world_h[idxN].T).T[:, :3]
        infront = cam[:, 2] > 0.05
        idxF = idxN[infront]; camf = cam[infront]
        if idxF.size == 0:
            continue
        p2d = cv2.fisheye.projectPoints(camf.reshape(-1, 1, 3),
                                        np.zeros((3, 1)), np.zeros((3, 1)), K, D)[0].squeeze(1)
        h, w = img.shape[:2]
        ix = np.round(p2d[:, 0]).astype(np.int32)
        iy = np.round(p2d[:, 1]).astype(np.int32)
        inb = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        idxF = idxF[inb]; depth = camf[inb, 2].astype(np.float32)
        ix, iy = ix[inb], iy[inb]
        if idxF.size == 0:
            continue

        # Depth buffer (downscaled) → reject points behind the nearest surface.
        wb = w // depth_ds + 1
        lin = (iy // depth_ds) * wb + (ix // depth_ds)
        dbuf = np.full((h // depth_ds + 1) * wb, np.inf, np.float32)
        np.minimum.at(dbuf, lin, depth)
        visible = depth <= dbuf[lin] * (1 + occl_tol) + 1e-3

        iv = idxF[visible]; dv = depth[visible]
        better = dv < best[iv]
        sb = better.nonzero()[0]
        if sb.size:
            tgt = iv[sb]
            rgb[tgt] = img[iy[visible][sb], ix[visible][sb]][:, ::-1]  # BGR→RGB
            best[tgt] = dv[sb]

    n_coloured = int((best < np.inf).sum())
    print(f"  multi-view colour: {n_img} frames, "
          f"{n_coloured:,}/{N:,} ({100*n_coloured/max(N,1):.1f}%) coloured", flush=True)
    return rgb, n_coloured


# ── PLY writer ───────────────────────────────────────────────────────

def write_ply(path: Path, xyz, rgb):
    """Write a binary little-endian PLY with x,y,z,red,green,blue."""
    import numpy as np

    n = len(xyz)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    data = np.zeros(n, dtype=[('x','f4'),('y','f4'),('z','f4'),
                               ('red','u1'),('green','u1'),('blue','u1')])
    data['x'] = xyz[:, 0].astype(np.float32)
    data['y'] = xyz[:, 1].astype(np.float32)
    data['z'] = xyz[:, 2].astype(np.float32)
    data['red']   = rgb[:, 0]
    data['green'] = rgb[:, 1]
    data['blue']  = rgb[:, 2]

    with open(path, 'wb') as f:
        f.write(header.encode())
        f.write(data.tobytes())


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lidar-bag",   required=True)
    parser.add_argument("--image-bag",   required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--params",      default=None)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--voxel-size",  type=float, default=0.05)
    parser.add_argument("--camera",      default="front")
    parser.add_argument("--image-rot",   default="ccw",
                        choices=["cw", "ccw", "180", "none"],
                        help="rotate camera frames to portrait to match K "
                             "(ccw verified best by multi-view colour consistency)")
    parser.add_argument("--max-range",   type=float, default=50.0,
                        help="KISS-ICP max LiDAR range (m)")
    parser.add_argument("--kiss-voxel",  type=float, default=None,
                        help="KISS-ICP map voxel size (m); default auto")
    parser.add_argument("--no-slam",     action="store_true",
                        help="skip KISS-ICP; gyro-only stacking (tripod scans)")
    parser.add_argument("--single-frame-colour", action="store_true",
                        help="colour each scan from only its time-nearest image "
                             "(old behaviour; default is multi-view)")
    parser.add_argument("--colour-range", type=float, default=20.0,
                        help="max camera→point distance for multi-view colouring (m)")
    args = parser.parse_args()

    _require("rosbags", "numpy", "cv2")

    import numpy as np

    lidar_bag = Path(args.lidar_bag)
    image_bag = Path(args.image_bag)
    output    = Path(args.output)
    calib_dir = Path(args.calibration) if args.calibration else None

    # Load calibration
    calib = None
    if calib_dir and calib_dir.exists():
        progress(5, "Loading calibration…")
        try:
            calib = load_calibration(calib_dir, args.camera)
            print(f"  Calibration loaded for camera '{args.camera}'", flush=True)
        except Exception as e:
            print(f"  Warning: could not load calibration — {e}", flush=True)
            print("  Proceeding without colour projection (XYZ only)", flush=True)

    # Read all LiDAR frames
    progress(10, "Reading LIDAR bag…")
    all_xyz = []
    frame_count = 0
    lidar_timestamps = []
    for ts, xyz in read_lidar_points(lidar_bag):
        all_xyz.append(xyz)
        lidar_timestamps.append(ts)
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"  LiDAR frames read: {frame_count}", flush=True)
    print(f"  Total LiDAR frames: {frame_count}", flush=True)

    if not all_xyz:
        print("ERROR: No LiDAR frames extracted", flush=True)
        sys.exit(1)

    # Recover each scan's full sensor→world pose so the frames stack into a
    # coherent scene.  KISS-ICP (scan-to-map ICP) gives rotation AND translation
    # — essential for handheld walk-throughs.  The IMU is still read, but only to
    # gravity-level the finished map (and as the fallback if SLAM is unavailable).
    progress(38, "Reading IMU…")
    imu_ts, imu_w, imu_a = read_imu(lidar_bag)
    L = _level_transform(imu_a)        # gravity → down (heading undetermined)

    poses = None
    if not args.no_slam:
        try:
            progress(40, "Registering scans with KISS-ICP…")
            raw_poses = register_with_kiss(
                all_xyz, max_range=args.max_range, voxel_size=args.kiss_voxel,
                progress_cb=lambda i, n: progress(
                    40 + int(15 * i / n), f"KISS-ICP registering {i}/{n}…"))
            poses = [L @ P for P in raw_poses]   # level the whole trajectory
            print(f"  Registered {frame_count} scans with KISS-ICP "
                  f"(full 6-DoF odometry), gravity-levelled", flush=True)
        except ImportError:
            print("  kiss-icp not installed — falling back to gyro-only "
                  "stacking (run: pip install kiss-icp)", flush=True)

    if poses is not None:
        all_xyz_world = [apply_pose(P, xyz) for P, xyz in zip(poses, all_xyz)]
        save_trajectory(output, lidar_timestamps, poses)
    elif imu_ts is not None and len(imu_ts) > 1:
        _, quats = integrate_orientation(imu_ts, imu_w, imu_a)
        Rs = frame_rotations(lidar_timestamps, imu_ts, quats)
        all_xyz_world = [(R @ xyz.T).T for R, xyz in zip(Rs, all_xyz)]
        print(f"  Registered {frame_count} frames using gyro orientation only "
              f"({len(imu_ts):,} IMU samples) — no translation", flush=True)
    else:
        all_xyz_world = all_xyz
        print("  No IMU found — stacking frames without registration", flush=True)

    # Geometry uses the world-frame points; colour projection below still uses
    # the per-frame sensor coordinates (the camera extrinsic is sensor-relative).
    combined_xyz = np.vstack(all_xyz_world)  # (N, 3)
    print(f"  Total points: {len(combined_xyz):,}", flush=True)
    progress(40, f"Read {len(combined_xyz):,} LiDAR points across {frame_count} frames")

    # Colour the points
    rgb = np.zeros((len(combined_xyz), 3), dtype=np.uint8)

    if calib is not None and poses is not None and not args.single_frame_colour:
        # Preferred: project every world point into *all* camera frames so points
        # the camera only saw later in the walk-through still get coloured.
        progress(45, "Multi-view colour projection…")
        rgb, _ = colour_points_multiview(
            combined_xyz, poses, lidar_timestamps, image_bag, calib,
            camera=args.camera, image_rot=args.image_rot,
            max_range=args.colour_range)
    elif calib is not None:
        progress(45, "Reading image bag and projecting…")

        # Precompute the start offset of each LiDAR frame in the flat rgb array
        offsets = []
        c = 0
        for xyz in all_xyz:
            offsets.append(c)
            c += len(xyz)

        # Sort LiDAR frames by timestamp so we can merge-join with the image stream
        order = sorted(range(len(lidar_timestamps)), key=lambda i: lidar_timestamps[i])

        # Stream images one at a time — keep only two frames (prev/curr) in memory
        # to bracket each LiDAR timestamp without loading the whole bag.
        img_gen = read_images(image_bag, args.camera, rot=args.image_rot)
        img_prev = None   # (ts, img) or None
        img_curr = None   # (ts, img) or None
        img_count = 0
        img_exhausted = False

        def _next_image():
            nonlocal img_exhausted, img_count
            if img_exhausted:
                return None
            try:
                ts, img = next(img_gen)
                img_count += 1
                return (ts, img)
            except StopIteration:
                img_exhausted = True
                return None

        img_curr = _next_image()

        for step, f_idx in enumerate(order):
            lidar_ts = lidar_timestamps[f_idx]
            xyz_chunk = all_xyz[f_idx]

            # Advance image stream until img_curr is at or past the lidar timestamp
            while img_curr is not None and img_curr[0] < lidar_ts:
                img_prev = img_curr
                img_curr = _next_image()

            # Pick the nearer of the two bracketing images
            if img_prev is None and img_curr is None:
                break
            elif img_prev is None:
                img = img_curr[1]
            elif img_curr is None:
                img = img_prev[1]
            else:
                img = (img_prev if abs(img_prev[0] - lidar_ts) <= abs(img_curr[0] - lidar_ts)
                       else img_curr)[1]

            mask, colours = project_and_colour(
                xyz_chunk, img, calib["K"], calib["D"], calib["T"]
            )
            start = offsets[f_idx]
            rgb[start:start + len(xyz_chunk)] = colours

            if step % 20 == 0:
                pct = 45 + int(45 * step / max(len(order), 1))
                progress(pct, f"Colouring frame {step+1}/{len(order)}…")

        print(f"  Image frames read: {img_count}", flush=True)
        if img_count == 0:
            print("  Warning: no image frames found — writing uncoloured point cloud", flush=True)
    else:
        print("  No calibration — writing uncoloured point cloud", flush=True)

    # Drop invalid (non-finite) returns before downsampling.  Keeping rgb
    # aligned with xyz; NaN points otherwise corrupt the voxel grid.
    finite = np.isfinite(combined_xyz).all(axis=1)
    dropped = int((~finite).sum())
    if dropped:
        combined_xyz = combined_xyz[finite]
        rgb = rgb[finite]
        print(f"  Dropped {dropped:,} invalid (NaN/Inf) points", flush=True)

    # Voxel downsample (simple grid approach without open3d)
    progress(90, f"Voxel downsampling (size={args.voxel_size} m)…")
    try:
        import open3d as o3d  # optional — used for efficient voxel downsampling
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
        pcd = pcd.voxel_down_sample(args.voxel_size)
        final_xyz = np.asarray(pcd.points).astype(np.float32)
        final_rgb = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    except ImportError:
        # Manual grid-based downsample
        voxel = args.voxel_size
        keys = np.floor(combined_xyz / voxel).astype(np.int32)
        _, unique_idx = np.unique(keys, axis=0, return_index=True)
        final_xyz = combined_xyz[unique_idx]
        final_rgb = rgb[unique_idx]

    print(f"  Points after downsample: {len(final_xyz):,}", flush=True)

    progress(95, f"Writing {len(final_xyz):,} points to {output.name}…")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_ply(output, final_xyz, final_rgb)

    progress(100, "Complete!")
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
