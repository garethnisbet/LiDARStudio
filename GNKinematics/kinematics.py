#---------------------------------------------------------------------------#
#                    Generalised Kinematics Library
#                      Created by Gareth Nisbet
#---------------------------------------------------------------------------#

import numpy as np

def vanglev(v1, v2):
    """Angle between two vectors (radians)."""
    # Clamp to arccos's domain: floating-point error can push the cosine just
    # past ±1 near singular configurations, which would yield NaN.
    return np.arccos(np.clip(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))


def vp_angle(v1, v2, v3):
    """Angle between a vector and the plane defined by v2 and v3 (radians)."""
    plane_normal = np.cross(v2, v3)
    return np.arccos(np.clip(
        np.dot(v1, plane_normal) / (np.linalg.norm(v1) * np.linalg.norm(plane_normal)),
        -1.0, 1.0))


def rotmat(u, angle):
    """Rotation matrix for clockwise rotation about axis u by angle (degrees)."""
    u = np.asarray(u, dtype=float).ravel()
    u = u / np.linalg.norm(u)
    ux, uy, uz = u
    c = np.cos(np.radians(angle))
    s = np.sin(np.radians(angle))
    return np.array([
        [ux*ux + (1 - ux*ux)*c, ux*uy*(1-c) - uz*s,    ux*uz*(1-c) + uy*s],
        [ux*uy*(1-c) + uz*s,    uy*uy + (1 - uy*uy)*c,  uy*uz*(1-c) - ux*s],
        [ux*uz*(1-c) - uy*s,    uy*uz*(1-c) + ux*s,     uz*uz + (1 - uz*uz)*c],
    ])


def rotxyz(v, u, angle):
    """Rotate vector v about axis u by angle (degrees)."""
    v = np.atleast_2d(v)
    return (rotmat(u, angle) @ v.T).T


def eulerMatrix(alpha, beta, gamma):
    """ZYX Euler rotation matrix from angles in degrees."""
    a, b, g = np.radians(alpha), np.radians(beta), np.radians(gamma)
    M_alpha = np.array([
        [ np.cos(a), np.sin(a), 0],
        [-np.sin(a), np.cos(a), 0],
        [         0,         0, 1],
    ])
    M_beta = np.array([
        [np.cos(b), 0, -np.sin(b)],
        [        0, 1,          0],
        [np.sin(b), 0,  np.cos(b)],
    ])
    M_gamma = np.array([
        [ np.cos(g), np.sin(g), 0],
        [-np.sin(g), np.cos(g), 0],
        [         0,         0, 1],
    ])
    return M_alpha @ M_beta @ M_gamma


def rotationMatrixToEulerZYX(rmatrix):
    """Convert rotation matrix to ZYX Euler angles (radians)."""
    sy = np.sqrt(rmatrix[2, 1]**2 + rmatrix[2, 2]**2)
    if sy >= 1e-6:
        alpha = -np.arctan2(rmatrix[2, 1], rmatrix[2, 2])
        beta = -np.arctan2(-rmatrix[2, 0], sy)
        gamma = -np.arctan2(rmatrix[1, 0], rmatrix[0, 0])
    else:
        alpha = np.pi - np.arctan2(rmatrix[0, 1], rmatrix[0, 2])
        beta = -np.arctan2(-rmatrix[2, 0], sy)
        gamma = -np.arctan2(rmatrix[1, 0], rmatrix[0, 0])
    return alpha, beta, gamma


def rotationMatrixToEulerZYZ_extrinsic(rmatrix):
    """Convert rotation matrix to ZYZ extrinsic Euler angles (radians)."""
    alpha = np.pi/2 - np.arctan2(rmatrix[2, 0], rmatrix[2, 1])
    beta = np.pi/2 - np.arccos(rmatrix[2, 2])
    gamma = -(np.arctan2(rmatrix[0, 2], rmatrix[1, 2]) + np.pi/2)
    return alpha, beta, gamma


def set_mu_eta_chi_phi(mu, eta, chi, phi):
    """Convert diffractometer angles to ZYX Euler angles (degrees)."""
    v = np.identity(3)
    rmatrix = (rotmat(v[2, :], phi) @ rotmat(v[1, :], -(90 - chi))
               @ rotmat(v[0, :], eta) @ rotmat(v[2, :], mu))
    rmatrix[np.abs(rmatrix) < 1e-5] = 0
    alpha, beta, gamma = rotationMatrixToEulerZYX(rmatrix)
    return np.degrees(alpha), np.degrees(beta), np.degrees(gamma)


class getMuEtaChiPhi:
    """Control for mu eta chi phi. Set constraint to 'mu' or 'eta'.
    Input angles in degrees."""

    def __init__(self, constraint):
        self.constraint = constraint

    def getValues(self, alpha, beta, gamma):
        v = np.identity(3)
        em = rotmat(v[:, 0], alpha) @ rotmat(v[:, 1], beta) @ rotmat(v[:, 2], gamma)
        R = np.array([em @ v[:, 0], em @ v[:, 1], em @ v[:, 2]])
        if self.constraint == 'mu':
            mu = 0
            eta = -alpha
            chi = 90 - beta
            phi = -gamma
        else:
            eta = 0
            if R[0, 2] != 0:
                alpha2, beta2, gamma2 = rotationMatrixToEulerZYZ_extrinsic(R)
                mu = np.degrees(alpha2)
                chi = np.degrees(beta2)
                phi = np.degrees(gamma2)
            else:
                mu = -alpha
                chi = 90 - beta
                phi = -gamma
        return mu, eta, chi, phi


def minanglediff(a, b):
    """Minimum angle difference in degrees (handles wrapping)."""
    return np.degrees(np.arccos(
        np.cos(np.radians(a)) * np.cos(np.radians(b)) +
        np.sin(np.radians(a)) * np.sin(np.radians(b))
    ))


class kinematics:
    """Forward and inverse kinematics for a 6-DOF robot arm
    with off-centre base rotation.

    Instantiate with:
        kin = kinematics(axis_vects, L_vects, motor_limits, motor_offsets,
                         centre_offset, tool_offset, strategy, weighting)
    """

    @classmethod
    def from_home_positions(cls, v0, v1, v2, v3, v4, motor_limits,
                            centre_offset, tool_offset, strategy, weighting=None):
        """Create kinematics from joint positions at the home (zero-angle) pose.

        Converts home-position geometry into the unrotated link vectors and
        motor offsets required by the IK solver.
        Assumes a 6-DOF arm with Z-Y-Y-roll-Y-Z joint axes.
        """
        v0, v1, v2, v3, v4 = (np.asarray(x, dtype=float) for x in (v0, v1, v2, v3, v4))
        L0 = v0
        L1_home = v1 - v0
        L2_home = v2 - v1
        L3_home = v3 - v2
        L4_home = v4 - v3

        # J2 offset: angle of L1 from vertical (about Y)
        offset_j2 = np.degrees(np.arctan2(L1_home[0], L1_home[2]))
        L1 = np.array([0, 0, np.linalg.norm(L1_home)])
        R_j2 = rotmat([0, 1, 0], -offset_j2)

        # J3 offset: angle of L2 from vertical (after undoing J2)
        L2_temp = R_j2 @ L2_home
        offset_j3 = np.degrees(np.arctan2(L2_temp[0], L2_temp[2]))
        L2 = np.array([0, 0, np.linalg.norm(L2_home)])
        R_j23 = rotmat([0, 1, 0], -offset_j3) @ R_j2

        # Un-rotate L3 by cumulative J2+J3 rotation
        L3 = R_j23 @ L3_home

        # J5 offset: angle of L4 from vertical (after undoing J2+J3)
        L4_temp = R_j23 @ L4_home
        offset_j5 = np.degrees(np.arctan2(L4_temp[0], L4_temp[2]))
        L4 = rotmat([0, 1, 0], -offset_j5) @ L4_temp

        vx = np.array([1, 0, 0])
        vy = np.array([0, 1, 0])
        vz = np.array([0, 0, 1])
        L_vects = np.array([L0, L1, L2, L3, L4, vx, vy, vz])
        axis_vects = np.array([vz, vy, vy, L3, vy, vz])
        motor_offsets = (0, offset_j2, offset_j3, 0, offset_j5, 0)

        return cls(axis_vects, L_vects, motor_limits, motor_offsets,
                   centre_offset, tool_offset, strategy, weighting)

    def __init__(self, axis_vects, L_vects, motor_limits, motor_offsets,
                 centre_offset, tool_offset, strategy, weighting=None):
        self.axis_vects = axis_vects
        self.L_vects = L_vects
        self.motor_limits = motor_limits
        self.motor_offsets = np.array(motor_offsets)
        self.centre_offset = np.array(centre_offset)
        self.tool_offset = tool_offset
        self.strategy = strategy
        self.weighting = weighting
        self.new_offset = [0, 0, 0]
        self.motor_pos = np.array([0, 0, 0, 0, 0, 0])

        # Compute the home EE rotation (all joints = 0) for euler convention
        home_angles = np.zeros(6) + self.motor_offsets
        v5h = self._rotate_through_joints_raw(L_vects[5, :], axis_vects, home_angles, 5)
        v6h = self._rotate_through_joints_raw(L_vects[6, :], axis_vects, home_angles, 5)
        v7h = self._rotate_through_joints_raw(L_vects[7, :], axis_vects, home_angles, 5)
        self._home_rot = np.concatenate((v5h, v6h, v7h), 0)
        self._home_rot_inv = self._home_rot.T  # orthogonal matrix: inverse = transpose

    def setStrategy(self, strategy):
        self.strategy = strategy

    def setWeighting(self, weighting):
        self.weighting = weighting

    def setBase_cut_off(self, base_cut_off):
        self.base_cut_off = base_cut_off

    def setLimits(self, motor_limits):
        self.motor_limits = motor_limits

    def setToolOffset(self, tool_offset):
        self.tool_offset = tool_offset

    def setCentreOffset(self, centre_offset):
        self.centre_offset = centre_offset

    def setMotorOffset(self, motor_offsets):
        self.motor_offsets = motor_offsets

    def storeCurrentPosition(self, set_pos):
        self.motor_pos = set_pos

    @staticmethod
    def _rotate_through_joints_raw(vec, axis_vects, angles, from_joint):
        """Rotate vec through a joint chain (static version for use during init)."""
        v = np.atleast_2d(vec)
        for j in range(from_joint, -1, -1):
            v = rotxyz(v, axis_vects[j, :], angles[j])
        return v

    def _rotate_through_joints(self, vec, angles, from_joint):
        """Rotate vec through the joint chain from from_joint down to joint 0."""
        return self._rotate_through_joints_raw(vec, self.axis_vects, angles, from_joint)

    def setEulerTarget(self, xyz_or_six, r_alpha=None, r_beta=None, r_gamma=None):
        if r_alpha is None:
            xyz_or_six = list(xyz_or_six)
            xyz_or_six, r_alpha, r_beta, r_gamma = xyz_or_six[:3], xyz_or_six[3], xyz_or_six[4], xyz_or_six[5]
        xyz = xyz_or_six
        vx, vy, vz = np.identity(3)
        # World-frame ZYX Euler (identity = tool frame aligned with world).
        em = rotmat(vx, r_alpha) @ rotmat(vy, r_beta) @ rotmat(vz, r_gamma)
        targetmatrix = np.array([em @ vx, em @ vy, em @ vz])
        tool = np.asarray(self.tool_offset) @ targetmatrix
        xyz = np.array(xyz) + tool
        tv1 = (vx @ targetmatrix).tolist()
        tv2 = (vy @ targetmatrix).tolist()
        tv3 = (vz @ targetmatrix).tolist()
        return self.i_kinematics(np.array([xyz, tv1, tv2, tv3]))

    def f_kinematics(self, inputs):
        angles = inputs + self.motor_offsets

        self.v0 = self._rotate_through_joints(self.L_vects[0, :], angles, 0)
        self.v1 = self.v0 + self._rotate_through_joints(self.L_vects[1, :], angles, 1)
        self.v2 = self.v1 + self._rotate_through_joints(self.L_vects[2, :], angles, 2)
        self.v3 = self.v2 + self._rotate_through_joints(self.L_vects[3, :], angles, 3)
        self.v4 = self.v3 + self._rotate_through_joints(self.L_vects[4, :], angles, 4)
        self.v5 = self._rotate_through_joints(self.L_vects[5, :], angles, 5)
        self.v6 = self._rotate_through_joints(self.L_vects[6, :], angles, 5)
        self.v7 = self._rotate_through_joints(self.L_vects[7, :], angles, 5)
        self._t_off = self._rotate_through_joints(self.tool_offset, angles, 5)
        new_centre = rotxyz(np.atleast_2d(self.centre_offset), self.axis_vects[0, :], angles[0])

        # World-frame ZYX Euler angles of the tool frame.
        rmatrix = np.concatenate((self.v5, self.v6, self.v7), 0)
        rmatrix[np.abs(rmatrix) < 1e-5] = 0
        alpha, beta, gamma = rotationMatrixToEulerZYX(rmatrix)
        self.al_be_gam = np.array([[alpha, beta, gamma]])
        self.position = self.v4 - self._t_off + new_centre

        return np.concatenate((self.v0, self.v1, self.v2, self.v3, self.v4,
                               self.v5, self.v6, self.v7, self.position,
                               self.al_be_gam), 0)

    def i_kinematics(self, target):
        self.target = target
        L_vects = np.copy(self.L_vects)
        L_vects[:3, 0] = 0
        L1 = np.linalg.norm(L_vects[1, :])
        L2 = np.linalg.norm(L_vects[2, :] + L_vects[3, :])
        v0 = target[0, :]
        v1 = target[1, :]
        v3 = target[3, :]

        vlength = np.linalg.norm(L_vects[4, :])
        vc1 = (v0 - v3 / np.linalg.norm(v3) * vlength) - L_vects[0, :]

        # Angle/length calculations for base offset geometry
        vp = vp_angle(vc1, [1, 0, 0], [0, 1, 0])
        A1 = np.pi/2 + vp
        A2 = np.pi/2 - vp
        A1n = 1.5 * np.pi - vp
        A2n = vp - np.pi/2

        b = np.linalg.norm(vc1)
        c = np.linalg.norm([self.L_vects[0, :][0], self.L_vects[0, :][1], 0])
        a1 = np.sqrt(b**2 + c**2 - 2*b*c*np.cos(A1))
        a2 = np.sqrt(b**2 + c**2 - 2*b*c*np.cos(A2))
        a1n = np.sqrt(b**2 + c**2 - 2*b*c*np.cos(A1n))
        a2n = np.sqrt(b**2 + c**2 - 2*b*c*np.cos(A2n))
        tc_offset1 = np.arccos(np.clip((c**2 - a1**2 - b**2) / (-2*a1*b), -1.0, 1.0))
        tc_offset2 = np.arccos(np.clip((c**2 - a2**2 - b**2) / (-2*a2*b), -1.0, 1.0))
        tc_offset1n = np.arccos(np.clip((c**2 - a1n**2 - b**2) / (-2*a1n*b), -1.0, 1.0))
        tc_offset2n = np.arccos(np.clip((c**2 - a2n**2 - b**2) / (-2*a2n*b), -1.0, 1.0))

        theta0check = np.arctan2(vc1[1], vc1[0])
        L23_vp = vp_angle(L_vects[3, :] + L_vects[2, :], [1, 0, 0], [0, 1, 0])
        vc1_vp = vp_angle(vc1, [1, 0, 0], [0, 1, 0])

        NUM_CANDIDATES = 8
        solutions = np.empty((0, 6))

        for ii in range(NUM_CANDIDATES):
            # Compute theta0, theta1, theta2 for each candidate configuration
            if ii in (0, 4):
                if vc1[-1] > 0:
                    vc1n, tc_offset = a2, tc_offset2
                else:
                    vc1n, tc_offset = a2n, -tc_offset2n
                theta0 = theta0check
                theta1 = np.arccos(np.clip((L1**2 + vc1n**2 - L2**2) / (2*L1*vc1n), -1.0, 1.0)) + tc_offset
                theta2 = np.pi - np.arccos(np.clip((L1**2 + L2**2 - vc1n**2) / (2*L1*L2), -1.0, 1.0))
                theta2 -= L23_vp
                theta1 = -theta1 + vc1_vp

            elif ii in (1, 5):
                if vc1[-1] > 0:
                    vc1n, tc_offset = a1, tc_offset1
                else:
                    vc1n, tc_offset = a1n, -tc_offset1n
                theta0 = theta0check + np.pi
                theta1 = np.arccos(np.clip((L1**2 + vc1n**2 - L2**2) / (2*L1*vc1n), -1.0, 1.0)) + tc_offset
                theta2 = np.pi - np.arccos(np.clip((L1**2 + L2**2 - vc1n**2) / (2*L1*L2), -1.0, 1.0))
                theta2 -= L23_vp
                theta1 = -theta1 - vc1_vp

            elif ii in (2, 6):
                if vc1[-1] > 0:
                    vc1n, tc_offset = a2, tc_offset2
                else:
                    vc1n, tc_offset = a2n, -tc_offset2n
                theta0 = theta0check
                theta1 = -np.arccos(np.clip((L1**2 + vc1n**2 - L2**2) / (2*L1*vc1n), -1.0, 1.0)) + tc_offset
                theta2 = -(np.pi - np.arccos(np.clip((L1**2 + L2**2 - vc1n**2) / (2*L1*L2), -1.0, 1.0)))
                theta2 -= L23_vp
                theta1 = -theta1 + vc1_vp

            elif ii in (3, 7):
                if vc1[-1] > 0:
                    vc1n, tc_offset = a1, tc_offset1
                else:
                    vc1n, tc_offset = a1n, -tc_offset1n
                theta0 = theta0check + np.pi
                theta1 = -np.arccos(np.clip((L1**2 + vc1n**2 - L2**2) / (2*L1*vc1n), -1.0, 1.0)) + tc_offset
                theta2 = -(np.pi - np.arccos(np.clip((L1**2 + L2**2 - vc1n**2) / (2*L1*L2), -1.0, 1.0)))
                theta2 -= L23_vp
                theta1 = -theta1 - vc1_vp

            # Solve wrist angles (theta3, theta4, theta5)
            degs = [np.degrees(theta0), np.degrees(theta1), np.degrees(theta2),
                    0, 0, 0]

            vec3 = self._rotate_through_joints(L_vects[3, :], degs, 2)
            av3 = self._rotate_through_joints(self.axis_vects[4, :], degs, 2)

            av3_dot_v3 = np.dot(av3[0], v3)
            if np.abs(av3_dot_v3) > 0.0001:
                theta3i = vp_angle(av3[0], v3, vec3[0])
                if ii < 4:
                    theta3 = vp_angle(av3[0], v3, vec3[0])
                elif theta3i > np.pi/2:
                    theta3 = -vp_angle(av3[0], vec3[0], v3)
            else:
                theta3 = 0

            theta3 = -np.sign(av3_dot_v3) * theta3
            degs[3] = np.degrees(theta3)

            vec4 = self._rotate_through_joints(L_vects[4, :], degs, 3)
            theta4 = vanglev(v3, vec4[0])
            degs[4] = np.degrees(theta4)

            vec5 = self._rotate_through_joints(L_vects[5, :], degs, 4)
            if np.abs(vanglev(v3, vec5[0]) - np.pi/2) > 0.0001:
                theta4 = -vanglev(v3, vec4[0])
                degs[4] = np.degrees(theta4)

            vec5 = self._rotate_through_joints(L_vects[5, :], degs, 4)
            theta5 = -vanglev(v1, vec5[0])
            degs[5] = np.degrees(theta5)

            vec5 = self._rotate_through_joints(L_vects[5, :], degs, 5)
            if np.abs(vanglev(v1, vec5[0])) > 0.0001:
                theta5 = -theta5
                degs[5] = np.degrees(theta5)

            vec5 = self._rotate_through_joints(L_vects[5, :], degs, 5)
            if np.abs(vanglev(v1, vec5[0])) > 0.0001:
                theta5 = vanglev(v1, vec5[0]) + np.pi

            if np.isnan(theta5):
                theta5 = 0

            # Normalize angles to [-180, 180] degrees
            thetas = np.array([theta0, theta1, theta2, theta3, theta4, theta5])
            output = np.degrees(np.mod(thetas + np.pi, 2*np.pi) - np.pi)
            solutions = np.vstack([solutions, output])

        solutions -= self.motor_offsets

        # Filter solutions within motor limits
        within_limits = (
            np.all(solutions >= self.motor_limits[:, 0], axis=1) &
            np.all(solutions <= self.motor_limits[:, 1], axis=1)
        )
        valid_solutions = solutions[within_limits]

        # Forward-kinematics consistency check on EVERY candidate, not just the
        # one the strategy would pick. The analytic solver emits spurious roots
        # that pass the limit filter but do not actually reach the target; if
        # the strategy (e.g. minimum_movement) happens to select such a root,
        # checking only that pick makes the whole solve return NaN even though a
        # genuine solution exists — a seed-dependent false "no solution". Drop
        # the spurious roots first, then let the strategy choose among the ones
        # that truly reach the target.
        if valid_solutions.shape[0] >= 1:
            # Use sum-of-absolute deviations (not abs-of-sum, which can cancel
            # and hide a wrong pose) and a physically-meaningful tolerance. The
            # analytic solve loses a little precision near orientation
            # singularities (residual ~0.02 mm — negligible), whereas spurious
            # roots miss the target by tens of mm; 1e-4 wrongly rejected the
            # good near-singular solutions.
            reaches = np.array([
                np.any(np.sum(np.abs(self.f_kinematics(cand) - self.target[0, :]),
                              axis=1) < 1.0)
                for cand in valid_solutions
            ])
            valid_solutions = valid_solutions[reaches]

        if valid_solutions.shape[0] < 1:
            best_solution = np.full(6, np.nan)
        elif self.strategy == 'minimum_movement':
            cost = np.sum(np.abs(valid_solutions - self.motor_pos), axis=1)
            best_solution = valid_solutions[np.argmin(cost)]
        elif self.strategy == 'minimum_movement_weighted':
            cost = np.sum(
                np.abs(valid_solutions - self.motor_pos) * self.weighting, axis=1)
            best_solution = valid_solutions[np.argmin(cost)]
        elif self.strategy == 'comfortable_limits':
            limit_centres = np.mean(self.motor_limits, axis=1)
            cost = np.sum(np.abs(limit_centres - valid_solutions), axis=1)
            best_solution = valid_solutions[np.argmin(cost)]
        else:
            best_solution = valid_solutions[0]

        return best_solution
