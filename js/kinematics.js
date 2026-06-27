// ============================================================
// js/kinematics.js — pure math: FK, IK, kappa geometry
// ============================================================
import * as THREE from 'three';
import * as State from './state.js';

const deg2rad = Math.PI / 180;
const rad2deg = 180 / Math.PI;

// ============================================================
// Per-device FK/IK functions
// ============================================================

export function updateFK(dev) {
  for (let i = 0; i < dev.numJoints; i++) {
    dev.jointRotGroups[i].quaternion.setFromAxisAngle(dev.jointAxes[i], dev.jointAngles[i]);
  }
  updateVirtualAngles(dev);
}

export function getEEWorldPosition(dev) {
  dev.eeMarker.updateWorldMatrix(true, false);
  return new THREE.Vector3().setFromMatrixPosition(dev.eeMarker.matrixWorld);
}

export function getEEWorldQuaternion(dev) {
  dev.eeMarker.updateWorldMatrix(true, false);
  return new THREE.Quaternion().setFromRotationMatrix(dev.eeMarker.matrixWorld);
}

export function getJointWorldAxis(dev, i) {
  const worldQuat = new THREE.Quaternion();
  dev.jointRestGroups[i].getWorldQuaternion(worldQuat);
  return dev.jointAxes[i].clone().applyQuaternion(worldQuat).normalize();
}

export function getJointWorldPosition(dev, i) {
  dev.jointRestGroups[i].updateWorldMatrix(true, false);
  return new THREE.Vector3().setFromMatrixPosition(dev.jointRestGroups[i].matrixWorld);
}

export function clampJoints(dev) {
  for (let i = 0; i < dev.numJoints; i++) {
    dev.jointAngles[i] = Math.max(dev.jointLimits[i][0], Math.min(dev.jointLimits[i][1], dev.jointAngles[i]));
  }
}

// ============================================================
// Kappa geometry functions
// ============================================================

export function kappaToEuler(dev, kappaDeg) {
  const kappaRad = kappaDeg * deg2rad;
  const chi = 2 * Math.asin(Math.sin(kappaRad / 2) * Math.sin(dev.kappaAlpha));
  return { chi: chi * rad2deg, kappa: kappaDeg };
}

export function eulerToKappa(dev, chiDeg) {
  const chiRad = chiDeg * deg2rad;
  const sinHalfChi = Math.sin(chiRad / 2);
  const sinAlpha = Math.sin(dev.kappaAlpha);
  if (Math.abs(sinHalfChi / sinAlpha) > 1) return null;
  const kappaRad = 2 * Math.asin(sinHalfChi / sinAlpha);
  return { kappa: kappaRad * rad2deg, chi: chiDeg };
}

export function getCompensation(dev, kappaDeg) {
  const halfK = kappaDeg * deg2rad / 2;
  const delta = Math.atan2(Math.cos(dev.kappaAlpha) * Math.sin(halfK), Math.cos(halfK)) * rad2deg;
  return { theta: -delta, phi: -delta * dev.kappaPhiSign };
}

export function updateVirtualAngles(dev) {
  if (!dev.isKappaGeometry || !dev.kappaAlpha) return;
  if (dev !== State.activeDevice) return;
  const sign = dev.kappaSignPositive ? 1 : -1;
  const kappaDeg = sign * dev.jointAngles[dev.kappaJointIdx] * rad2deg;
  const chiDeg = kappaToEuler(dev, kappaDeg).chi;
  const comp = getCompensation(dev, kappaDeg);
  let thetaDeg = sign * dev.kappaThetaSign * dev.jointAngles[dev.thetaJointIdx] * rad2deg - comp.theta + 90;
  let phiDeg = sign * dev.kappaThetaSign * dev.jointAngles[dev.phiJointIdx] * rad2deg - comp.phi + 90;

  const snap = v => Math.abs(v) < 0.05 ? 0 : v;
  const chiSlider = document.getElementById('chiSlider');
  if (chiSlider) {
    chiSlider.value = -chiDeg;
    document.getElementById('vchi').textContent = snap(-chiDeg).toFixed(1);
    document.getElementById('vthetaSlider').value = thetaDeg;
    document.getElementById('vvtheta').textContent = snap(thetaDeg).toFixed(1);
    document.getElementById('vphiSlider').value = phiDeg;
    document.getElementById('vvphi').textContent = snap(phiDeg).toFixed(1);
  }
}

export function updateChain(dev) {
  State.scene.updateMatrixWorld(true);
  for (let i = 0; i < dev.numJoints; i++) {
    const p = getJointWorldPosition(dev, i);
    dev.chainPts[i * 3]     = p.x;
    dev.chainPts[i * 3 + 1] = p.y;
    dev.chainPts[i * 3 + 2] = p.z;
    dev.chainSpheres[i].position.copy(p);
  }
  const ee = getEEWorldPosition(dev);
  const eeIdx = dev.numJoints * 3;
  dev.chainPts[eeIdx] = ee.x; dev.chainPts[eeIdx + 1] = ee.y; dev.chainPts[eeIdx + 2] = ee.z;
  dev.chainSpheres[dev.numJoints].position.copy(ee);
  dev.chainLineGeo.attributes.position.needsUpdate = true;
}

// ============================================================
// IK solver
// ============================================================

export function orientationError(qTarget, qCurrent) {
  const qErr = qTarget.clone().multiply(qCurrent.clone().conjugate());
  if (qErr.w < 0) { qErr.x = -qErr.x; qErr.y = -qErr.y; qErr.z = -qErr.z; qErr.w = -qErr.w; }
  const sinHalf = Math.sqrt(qErr.x*qErr.x + qErr.y*qErr.y + qErr.z*qErr.z);
  if (sinHalf < 1e-8) return new THREE.Vector3(0, 0, 0);
  const angle = 2 * Math.atan2(sinHalf, qErr.w);
  return new THREE.Vector3(qErr.x/sinHalf * angle, qErr.y/sinHalf * angle, qErr.z/sinHalf * angle);
}

export function solveIK(dev, targetPos, targetQuat, maxIter = 20, tolerance = 0.0005) {
  const lambda = 0.5;
  const oriWeight = 0.3;

  const movable = [];
  for (let i = 0; i < dev.numJoints; i++) {
    if (!dev.jointFixed || !dev.jointFixed[i]) movable.push(i);
  }
  const nMov = movable.length;

  for (let iter = 0; iter < maxIter; iter++) {
    updateFK(dev);
    State.scene.updateMatrixWorld(true);

    const eePos = getEEWorldPosition(dev);
    const eeQuat = getEEWorldQuaternion(dev);

    const posErr = new THREE.Vector3().subVectors(targetPos, eePos);
    const oriErr = orientationError(targetQuat, eeQuat);
    const posNorm = posErr.length();

    if (posNorm < tolerance && oriErr.length() < 0.01) return posNorm;

    const J = Array.from({length: 6}, () => Array(nMov).fill(0));
    for (let m = 0; m < nMov; m++) {
      const j = movable[m];
      const axis = getJointWorldAxis(dev, j);
      const jPos = getJointWorldPosition(dev, j);
      const r = new THREE.Vector3().subVectors(eePos, jPos);
      const lin = new THREE.Vector3().crossVectors(axis, r);
      J[0][m] = lin.x;  J[1][m] = lin.y;  J[2][m] = lin.z;
      J[3][m] = axis.x * oriWeight;
      J[4][m] = axis.y * oriWeight;
      J[5][m] = axis.z * oriWeight;
    }

    const e = [
      posErr.x, posErr.y, posErr.z,
      oriErr.x * oriWeight, oriErr.y * oriWeight, oriErr.z * oriWeight
    ];

    const n = 6;
    const JJT = Array.from({length: n}, () => Array(n).fill(0));
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        let sum = 0;
        for (let k = 0; k < nMov; k++) sum += J[r][k] * J[c][k];
        JJT[r][c] = sum;
      }
    }

    const lam2 = lambda * lambda;
    for (let i = 0; i < n; i++) JJT[i][i] += lam2;

    const inv = invertNxN(JJT, n);
    if (!inv) continue;

    const v = Array(n).fill(0);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        v[i] += inv[i][j] * e[j];

    for (let m = 0; m < nMov; m++) {
      let dq = 0;
      for (let j = 0; j < n; j++) dq += J[j][m] * v[j];
      dev.jointAngles[movable[m]] += dq;
    }

    clampJoints(dev);
  }

  updateFK(dev);
  State.scene.updateMatrixWorld(true);
  return getEEWorldPosition(dev).distanceTo(targetPos);
}

export function invertNxN(m, n) {
  const aug = Array.from({length: n}, (_, i) => {
    const row = Array(2 * n).fill(0);
    for (let j = 0; j < n; j++) row[j] = m[i][j];
    row[n + i] = 1;
    return row;
  });
  for (let col = 0; col < n; col++) {
    let maxRow = col, maxVal = Math.abs(aug[col][col]);
    for (let row = col + 1; row < n; row++) {
      const v = Math.abs(aug[row][col]);
      if (v > maxVal) { maxVal = v; maxRow = row; }
    }
    if (maxVal < 1e-12) return null;
    [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];
    const pivot = aug[col][col];
    for (let j = 0; j < 2 * n; j++) aug[col][j] /= pivot;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const f = aug[row][col];
      for (let j = 0; j < 2 * n; j++) aug[row][j] -= f * aug[col][j];
    }
  }
  return aug.map(row => row.slice(n));
}

// ============================================================
// Python-compatible world-frame ZYX Euler of the tool frame.
//
// The Three.js viewer is Y-up; the Python kinematics are Z-up. The proper
// rotation taking JS-world axes to Python-world axes is W = Rx(90°) (so
// JS-Y maps to Py-Z). A rotation matrix R expressed in JS-world therefore
// converts to Python-world via the similarity R_py = W · R · W^T. This
// conjugation preserves the rotation angle, unlike the naive Y↔Z basis
// swap (which is a reflection, det = -1).
//
// At home, Python's em_home_py depends on the robot's geometry; for the
// arms currently supported it's Ry(90°) (tool along +X). The home value
// is hardcoded here; if a future robot needs a different home convention
// it can be made per-device.
//
// These helpers match Python's GNKinematics.rotationMatrixToEulerZYX, so
// the viewer's displayed (alpha, beta, gamma) agrees with al_be_gam from
// f_kinematics — including the canonical branch at beta = ±90° gimbal lock.
// ============================================================

function _jsToPyBasis(r) {
  // W · r · W^T with W = Rx(90°). Converts a rotation matrix from the
  // Three.js Y-up world basis to the Python Z-up world basis.
  return [
    [ r[0][0], -r[0][2],  r[0][1]],
    [-r[2][0],  r[2][2], -r[2][1]],
    [ r[1][0], -r[1][2],  r[1][1]],
  ];
}

function _pyToJsBasis(r) {
  // Inverse of _jsToPyBasis: W^T · r · W with W = Rx(90°).
  return [
    [ r[0][0],  r[0][2], -r[0][1]],
    [ r[2][0],  r[2][2], -r[2][1]],
    [-r[1][0], -r[1][2],  r[1][1]],
  ];
}

function _matFromQuat(q) {
  const m = new THREE.Matrix4().makeRotationFromQuaternion(q);
  const e = m.elements; // column-major
  return [
    [e[0], e[4], e[8]],
    [e[1], e[5], e[9]],
    [e[2], e[6], e[10]],
  ];
}

function _matMul(a, b) {
  const r = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      r[i][j] = a[i][0]*b[0][j] + a[i][1]*b[1][j] + a[i][2]*b[2][j];
  return r;
}

function _quatFromMat(m) {
  const m4 = new THREE.Matrix4().set(
    m[0][0], m[0][1], m[0][2], 0,
    m[1][0], m[1][1], m[1][2], 0,
    m[2][0], m[2][1], m[2][2], 0,
    0, 0, 0, 1,
  );
  return new THREE.Quaternion().setFromRotationMatrix(m4);
}

export function pyEulerFromRelQuat(relQuat) {
  const rThree = _matFromQuat(relQuat);
  const rDeltaPy = _jsToPyBasis(rThree);
  const ry90 = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]];
  const em = _matMul(rDeltaPy, ry90);
  // rmatrix = em^T, with Python's noise clip (|v| < 1e-5 → 0) so the
  // singular branch produces a deterministic (alpha, 90°, 0) at gimbal lock.
  const clip = (v) => (Math.abs(v) < 1e-5 ? 0 : v);
  const r00 = clip(em[0][0]), r01 = clip(em[1][0]), r02 = clip(em[2][0]);
  const r10 = clip(em[0][1]);
  const r20 = clip(em[0][2]), r21 = clip(em[1][2]), r22 = clip(em[2][2]);
  const sy = Math.hypot(r21, r22);
  let alpha, beta, gamma;
  if (sy >= 1e-6) {
    alpha = -Math.atan2(r21, r22);
    beta  = -Math.atan2(-r20, sy);
    gamma = -Math.atan2(r10, r00);
  } else {
    alpha = Math.atan2(clip(em[2][1]), clip(em[1][1]));
    beta  = -Math.atan2(-r20, sy);
    gamma = 0;
  }
  while (alpha >  Math.PI) alpha -= 2 * Math.PI;
  while (alpha <= -Math.PI) alpha += 2 * Math.PI;
  return [alpha * rad2deg, beta * rad2deg, gamma * rad2deg];
}

export function relQuatFromPyEuler(alphaDeg, betaDeg, gammaDeg) {
  const Rx = new THREE.Matrix4().makeRotationX(alphaDeg * deg2rad);
  const Ry = new THREE.Matrix4().makeRotationY(betaDeg  * deg2rad);
  const Rz = new THREE.Matrix4().makeRotationZ(gammaDeg * deg2rad);
  const em4 = Rx.multiply(Ry).multiply(Rz);
  const ee = em4.elements;
  const em = [
    [ee[0], ee[4], ee[8]],
    [ee[1], ee[5], ee[9]],
    [ee[2], ee[6], ee[10]],
  ];
  const ryNeg90 = [[0, 0, -1], [0, 1, 0], [1, 0, 0]];
  const rDeltaPy = _matMul(em, ryNeg90);
  const rThree = _pyToJsBasis(rDeltaPy);
  return _quatFromMat(rThree);
}
