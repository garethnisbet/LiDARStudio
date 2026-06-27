// ============================================================
// js/hexapod.js — Stewart platform / hexapod loader and IK
// ============================================================
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { MeshBVH } from 'three-mesh-bvh';
import * as State from './state.js';

const deg2rad = Math.PI / 180;

// ============================================================
// loadHexapod
// ============================================================
export async function loadHexapod(configFile) {
  const config = await fetch(configFile + '?_=' + Date.now()).then(r => r.json());
  const id = State.incrementDeviceId();

  const rootGroup = new THREE.Group();
  rootGroup.name = config.name + '_root';
  State.scene.add(rootGroup);

  const platformRestPos = new THREE.Vector3(...config.platform.restPosition);

  const platformGroup = new THREE.Group();
  platformGroup.name = 'platform';
  platformGroup.position.copy(platformRestPos);
  rootGroup.add(platformGroup);

  const legData = config.legs.map((leg, i) => {
    const basePivot = new THREE.Vector3(...leg.basePivot);
    const platformPivotLocal = new THREE.Vector3(...leg.platformPivotLocal);

    const lowerPivot = new THREE.Group();
    lowerPivot.name = `leg${i}_lower_pivot`;
    lowerPivot.position.copy(basePivot);
    rootGroup.add(lowerPivot);

    const upperPivotRestPos = new THREE.Vector3().copy(platformPivotLocal).add(platformRestPos);
    const upperPivot = new THREE.Group();
    upperPivot.name = `leg${i}_upper_pivot`;
    upperPivot.position.copy(upperPivotRestPos);
    rootGroup.add(upperPivot);

    return {
      basePivot,
      platformPivotLocal,
      lowerPivotGroup: lowerPivot,
      upperPivotGroup: upperPivot,
      upperPivotRestPos,
      restLowerDir: new THREE.Vector3().subVectors(upperPivotRestPos, basePivot).normalize(),
      restUpperDir: new THREE.Vector3().subVectors(basePivot, upperPivotRestPos).normalize(),
      lowerMeshName: leg.lowerMesh,
      upperMeshName: leg.upperMesh,
    };
  });

  const originHelpers = [];
  const originLabels = [];
  {
    const axes = new THREE.AxesHelper(0.04);
    axes.renderOrder = 2;
    axes.material.depthTest = false;
    axes.visible = State.originsOn;
    rootGroup.add(axes);
    originHelpers.push(axes);

    const div = document.createElement('div');
    div.className = 'origin-label';
    div.textContent = '';
    const lbl = new CSS2DObject(div);
    lbl.position.set(0, 0.048, 0);
    lbl.visible = State.originsOn;
    rootGroup.add(lbl);
    originLabels.push(lbl);
  }

  const dummyGeo = new THREE.BufferGeometry();
  const dummyMat = new THREE.LineBasicMaterial();

  const dev = {
    id,
    config,
    configFile,
    name: config.name,
    type: 'hexapod',
    rootGroup,
    platformGroup,
    platformRestPos,
    legData,
    platformPose: [0, 0, 0, 0, 0, 0],
    numJoints: 0,
    jointAngles: [],
    jointLimits: [],
    jointFixed: [],
    jointRestGroups: [],
    jointRotGroups: [],
    jointAxes: [],
    apiSign: [],
    sliderJointMap: [],
    kappaSliderNames: {},
    linkToJoint: {},
    parentGroups: {},
    isBranching: true,
    isKappaGeometry: false,
    eeMarker: new THREE.Group(),
    meshLabels: [],
    robotLinkMeshes: [],
    staticMeshes: [],
    originHelpers,
    originLabels,
    chainVisible: false,
    chainLine: new THREE.LineSegments(dummyGeo, dummyMat),
    chainSpheres: [],
    chainPts: new Float32Array(0),
    chainLineGeo: dummyGeo,
    ikMode: false,
    ikTarget: new THREE.Mesh(),
    ikTargetQuat: new THREE.Quaternion(),
    ikTargetEuler: new THREE.Euler(),
    ikLine: new THREE.Line(dummyGeo, dummyMat),
    adjPairs: new Set(),
    parentLink: null,
    loaded: false,
  };

  await new Promise((resolve, reject) => {
    const loader = new GLTFLoader();
    loader.load(config.model, (gltf) => {
      const model = gltf.scene;
      const allNodes = {};
      model.traverse((child) => {
        if (child.name) allNodes[child.name] = child;
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
        }
      });
      console.log(`[${config.name}] glTF nodes:`, Object.keys(allNodes));

      function findNode(name) {
        return allNodes[name] || allNodes[name.replace(/\./g, '_')] || allNodes[name.replace(/\./g, '')] || null;
      }

      function createLabel(text, parent) {
        const div = document.createElement('div');
        div.className = 'mesh-label';
        div.textContent = text;
        const label = new CSS2DObject(div);
        label.visible = false;
        const box = new THREE.Box3().setFromObject(parent);
        const center = box.getCenter(new THREE.Vector3());
        parent.worldToLocal(center);
        label.position.copy(center);
        parent.add(label);
        dev.meshLabels.push(label);
      }

      function reparentTo(node, targetGroup) {
        node.updateWorldMatrix(true, false);
        const wm = node.matrixWorld.clone();
        node.removeFromParent();
        targetGroup.updateWorldMatrix(true, false);
        const localMat = targetGroup.matrixWorld.clone().invert().multiply(wm);
        localMat.decompose(node.position, node.quaternion, node.scale);
        targetGroup.add(node);
      }

      const baseMesh = findNode(config.base.mesh);
      if (baseMesh) {
        reparentTo(baseMesh, rootGroup);
        baseMesh.userData.deviceId = dev.id;
        baseMesh.traverse(c => {
          if (c.isMesh) {
            c.geometry.boundsTree = new MeshBVH(c.geometry);
            c.userData.deviceId = dev.id;
          }
        });
        dev.robotLinkMeshes.push({ name: 'BasePlate', meshes: [baseMesh], jointIdx: -1 });
        createLabel('Base Plate', baseMesh);
      }

      const topMesh = findNode(config.platform.mesh);
      if (topMesh) {
        reparentTo(topMesh, platformGroup);
        topMesh.userData.deviceId = dev.id;
        topMesh.traverse(c => {
          if (c.isMesh) {
            c.geometry.boundsTree = new MeshBVH(c.geometry);
            c.userData.deviceId = dev.id;
          }
        });
        dev.robotLinkMeshes.push({ name: 'TopPlate', meshes: [topMesh], jointIdx: -1 });
        createLabel('Top Plate', topMesh);
      }

      for (let i = 0; i < legData.length; i++) {
        const leg = legData[i];

        const lowerNode = findNode(leg.lowerMeshName);
        if (lowerNode) {
          reparentTo(lowerNode, leg.lowerPivotGroup);
          lowerNode.userData.deviceId = dev.id;
          const meshes = [];
          lowerNode.traverse(c => {
            if (c.isMesh) {
              c.geometry.boundsTree = new MeshBVH(c.geometry);
              c.userData.deviceId = dev.id;
              meshes.push(c);
            }
          });
          dev.robotLinkMeshes.push({ name: leg.lowerMeshName, meshes, jointIdx: -1 });
          createLabel(`Leg ${i + 1} Lower`, lowerNode);
        } else {
          console.warn(`[Hexapod] Lower mesh "${leg.lowerMeshName}" not found`);
        }

        const upperNode = findNode(leg.upperMeshName);
        if (upperNode) {
          reparentTo(upperNode, leg.upperPivotGroup);
          upperNode.userData.deviceId = dev.id;
          const meshes = [];
          upperNode.traverse(c => {
            if (c.isMesh) {
              c.geometry.boundsTree = new MeshBVH(c.geometry);
              c.userData.deviceId = dev.id;
              meshes.push(c);
            }
          });
          dev.robotLinkMeshes.push({ name: leg.upperMeshName, meshes, jointIdx: -1 });
          createLabel(`Leg ${i + 1} Upper`, upperNode);
        } else {
          console.warn(`[Hexapod] Upper mesh "${leg.upperMeshName}" not found`);
        }
      }

      const platName = config.platform.mesh;
      const baseName = config.base.mesh;
      dev.parentGroups[platName] = platformGroup;
      dev.parentGroups[baseName] = rootGroup;
      config.links = [
        { name: platName, label: 'Top Plate (platform)' },
        { name: baseName, label: 'Base Plate (fixed)' },
      ];

      dev.loaded = true;
      resolve();
    }, (progress) => {
      const pct = progress.total ? Math.round(progress.loaded / progress.total * 100) : '?';
      document.getElementById('loading').textContent = `Loading ${config.name}... ${pct}%`;
    }, (err) => {
      console.error('Load error:', err);
      reject(err);
    });
  });

  updateHexapodPose(dev);
  return dev;
}

// ============================================================
// updateHexapodPose — solve all legs for current platform pose
// ============================================================
const _platQuat = new THREE.Quaternion();
const _platEuler = new THREE.Euler();
const _pivotWorld = new THREE.Vector3();
const _newDir = new THREE.Vector3();
const _deltaQ = new THREE.Quaternion();

export function updateHexapodPose(dev) {
  const [xmm, ymm, zmm, rxDeg, ryDeg, rzDeg] = dev.platformPose;

  dev.platformGroup.position.set(
    dev.platformRestPos.x + xmm / 1000,
    dev.platformRestPos.y + zmm / 1000,
    dev.platformRestPos.z - ymm / 1000,
  );

  _platEuler.set(rxDeg * deg2rad, rzDeg * deg2rad, -ryDeg * deg2rad, 'XZY');
  _platQuat.setFromEuler(_platEuler);
  dev.platformGroup.quaternion.copy(_platQuat);

  for (const leg of dev.legData) {
    _pivotWorld.copy(leg.platformPivotLocal)
      .applyQuaternion(_platQuat)
      .add(dev.platformGroup.position);

    leg.upperPivotGroup.position.copy(_pivotWorld);

    _newDir.subVectors(_pivotWorld, leg.basePivot).normalize();
    _deltaQ.setFromUnitVectors(leg.restLowerDir, _newDir);
    leg.lowerPivotGroup.quaternion.copy(_deltaQ);

    _newDir.subVectors(leg.basePivot, _pivotWorld).normalize();
    _deltaQ.setFromUnitVectors(leg.restUpperDir, _newDir);
    leg.upperPivotGroup.quaternion.copy(_deltaQ);
  }
}

// ============================================================
// syncHexapodFromTransform — read platformGroup back to pose
// ============================================================
const rad2deg = 180 / Math.PI;

export function syncHexapodFromTransform(dev) {
  const pos = dev.platformGroup.position;
  const rest = dev.platformRestPos;

  dev.platformPose[0] = (pos.x - rest.x) * 1000;
  dev.platformPose[1] = -(pos.z - rest.z) * 1000;
  dev.platformPose[2] = (pos.y - rest.y) * 1000;

  _platEuler.setFromQuaternion(dev.platformGroup.quaternion, 'XZY');
  dev.platformPose[3] = _platEuler.x * rad2deg;
  dev.platformPose[4] = -_platEuler.z * rad2deg;
  dev.platformPose[5] = _platEuler.y * rad2deg;

  const lim = dev.config.limits;
  const keys = ['x', 'y', 'z', 'rx', 'ry', 'rz'];
  for (let i = 0; i < 6; i++) {
    const [lo, hi] = lim[keys[i]];
    dev.platformPose[i] = Math.max(lo, Math.min(hi, dev.platformPose[i]));
  }

  updateHexapodPose(dev);
}

// ============================================================
// computeLegLengthsFromPose
// ============================================================
const _fkPos = new THREE.Vector3();
const _fkEuler = new THREE.Euler();
const _fkQuat = new THREE.Quaternion();
const _fkPivot = new THREE.Vector3();

export function computeLegLengthsFromPose(dev, pose) {
  const [xmm, ymm, zmm, rxDeg, ryDeg, rzDeg] = pose;
  _fkPos.set(
    dev.platformRestPos.x + xmm / 1000,
    dev.platformRestPos.y + zmm / 1000,
    dev.platformRestPos.z - ymm / 1000,
  );
  _fkEuler.set(rxDeg * deg2rad, rzDeg * deg2rad, -ryDeg * deg2rad, 'XZY');
  _fkQuat.setFromEuler(_fkEuler);

  return dev.legData.map(leg => {
    _fkPivot.copy(leg.platformPivotLocal).applyQuaternion(_fkQuat).add(_fkPos);
    return _fkPivot.distanceTo(leg.basePivot);
  });
}

// ============================================================
// solveHexapodFK — Newton-Raphson: leg lengths → platform pose
// ============================================================
export function solveHexapodFK(dev, targetLengths, maxIter = 50, tol = 1e-7) {
  const pose = [...dev.platformPose];
  const eps = [0.01, 0.01, 0.01, 0.001, 0.001, 0.001];

  for (let iter = 0; iter < maxIter; iter++) {
    const cur = computeLegLengthsFromPose(dev, pose);
    const err = targetLengths.map((t, i) => t - cur[i]);
    if (Math.max(...err.map(Math.abs)) < tol) break;

    const J = Array.from({ length: 6 }, () => new Array(6));
    for (let j = 0; j < 6; j++) {
      const pp = [...pose], pm = [...pose];
      pp[j] += eps[j];
      pm[j] -= eps[j];
      const lp = computeLegLengthsFromPose(dev, pp);
      const lm = computeLegLengthsFromPose(dev, pm);
      for (let i = 0; i < 6; i++) J[i][j] = (lp[i] - lm[i]) / (2 * eps[j]);
    }

    const aug = J.map((row, i) => [...row, err[i]]);
    let singular = false;
    for (let col = 0; col < 6; col++) {
      let best = col;
      for (let r = col + 1; r < 6; r++) {
        if (Math.abs(aug[r][col]) > Math.abs(aug[best][col])) best = r;
      }
      [aug[col], aug[best]] = [aug[best], aug[col]];
      if (Math.abs(aug[col][col]) < 1e-14) { singular = true; break; }
      for (let r = col + 1; r < 6; r++) {
        const f = aug[r][col] / aug[col][col];
        for (let k = col; k <= 6; k++) aug[r][k] -= f * aug[col][k];
      }
    }
    if (singular) break;

    const delta = new Array(6);
    for (let i = 5; i >= 0; i--) {
      delta[i] = aug[i][6];
      for (let j = i + 1; j < 6; j++) delta[i] -= aug[i][j] * delta[j];
      delta[i] /= aug[i][i];
    }

    for (let i = 0; i < 6; i++) pose[i] += delta[i];
  }

  const lim = dev.config.limits;
  const keys = ['x', 'y', 'z', 'rx', 'ry', 'rz'];
  for (let i = 0; i < 6; i++) {
    const [lo, hi] = lim[keys[i]];
    pose[i] = Math.max(lo, Math.min(hi, pose[i]));
  }
  return pose;
}

// ============================================================
// syncHexapodSliders — update slider DOM from current pose
// ============================================================
export function syncHexapodSliders(dev) {
  if (dev !== State.activeDevice) return;
  const pose = dev.platformPose;
  const ids = ['hpx', 'hpy', 'hpz', 'hprx', 'hpry', 'hprz'];
  for (let i = 0; i < 6; i++) {
    const slider = document.getElementById(ids[i]);
    const label = document.getElementById(ids[i] + 'v');
    if (slider) slider.value = pose[i];
    if (label) label.textContent = i >= 3 ? pose[i].toFixed(2) : pose[i].toFixed(1);
  }
  const lengths = computeLegLengthsFromPose(dev, pose);
  for (let i = 0; i < 6; i++) {
    const mm = lengths[i] * 1000;
    const s = document.getElementById(`hleg${i + 1}`);
    const l = document.getElementById(`hleg${i + 1}v`);
    if (s) s.value = mm;
    if (l) l.textContent = mm.toFixed(2);
  }
}
