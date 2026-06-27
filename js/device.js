// ============================================================
// js/device.js — loadDevice, updateSliders, updateChain,
//                setIKMode, syncIKSliders
// ============================================================
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { MeshBVH } from 'three-mesh-bvh';

import * as State from './state.js';
import {
  updateFK, getEEWorldPosition, getEEWorldQuaternion,
  getJointWorldAxis, clampJoints,
  kappaToEuler, eulerToKappa, getCompensation, updateVirtualAngles,
  pyEulerFromRelQuat,
} from './kinematics.js';
import { loadHexapod } from './hexapod.js';

const deg2rad = Math.PI / 180;
const rad2deg = 180 / Math.PI;

// ============================================================
// buildAdjacencyPairs
// ============================================================
export function buildAdjacencyPairs(config) {
  const adjPairs = new Set();
  function movableAncestor(jointIdx) {
    let idx = config.joints[jointIdx].parent;
    while (idx >= 0) {
      if (!config.joints[idx].fixed) return idx;
      idx = config.joints[idx].parent;
    }
    return -1;
  }
  const linksByJoint = {};
  for (const link of config.links) {
    if (!linksByJoint[link.joint]) linksByJoint[link.joint] = [];
    linksByJoint[link.joint].push(link.name);
  }
  for (const names of Object.values(linksByJoint)) {
    for (let a = 0; a < names.length; a++)
      for (let b = a + 1; b < names.length; b++)
        adjPairs.add([names[a], names[b]].sort().join('|'));
  }
  for (const linkA of config.links) {
    const ancA = movableAncestor(linkA.joint);
    for (const linkB of config.links) {
      if (linkA.name >= linkB.name) continue;
      const ancB = movableAncestor(linkB.joint);
      const jA = config.joints[linkA.joint].fixed ? movableAncestor(linkA.joint) : linkA.joint;
      const jB = config.joints[linkB.joint].fixed ? movableAncestor(linkB.joint) : linkB.joint;
      if (jA === jB || jA === ancB || jB === ancA) {
        adjPairs.add([linkA.name, linkB.name].sort().join('|'));
      }
    }
  }
  return adjPairs;
}

// ============================================================
// loadDevice
// ============================================================
export async function loadDevice(configFile) {
  const peek = await fetch(configFile + '?_=' + Date.now()).then(r => r.json());
  if (peek.type === 'hexapod') return loadHexapod(configFile);

  const config = peek;
  const id = State.incrementDeviceId();
  const numJoints = config.joints.length;

  // Root group for the entire device (movable origin)
  const rootGroup = new THREE.Group();
  rootGroup.name = config.name + '_root';
  State.scene.add(rootGroup);

  // Build joint limits, axes, FK chain
  const jointLimits = config.joints.map(j => [j.limits[0] * deg2rad, j.limits[1] * deg2rad]);
  const jointFixed = config.joints.map(j => !!j.fixed);
  const apiSign = config.joints.map(j => (j.apiSign !== undefined) ? j.apiSign : 1);

  const linkToJoint = {};
  for (const link of config.links) linkToJoint[link.name] = link.joint;

  const jointRestGroups = [];
  const jointRotGroups = [];
  const jointAxes = [];

  const isBranching = config.joints.some((j, i) => {
    const p = j.parent !== undefined ? j.parent : i - 1;
    return i > 0 && p === -1;
  });

  for (let i = 0; i < numJoints; i++) {
    const d = config.joints[i];
    const parentIdx = d.parent !== undefined ? d.parent : i - 1;
    const parentGroup = parentIdx < 0 ? rootGroup : jointRotGroups[parentIdx];

    const restGrp = new THREE.Group();
    restGrp.name = `J${i+1}_rest`;
    restGrp.position.set(d.restPos[0], d.restPos[1], d.restPos[2]);
    restGrp.quaternion.set(d.restQuat[1], d.restQuat[2], d.restQuat[3], d.restQuat[0]);
    parentGroup.add(restGrp);
    jointRestGroups.push(restGrp);

    const rotGrp = new THREE.Group();
    rotGrp.name = `J${i+1}_rot`;
    restGrp.add(rotGrp);
    jointRotGroups.push(rotGrp);

    if (d.name && d.name.startsWith('virtual_axis')) {
      rotGrp.add(new THREE.AxesHelper(0.1));
    }

    jointAxes.push(new THREE.Vector3(d.axis[0], d.axis[1], d.axis[2]).normalize());
  }

  // End-effector marker
  const eeMarker = new THREE.Group();
  const eeParentGroup = isBranching ? jointRotGroups[jointRotGroups.length - 1] : jointRotGroups[numJoints - 1];
  eeParentGroup.add(eeMarker);
  if (config.eeOffset) eeMarker.position.set(...config.eeOffset);
  const axLen = 0.03;
  function makeAxis(dir, color) {
    const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), dir.clone().multiplyScalar(axLen)]);
    return new THREE.Line(g, new THREE.LineBasicMaterial({ color }));
  }
  const eeAxes = config.eeAxes || [[1,0,0],[0,-1,0],[0,0,1]];
  eeMarker.add(makeAxis(new THREE.Vector3(...eeAxes[0]), 0x0000ff));
  eeMarker.add(makeAxis(new THREE.Vector3(...eeAxes[1]), 0x00ff00));
  eeMarker.add(makeAxis(new THREE.Vector3(...eeAxes[2]), 0xff0000));

  // Chain visualization
  const chainPts = new Float32Array((numJoints + 1) * 3);
  const chainLineGeo = new THREE.BufferGeometry();
  chainLineGeo.setAttribute('position', new THREE.BufferAttribute(chainPts, 3));
  const chainLineIndices = [];
  for (let i = 0; i < numJoints; i++) { chainLineIndices.push(i, i + 1); }
  chainLineGeo.setIndex(chainLineIndices);
  const chainLine = new THREE.LineSegments(
    chainLineGeo,
    new THREE.LineBasicMaterial({ color: 0xffdd44, linewidth: 2, depthTest: false, transparent: true, opacity: 0.85 })
  );
  chainLine.renderOrder = 1;
  chainLine.visible = false;
  State.scene.add(chainLine);

  const chainSphereMat = new THREE.MeshStandardMaterial({ color: 0xffdd44, emissive: 0x554400, depthTest: false });
  const chainSpheres = [];
  for (let i = 0; i < numJoints + 1; i++) {
    const r = i === 0 ? 0.009 : i === numJoints ? 0.007 : 0.006;
    const s = new THREE.Mesh(new THREE.SphereGeometry(r, 10, 10), chainSphereMat);
    s.renderOrder = 1;
    s.visible = false;
    State.scene.add(s);
    chainSpheres.push(s);
  }

  // IK target
  const targetGeo = new THREE.SphereGeometry(0.012, 16, 16);
  const targetMat = new THREE.MeshStandardMaterial({
    color: 0x22ff44, emissive: 0x115522, transparent: true, opacity: 0.8
  });
  const ikTarget = new THREE.Mesh(targetGeo, targetMat);
  ikTarget.position.set(0.19, 0.308, 0);
  ikTarget.visible = false;
  State.scene.add(ikTarget);

  const ikTargetQuat = new THREE.Quaternion();
  const ikTargetEuler = new THREE.Euler(0, 0, 0, 'YZX');

  const lineGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(), new THREE.Vector3()
  ]);
  const lineMat = new THREE.LineBasicMaterial({ color: 0x22ff44, transparent: true, opacity: 0.5 });
  const ikLine = new THREE.Line(lineGeo, lineMat);
  ikLine.visible = false;
  State.scene.add(ikLine);

  // Slider mapping (skip fixed joints)
  const sliderJointMap = [];
  const kappaSliderNames = {};
  {
    const ki = config.joints.findIndex(j => j.name === 'kappa');
    const ti = config.joints.findIndex(j => j.name === 'theta');
    const pi = config.joints.findIndex(j => j.name === 'phi');
    if (ki >= 0 && ti >= 0 && pi >= 0) {
      kappaSliderNames[ti] = 'ktheta';
      kappaSliderNames[pi] = 'kphi';
    }
  }
  for (let i = 0; i < numJoints; i++) {
    if (config.joints[i].fixed) continue;
    sliderJointMap.push(i);
  }

  // Kappa geometry detection
  const kappaJointIdx = config.joints.findIndex(j => j.name === 'kappa');
  const thetaJointIdx = config.joints.findIndex(j => j.name === 'theta');
  const phiJointIdx   = config.joints.findIndex(j => j.name === 'phi');
  const isKappaGeometry = kappaJointIdx >= 0 && thetaJointIdx >= 0 && phiJointIdx >= 0;

  // Origin helper — axis gizmo + coordinate label at device base
  const originHelpers = [];
  const originLabels = [];
  const originSize = 0.04;
  {
    const axes = new THREE.AxesHelper(originSize);
    axes.renderOrder = 2;
    axes.material.depthTest = false;
    axes.visible = State.originsOn;
    rootGroup.add(axes);
    originHelpers.push(axes);

    const div = document.createElement('div');
    div.className = 'origin-label';
    div.textContent = '';
    const lbl = new CSS2DObject(div);
    lbl.position.set(0, originSize * 1.2, 0);
    lbl.visible = State.originsOn;
    rootGroup.add(lbl);
    originLabels.push(lbl);
  }

  // Build adjacency for collision detection
  const adjPairs = buildAdjacencyPairs(config);

  const dev = {
    id,
    config,
    configFile,
    name: config.name,
    numJoints,
    rootGroup,
    jointLimits,
    jointFixed,
    jointAngles: Array(numJoints).fill(0),
    jointRestGroups,
    jointRotGroups,
    jointAxes,
    apiSign,
    linkToJoint,
    sliderJointMap,
    kappaSliderNames,
    isBranching,
    eeMarker,
    meshLabels: [],
    robotLinkMeshes: [],
    staticMeshes: [],
    originHelpers,
    originLabels,
    chainVisible: false,
    chainLine,
    chainSpheres,
    chainLineGeo,
    chainPts,
    ikMode: false,
    ikTarget,
    ikTargetQuat,
    ikTargetEuler,
    ikLine,
    adjPairs,
    isKappaGeometry,
    kappaAlpha: 0,
    kappaJointIdx,
    thetaJointIdx,
    phiJointIdx,
    kappaSignPositive: true,
    kappaPhiSign: 1,
    kappaThetaSign: 1,
    parentLink: null,
    loaded: false,
  };

  // Compute kappa geometry parameters after FK chain is ready
  if (isKappaGeometry) {
    State.scene.updateMatrixWorld(true);
    const thetaWorldAxis = getJointWorldAxis(dev, thetaJointIdx);
    const kappaWorldAxis = getJointWorldAxis(dev, kappaJointIdx);
    const phiWorldAxis   = getJointWorldAxis(dev, phiJointIdx);
    dev.kappaAlpha      = Math.acos(Math.min(1, Math.abs(thetaWorldAxis.dot(kappaWorldAxis))));
    dev.kappaPhiSign    = thetaWorldAxis.dot(phiWorldAxis) >= 0 ? 1 : -1;
    dev.kappaThetaSign  = thetaWorldAxis.dot(kappaWorldAxis) >= 0 ? 1 : -1;
  }

  // Load GLB model
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

      const reparented = new Set();
      for (const [linkName, jointIdx] of Object.entries(linkToJoint)) {
        const node = allNodes[linkName];
        if (!node) {
          console.warn(`${linkName} not found in glTF`);
          continue;
        }
        node.updateWorldMatrix(true, false);
        const worldMat = node.matrixWorld.clone();
        node.removeFromParent();

        const target = jointRotGroups[jointIdx];
        target.updateWorldMatrix(true, false);
        const localMat = target.matrixWorld.clone().invert().multiply(worldMat);

        node.matrix.copy(localMat);
        node.matrix.decompose(node.position, node.quaternion, node.scale);
        target.add(node);

        reparented.add(linkName);
        node.traverse((c) => { if (c.name) reparented.add(c.name); });
      }

      // Remove hidden objects
      const hideNames = ['Icosphere', 'Cross'];
      State.scene.traverse((child) => {
        if (child.name && hideNames.includes(child.name)) {
          child.removeFromParent();
        }
      });
      const hiddenNodes = new Set();
      model.traverse((child) => {
        if (child.name && hideNames.includes(child.name)) {
          child.traverse((c) => hiddenNodes.add(c));
        }
      });
      const statics = [];
      model.traverse((child) => {
        if (child.isMesh && !reparented.has(child.name) && !hiddenNodes.has(child)) {
          statics.push(child);
        }
      });
      for (const mesh of statics) {
        mesh.updateWorldMatrix(true, false);
        const wm = mesh.matrixWorld.clone();
        mesh.removeFromParent();
        rootGroup.add(mesh);
        mesh.matrix.copy(wm);
        mesh.matrix.decompose(mesh.position, mesh.quaternion, mesh.scale);
        dev.staticMeshes.push(mesh);
      }

      // Create labels and build collision data
      function createLabel(name, parentObj) {
        const div = document.createElement('div');
        div.className = 'mesh-label';
        div.textContent = name;
        const label = new CSS2DObject(div);
        label.visible = false;
        const box = new THREE.Box3().setFromObject(parentObj);
        const center = box.getCenter(new THREE.Vector3());
        parentObj.worldToLocal(center);
        label.position.copy(center);
        parentObj.add(label);
        dev.meshLabels.push(label);
      }

      for (const [linkName, jointIdx] of Object.entries(linkToJoint)) {
        const node = allNodes[linkName];
        if (node) {
          createLabel(linkName, node);
          const meshes = [];
          node.traverse((c) => {
            if (c.isMesh) {
              meshes.push(c);
              c.geometry.boundsTree = new MeshBVH(c.geometry);
              c.userData.deviceId = dev.id;
            }
          });
          if (meshes.length > 0) dev.robotLinkMeshes.push({ name: linkName, meshes, jointIdx });
        }
      }
      for (const mesh of statics) {
        if (mesh.name) createLabel(mesh.name, mesh);
        mesh.userData.deviceId = dev.id;
      }

      // Kappa chi slider limits
      if (isKappaGeometry) {
        const kappaLimits = [jointLimits[kappaJointIdx][0] * rad2deg, jointLimits[kappaJointIdx][1] * rad2deg];
        const chiAtMin = -kappaToEuler(dev, kappaLimits[0]).chi;
        const chiAtMax = -kappaToEuler(dev, kappaLimits[1]).chi;
        dev._chiLimits = [Math.min(chiAtMin, chiAtMax), Math.max(chiAtMin, chiAtMax)];

        const test90 = eulerToKappa(dev, 90);
        if (test90) {
          const comp90 = getCompensation(dev, test90.kappa);
          console.log(`Kappa geometry (analytical): alpha=${(dev.kappaAlpha * rad2deg).toFixed(1)} deg, phiSign=${dev.kappaPhiSign}, chi=90 deg -> kappa=${test90.kappa.toFixed(1)} deg, comp_theta=${comp90.theta.toFixed(1)} deg, comp_phi=${comp90.phi.toFixed(1)} deg`);
        }
      }

      // Capture home EE quaternion (all joints at 0)
      State.scene.updateMatrixWorld(true);
      dev.homeQuaternion = getEEWorldQuaternion(dev);
      dev.homeQuaternionInv = dev.homeQuaternion.clone().invert();

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

  State.requestRender();   // device meshes load asynchronously (off the input path)
  return dev;
}

// ============================================================
// updateSliders
// ============================================================
export function updateSliders(dev) {
  if (dev !== State.activeDevice) return;
  for (let si = 0; si < dev.sliderJointMap.length; si++) {
    const ji = dev.sliderJointMap[si];
    const deg = dev.apiSign[ji] * dev.jointAngles[ji] * rad2deg;
    const slider = document.getElementById(`j${si+1}`);
    const label  = document.getElementById(`v${si+1}`);
    if (slider) slider.value = deg;
    if (label)  label.textContent = deg.toFixed(2);
  }
  updateVirtualAngles(dev);
}

// ============================================================
// setIKMode
// ============================================================
export function setIKMode(dev, on) {
  dev.ikMode = on;
  if (dev !== State.activeDevice) return;

  const btn = document.getElementById('ikBtn');

  if (dev.type === 'hexapod') {
    btn.textContent = `Drag Platform: ${on ? 'ON' : 'OFF'}`;
    btn.classList.toggle('active', on);
    document.getElementById('ik-panel').style.display = 'none';
    if (on) {
      State.transformControls.attach(dev.platformGroup);
    } else {
      State.transformControls.detach();
    }
    return;
  }

  btn.textContent = `IK Mode: ${on ? 'ON' : 'OFF'}`;
  btn.classList.toggle('active', on);

  dev.ikTarget.visible = on;
  dev.ikLine.visible = on;

  document.getElementById('ik-panel').style.display = on ? 'block' : 'none';

  if (on) {
    State.scene.updateMatrixWorld(true);
    dev.ikTarget.position.copy(getEEWorldPosition(dev));
    dev.ikTargetQuat.copy(getEEWorldQuaternion(dev));
    dev.ikTargetEuler.setFromQuaternion(dev.ikTargetQuat, 'YZX');
    dev.ikTarget.quaternion.copy(dev.ikTargetQuat);
    State.transformControls.attach(dev.ikTarget);
    syncIKSliders(dev);
  } else {
    State.transformControls.detach();
  }
}

// ============================================================
// syncIKSliders
// ============================================================
export function syncIKSliders(dev) {
  if (dev !== State.activeDevice) return;
  const p = dev.ikTarget.position;
  const xmm = Math.round(p.x * 1000);
  const ymm = Math.round(p.z * 1000);
  const zmm = Math.round(p.y * 1000);
  document.getElementById('ikx').value = xmm;
  document.getElementById('ikvx').textContent = xmm;
  document.getElementById('iky').value = ymm;
  document.getElementById('ikvy').textContent = ymm;
  document.getElementById('ikz').value = zmm;
  document.getElementById('ikvz').textContent = zmm;

  const relQuat = dev.ikTargetQuat.clone().multiply(dev.homeQuaternionInv);
  const [a, b, g] = pyEulerFromRelQuat(relQuat);
  const ad = Math.round(a);
  const bd = Math.round(b);
  const cd = Math.round(g);
  console.log('[IK-OUT]', {ikTargetQuat: dev.ikTargetQuat.toArray(),
    homeQInv: dev.homeQuaternionInv.toArray(),
    readbackRelQ: relQuat.toArray(), decoded: [a, b, g], rounded: [ad, bd, cd]});
  document.getElementById('ika').value = ad;
  document.getElementById('ikva').textContent = ad;
  document.getElementById('ikb').value = bd;
  document.getElementById('ikvb').textContent = bd;
  document.getElementById('ikc').value = cd;
  document.getElementById('ikvc').textContent = cd;
}
