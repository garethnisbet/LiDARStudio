// ============================================================
// js/main.js — animate loop, event handlers, initialisation
// ============================================================
import * as THREE from 'three';
import { MeshBVH, acceleratedRaycast } from 'three-mesh-bvh';

// Patch Three.js Mesh to use BVH-accelerated raycasting
THREE.Mesh.prototype.raycast = acceleratedRaycast;

// ============================================================
// Constants
// ============================================================
const deg2rad = Math.PI / 180;
const rad2deg = 180 / Math.PI;

// ============================================================
// Modules — scene must be imported first (it writes to State)
// ============================================================
import * as State from './state.js';
import {
  setOrtho, updateOrthoFrustum,
  navCanvas, NAV_AXES, navHitTest,
  navHovered, setNavHovered, renderNavGizmo,
  snapAnim, setSnapAnim, snapClock, easeNavSnap,
  setFloorSize,
} from './scene.js';
import {
  updateFK, getEEWorldPosition, getEEWorldQuaternion,
  updateChain, solveIK, relQuatFromPyEuler,
} from './kinematics.js';
import {
  loadDevice,
  updateSliders, setIKMode, syncIKSliders,
} from './device.js';
import { updateHexapodPose, syncHexapodFromTransform, syncHexapodSliders } from './hexapod.js';
import {
  buildControlPanel, rebuildDeviceList,
  setActiveDevice, findDeviceForObject,
  setDeviceParent, rebuildDeviceParentDropdown,
  rebuildPrimaryModelDropdown,
} from './panel.js';
import {
  buildScenePayload,
  buildSceneMetadataForDB, buildSceneBuffersForDB, sceneBufferSignature,
  exportSceneState, importSceneState, restoreSTLsFromState,
  loadSTLFile, loadOBJFile, loadPLYFile, loadGLBFile, loadSplatFile,
  updateSplatClip,
  addPrimitive,
  selectSTL, deselectSTL, setSTLTransformMode, setSTLParent, syncSTLNumericInputs,
} from './stl.js';
import { dbSave, dbLoad, BUFFERS_KEY } from './storage.js';
import { checkCollisions, clearCollisionHighlights, initCollisionWorker } from './collision.js';
import { initVR, updateVR } from './vr.js';
import {
  wsConnect, initWsInfoPanel, registerSetActiveDevice, registerAvailableConfigs,
} from './websocket.js';

// Register callbacks for websocket.js
// (avoids circular dependency: websocket -> panel -> websocket)
import { configFiles } from './panel.js';
import { initLidarPanel } from './lidar.js';
registerSetActiveDevice(setActiveDevice);
registerAvailableConfigs(configFiles);
try { initLidarPanel(); } catch (e) { console.error('LiDAR panel init failed:', e); }

// Start collision Web Worker (falls back to main thread if unavailable)
initCollisionWorker();

// ============================================================
// Raycaster (for click-to-select)
// ============================================================
const raycaster = new THREE.Raycaster();
raycaster.params.Points = { threshold: 0.005 };
const mouse = new THREE.Vector2();
const _originWP = new THREE.Vector3();

// ============================================================
// fmtV helper
// ============================================================
function fmtV(v) {
  return `(${(v.x*1000).toFixed(1)}, ${(v.z*1000).toFixed(1)}, ${(v.y*1000).toFixed(1)})mm`;
}

// Write textContent only when it actually changes — avoids needless
// style/layout invalidation on the readout elements every frame.
function setText(el, str) {
  if (el.textContent !== str) el.textContent = str;
}

// ============================================================
// Animate loop
// ============================================================
const eePosEl = document.getElementById('eePos');
const tgtPosEl = document.getElementById('tgtPos');
const ikErrEl  = document.getElementById('ikErr');

// Scratch state for the IK driver's end-effector movement test
const _ikEEPos = new THREE.Vector3();
const _ikPrevEE = new THREE.Vector3(NaN, NaN, NaN);
let _ikLastErr = 0;

function animate(time, frame) {
  updateVR(frame);

  const dev = State.activeDevice;

  // --- IK driver -----------------------------------------------------
  // IK must keep solving every frame so the arm animates toward a
  // (possibly moving) target. It is cheap once converged — solveIK
  // returns on the first iteration. A render is requested only while the
  // end-effector is actually moving, so a settled IK pose stays idle.
  if (dev && dev.ikMode && dev.type !== 'hexapod') {
    _ikLastErr = solveIK(dev, dev.ikTarget.position, dev.ikTargetQuat, 10, 0.00005);
    _ikEEPos.copy(getEEWorldPosition(dev));
    if (_ikEEPos.distanceToSquared(_ikPrevEE) > 1e-12) {
      _ikPrevEE.copy(_ikEEPos);
      State.requestRender();
    }
  }

  // --- Camera snap animation + controls ------------------------------
  if (!State.vrActive) {
    const currentSnapAnim = snapAnim;
    if (currentSnapAnim) {
      currentSnapAnim.t = Math.min(1, currentSnapAnim.t + snapClock.getDelta() / 0.4);
      const t = easeNavSnap(currentSnapAnim.t);
      State.activeCamera.position.lerpVectors(currentSnapAnim.sp, currentSnapAnim.ep, t);
      State.activeCamera.up.lerpVectors(currentSnapAnim.su, currentSnapAnim.eu, t).normalize();
      State.requestRender();
      if (currentSnapAnim.t >= 1) {
        setSnapAnim(null); State.orbitControls.enabled = true;
        if (State.orthoOn) updateOrthoFrustum();
      }
    } else {
      snapClock.getDelta();
    }
    // update() advances damping and dispatches 'change' (-> requestRender)
    // whenever the camera actually moves; it is a no-op once settled.
    State.orbitControls.update();
  }

  // --- On-demand render gate -----------------------------------------
  if (!(State.vrActive || State.needsRender || State.shouldRenderContinuously())) return;
  State.clearNeedsRender();

  // Readouts / chain / origin visuals — only refreshed on a drawn frame.
  if (dev) updateReadout(dev);
  for (const d of State.devices) {
    if (d.chainVisible) updateChain(d);
  }
  if (State.originsOn) {
    for (const d of State.devices) {
      d.rootGroup.getWorldPosition(_originWP);
      const x = +(_originWP.x * 1000).toFixed(1);
      const y = +(_originWP.z * 1000).toFixed(1);
      const z = +(_originWP.y * 1000).toFixed(1);
      setText(d.originLabels[0].element, `${d.name} ${x}, ${y}, ${z}`);
    }
  }

  // Foreground splat clip tracks camera distance, so refresh it on each
  // drawn frame before rendering.
  updateSplatClip();

  State.renderer.render(State.scene, State.activeCamera);

  // Collision check uses the world matrices refreshed by render(); its
  // highlight functions request a render only when the hit set changes,
  // so this does not spin the loop.
  checkCollisions();

  if (!State.vrActive) {
    State.labelRenderer.render(State.scene, State.activeCamera);
    renderNavGizmo();
  }
}

// Update the EE / target / error readout for the active device.
// Runs only on rendered frames; the IK solve itself happens in the driver.
function updateReadout(dev) {
  if (dev.ikMode && dev.type === 'hexapod') {
    syncHexapodFromTransform(dev);
    syncHexapodSliders(dev);
    setText(eePosEl, fmtV(dev.platformGroup.position));
    setText(tgtPosEl, '-');
    setText(ikErrEl, '-');
  } else if (dev.ikMode) {
    updateSliders(dev);
    const eePos = getEEWorldPosition(dev);
    const pts = dev.ikLine.geometry.attributes.position;
    pts.setXYZ(0, eePos.x, eePos.y, eePos.z);
    pts.setXYZ(1, dev.ikTarget.position.x, dev.ikTarget.position.y, dev.ikTarget.position.z);
    pts.needsUpdate = true;
    setText(eePosEl, fmtV(eePos));
    setText(tgtPosEl, fmtV(dev.ikTarget.position));
    setText(ikErrEl, (_ikLastErr * 1000).toFixed(2) + 'mm');
    syncIKSliders(dev);
  } else if (dev.type === 'hexapod') {
    dev.platformGroup.updateWorldMatrix(true, false);
    _originWP.setFromMatrixPosition(dev.platformGroup.matrixWorld);
    setText(eePosEl, fmtV(_originWP));
    setText(tgtPosEl, '-');
    setText(ikErrEl, '-');
  } else {
    setText(eePosEl, fmtV(getEEWorldPosition(dev)));
    setText(tgtPosEl, '-');
    setText(ikErrEl, '-');
  }
}

// ============================================================
// Button event handlers
// ============================================================

document.getElementById('resetBtn').addEventListener('click', () => {
  if (!State.activeDevice) return;
  const dev = State.activeDevice;
  if (dev.type === 'hexapod') {
    dev.platformPose.fill(0);
    updateHexapodPose(dev);
    buildControlPanel(dev);
    return;
  }
  for (let i = 0; i < dev.numJoints; i++) dev.jointAngles[i] = 0;
  updateFK(dev);
  updateSliders(dev);
  if (dev.ikMode) {
    State.scene.updateMatrixWorld(true);
    dev.ikTarget.position.copy(getEEWorldPosition(dev));
    dev.ikTargetQuat.copy(getEEWorldQuaternion(dev));
    dev.ikTargetEuler.setFromQuaternion(dev.ikTargetQuat, 'YZX');
    dev.ikTarget.quaternion.copy(dev.ikTargetQuat);
    syncIKSliders(dev);
  }
});

document.getElementById('demoBtn').addEventListener('click', () => {
  if (!State.activeDevice) return;
  const dev = State.activeDevice;
  if (dev.type === 'hexapod') {
    if (dev.config.demoPose) {
      for (let i = 0; i < 6; i++) dev.platformPose[i] = dev.config.demoPose[i] || 0;
      updateHexapodPose(dev);
      buildControlPanel(dev);
    }
    return;
  }
  if (dev.isKappaGeometry) {
    for (let i = 0; i < dev.numJoints; i++) dev.jointAngles[i] = 0;
    dev.jointAngles[dev.kappaJointIdx] = -134.6 * deg2rad;
    dev.jointAngles[dev.thetaJointIdx] = -33.5 * deg2rad;
    dev.jointAngles[dev.phiJointIdx]   = -146.9 * deg2rad;
  } else if (dev.config.demoPose) {
    const pose = dev.config.demoPose;
    for (let i = 0; i < dev.numJoints && i < pose.length; i++) {
      dev.jointAngles[i] = pose[i] * deg2rad;
    }
  }
  updateFK(dev);
  updateSliders(dev);
  if (dev.ikMode) {
    State.scene.updateMatrixWorld(true);
    dev.ikTarget.position.copy(getEEWorldPosition(dev));
    dev.ikTargetQuat.copy(getEEWorldQuaternion(dev));
    dev.ikTargetEuler.setFromQuaternion(dev.ikTargetQuat, 'YZX');
    dev.ikTarget.quaternion.copy(dev.ikTargetQuat);
    syncIKSliders(dev);
  }
});

document.getElementById('ikBtn').addEventListener('click', () => {
  if (!State.activeDevice) return;
  if (State.activeDevice.isBranching && State.activeDevice.type !== 'hexapod') return;
  setIKMode(State.activeDevice, !State.activeDevice.ikMode);
});

document.getElementById('chainBtn').addEventListener('click', () => {
  if (!State.activeDevice) return;
  State.activeDevice.chainVisible = !State.activeDevice.chainVisible;
  document.getElementById('chainBtn').textContent = `Chain: ${State.activeDevice.chainVisible ? 'ON' : 'OFF'}`;
  document.getElementById('chainBtn').classList.toggle('active', State.activeDevice.chainVisible);
  State.activeDevice.chainLine.visible = State.activeDevice.chainVisible;
  State.activeDevice.chainSpheres.forEach(s => s.visible = State.activeDevice.chainVisible);
  if (State.activeDevice.chainVisible) updateChain(State.activeDevice);
});

document.getElementById('orthoBtn').addEventListener('click', () => setOrtho(!State.orthoOn));

// --- Fly Around (auto-orbit) ---------------------------------------
// Smoothly orbits the camera around the OrbitControls target using the
// built-in autoRotate. A continuous-render key keeps the on-demand loop
// drawing for the duration; releasing it lets the scene settle to idle.
const flySpeedInput = document.getElementById('flySpeed');
State.orbitControls.autoRotateSpeed = +flySpeedInput.value;
function setFlyAround(on) {
  State.orbitControls.autoRotate = on;
  State.setContinuousRender('flyAround', on);
  const btn = document.getElementById('flyBtn');
  btn.textContent = `Fly Around: ${on ? 'ON' : 'OFF'}`;
  btn.classList.toggle('active', on);
  document.getElementById('flySpeedRow').style.display = on ? 'flex' : 'none';
}
document.getElementById('flyBtn').addEventListener('click', () => {
  setFlyAround(!State.orbitControls.autoRotate);
});
flySpeedInput.addEventListener('input', (e) => {
  State.orbitControls.autoRotateSpeed = +e.target.value;
  document.getElementById('flySpeedVal').textContent = `${(+e.target.value).toFixed(1)}×`;
});
// Any manual orbit/zoom drag stops the fly-around so the user takes over.
State.orbitControls.addEventListener('start', () => {
  if (State.orbitControls.autoRotate) setFlyAround(false);
});

document.getElementById('splatClip').addEventListener('input', (e) => {
  const pct = +e.target.value;
  State.setSplatClipFraction(pct / 100);
  document.getElementById('splatClipVal').textContent = `${pct}%`;
  State.requestRender();
});

document.getElementById('originsBtn').addEventListener('click', () => {
  State.setOriginsOn(!State.originsOn);
  document.getElementById('originsBtn').textContent = `Origins: ${State.originsOn ? 'ON' : 'OFF'}`;
  document.getElementById('originsBtn').classList.toggle('active', State.originsOn);
  for (const dev of State.devices) {
    dev.originHelpers.forEach(h => h.visible = State.originsOn);
    dev.originLabels.forEach(l => l.visible = State.originsOn);
  }
});

document.getElementById('labelBtn').addEventListener('click', () => {
  State.setLabelsOn(!State.labelsOn);
  document.getElementById('labelBtn').textContent = `Labels: ${State.labelsOn ? 'ON' : 'OFF'}`;
  document.getElementById('labelBtn').classList.toggle('active', State.labelsOn);
  for (const dev of State.devices) {
    dev.meshLabels.forEach(l => l.visible = State.labelsOn);
  }
});

// IK target position sliders
['ikx', 'iky', 'ikz'].forEach((id) => {
  document.getElementById(id).addEventListener('input', (e) => {
    if (!State.activeDevice) return;
    const mm = parseFloat(e.target.value);
    document.getElementById(id.replace('ik', 'ikv')).textContent = Math.round(mm);
    const sliderAxis = id.charAt(2);
    const threeAxis = sliderAxis === 'y' ? 'z' : sliderAxis === 'z' ? 'y' : 'x';
    State.activeDevice.ikTarget.position[threeAxis] = mm / 1000;
  });
});

// IK target orientation sliders
['ika', 'ikb', 'ikc'].forEach((id) => {
  document.getElementById(id).addEventListener('input', (e) => {
    if (!State.activeDevice) return;
    const deg = parseFloat(e.target.value);
    document.getElementById(id.replace('ik', 'ikv')).textContent = Math.round(deg);
    // Read all three user-convention slider values
    const aDeg = parseFloat(document.getElementById('ika').value);
    const bDeg = parseFloat(document.getElementById('ikb').value);
    const cDeg = parseFloat(document.getElementById('ikc').value);
    const relQuat = relQuatFromPyEuler(aDeg, bDeg, cDeg);
    State.activeDevice.ikTargetQuat.copy(relQuat).multiply(State.activeDevice.homeQuaternion);
    State.activeDevice.ikTargetEuler.setFromQuaternion(State.activeDevice.ikTargetQuat, 'YZX');
    State.activeDevice.ikTarget.quaternion.copy(State.activeDevice.ikTargetQuat);
    console.log('[IK-IN]', {input: [aDeg, bDeg, cDeg], relQuat: relQuat.toArray(),
      homeQ: State.activeDevice.homeQuaternion.toArray(),
      ikTargetQuat: State.activeDevice.ikTargetQuat.toArray()});
  });
});

// Move device origin button
document.getElementById('moveDeviceBtn').addEventListener('click', () => {
  if (!State.activeDevice) return;
  State.setMoveDeviceActive(!State.moveDeviceActive);
  const btn = document.getElementById('moveDeviceBtn');
  btn.textContent = State.moveDeviceActive ? 'Stop Moving Origin' : 'Move Device Origin';
  btn.classList.toggle('active', State.moveDeviceActive);
  document.getElementById('device-mode').style.display = State.moveDeviceActive ? 'block' : 'none';
  if (State.moveDeviceActive) {
    _syncDevNumericInputs(State.activeDevice);
    State.deviceTransformControls.attach(State.activeDevice.rootGroup);
  } else {
    State.deviceTransformControls.detach();
  }
});

// Device transform mode buttons
document.getElementById('devModeT').addEventListener('click', () => {
  State.deviceTransformControls.setMode('translate');
  document.getElementById('devModeT').classList.add('active');
  document.getElementById('devModeR').classList.remove('active');
});
document.getElementById('devModeR').addEventListener('click', () => {
  State.deviceTransformControls.setMode('rotate');
  document.getElementById('devModeR').classList.add('active');
  document.getElementById('devModeT').classList.remove('active');
});
document.getElementById('devSpaceBtn').addEventListener('click', () => {
  const ctrl = State.deviceTransformControls;
  const isLocal = ctrl.space === 'local';
  ctrl.setSpace(isLocal ? 'world' : 'local');
  const btn = document.getElementById('devSpaceBtn');
  btn.textContent = isLocal ? 'World' : 'Local';
  btn.classList.toggle('active', !isLocal);
});

document.getElementById('devResetPos').addEventListener('click', () => {
  if (!State.activeDevice) return;
  State.activeDevice.rootGroup.position.set(0, 0, 0);
  _syncDevNumericInputs(State.activeDevice);
});
document.getElementById('devResetOri').addEventListener('click', () => {
  if (!State.activeDevice) return;
  State.activeDevice.rootGroup.rotation.set(0, 0, 0);
  State.activeDevice.rootGroup.quaternion.identity();
  _syncDevNumericInputs(State.activeDevice);
});

// Numeric position/rotation inputs
function _syncDevNumericInputs(dev) {
  if (!dev) return;
  const p = dev.rootGroup.position;
  const r = dev.rootGroup.rotation;
  const fmt = v => +v.toFixed(2);
  document.getElementById('devPosX').value = fmt(p.x * 1000);
  document.getElementById('devPosY').value = fmt(p.z * 1000);
  document.getElementById('devPosZ').value = fmt(p.y * 1000);
  document.getElementById('devRotX').value = fmt(r.x * (180 / Math.PI));
  document.getElementById('devRotY').value = fmt(r.z * (180 / Math.PI));
  document.getElementById('devRotZ').value = fmt(r.y * (180 / Math.PI));
}

function _applyDevNumericInputs() {
  const dev = State.activeDevice;
  if (!dev) return;
  const x  = parseFloat(document.getElementById('devPosX').value) || 0;
  const y  = parseFloat(document.getElementById('devPosY').value) || 0;
  const z  = parseFloat(document.getElementById('devPosZ').value) || 0;
  const rx = parseFloat(document.getElementById('devRotX').value) || 0;
  const ry = parseFloat(document.getElementById('devRotY').value) || 0;
  const rz = parseFloat(document.getElementById('devRotZ').value) || 0;
  const d  = Math.PI / 180;
  dev.rootGroup.position.set(x / 1000, z / 1000, y / 1000);
  dev.rootGroup.rotation.set(rx * d, rz * d, ry * d);
}

['devPosX', 'devPosY', 'devPosZ', 'devRotX', 'devRotY', 'devRotZ'].forEach(id => {
  const el = document.getElementById(id);
  el.addEventListener('change', _applyDevNumericInputs);
  el.addEventListener('keydown', e => { if (e.key === 'Enter') { _applyDevNumericInputs(); el.blur(); } });
});

State.deviceTransformControls.addEventListener('objectChange', () => {
  if (State.activeDevice) _syncDevNumericInputs(State.activeDevice);
});

// Device parent dropdown
document.getElementById('deviceParentSelect').addEventListener('change', (e) => {
  if (!State.activeDevice) return;
  setDeviceParent(State.activeDevice, e.target.value);
});

// Primary model selector
document.getElementById('primaryModelSelect').addEventListener('change', async (e) => {
  const configFile = e.target.value;
  if (!configFile) return;

  // Remove the current primary device (first in the list)
  const primaryDev = State.devices[0];
  if (primaryDev && primaryDev.configFile === configFile) return; // same model

  const select = e.target;
  select.disabled = true;
  try {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('loading').textContent = 'Loading model...';

    // Load the new device first
    const dev = await loadDevice(configFile);

    // Remove old primary (allow removal even if it's the only device since we're replacing)
    if (primaryDev) {
      // Detach any active controls
      if (primaryDev === State.activeDevice) {
        State.transformControls.detach();
        State.deviceTransformControls.detach();
      }
      // Clean up scene objects
      State.scene.remove(primaryDev.rootGroup);
      primaryDev.rootGroup.traverse(child => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) {
          if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
          else child.material.dispose();
        }
      });
      for (const label of primaryDev.meshLabels) label.removeFromParent();
      if (primaryDev.chainLine) State.scene.remove(primaryDev.chainLine);
      for (const s of primaryDev.chainSpheres) State.scene.remove(s);
      if (primaryDev.ikTarget) State.scene.remove(primaryDev.ikTarget);
      if (primaryDev.ikLine) State.scene.remove(primaryDev.ikLine);
      const idx = State.devices.indexOf(primaryDev);
      if (idx >= 0) State.devices.splice(idx, 1);
    }

    // Insert new device at front as primary
    State.devices.unshift(dev);
    State.setActiveDevice(dev);
    if (dev.type === 'hexapod') updateHexapodPose(dev);
    else updateFK(dev);
    buildControlPanel(dev);
    rebuildDeviceList();

    // Auto-fit camera to new primary
    const fitBox = new THREE.Box3();
    dev.rootGroup.updateWorldMatrix(true, true);
    dev.rootGroup.traverse((child) => {
      if (child.isMesh) fitBox.expandByObject(child);
    });
    if (!fitBox.isEmpty()) {
      const fitCenter = fitBox.getCenter(new THREE.Vector3());
      const fitSize   = fitBox.getSize(new THREE.Vector3());
      const maxDim    = Math.max(fitSize.x, fitSize.y, fitSize.z);
      const fov       = State.camera.fov * (Math.PI / 180);
      const fitDist   = maxDim / (2 * Math.tan(fov / 2)) * 1.2;
      const direction = new THREE.Vector3(1, 0.6, 1).normalize();
      State.camera.position.copy(fitCenter).addScaledVector(direction, fitDist);
      State.orbitControls.target.copy(fitCenter);
      State.camera.updateProjectionMatrix();
      State.orbitControls.update();
    }

    document.getElementById('loading').style.display = 'none';
  } catch (err) {
    console.error('Failed to load primary model:', err);
    document.getElementById('loading').innerHTML =
      `<span style="color:#f88">Failed to load model</span><br>` +
      `<span style="color:#aaa; font-size:0.85em">${err?.message || err}</span>`;
    setTimeout(() => { document.getElementById('loading').style.display = 'none'; }, 3000);
  }
  select.disabled = false;
});

// Add device button
document.getElementById('addDeviceBtn').addEventListener('click', async () => {
  const select = document.getElementById('addDeviceSelect');
  const configFile = select.value;
  if (!configFile) return;

  const btn = document.getElementById('addDeviceBtn');
  btn.disabled = true;
  btn.textContent = '...';
  try {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('loading').textContent = `Loading device...`;
    const dev = await loadDevice(configFile);
    State.devices.push(dev);
    if (dev.type === 'hexapod') updateHexapodPose(dev);
    else updateFK(dev);
    setActiveDevice(dev);
    document.getElementById('loading').style.display = 'none';
  } catch (err) {
    console.error('Failed to load device:', err);
    document.getElementById('loading').innerHTML =
      `<span style="color:#f88">Failed to load device</span><br>` +
      `<span style="color:#aaa; font-size:0.85em">${err?.message || err}</span>`;
    setTimeout(() => { document.getElementById('loading').style.display = 'none'; }, 3000);
  }
  btn.disabled = false;
  btn.textContent = '+';
});

// STL import button
document.getElementById('stlBtn').addEventListener('click', () => {
  document.getElementById('stlFile').click();
});

document.getElementById('stlFile').addEventListener('change', (e) => {
  const files = [...e.target.files];
  const mtlFiles = new Map();
  for (const f of files) {
    if (f.name.toLowerCase().endsWith('.mtl'))
      mtlFiles.set(f.name.toLowerCase(), f);
  }
  for (const file of files) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (ext === 'mtl') continue;
    if (ext === 'stl')                        loadSTLFile(file);
    else if (ext === 'obj') {
      const mtlRef = file.name.replace(/\.obj$/i, '.mtl').toLowerCase();
      loadOBJFile(file, mtlFiles.get(mtlRef) || null);
    }
    else if (ext === 'ply')                   loadPLYFile(file);
    else if (ext === 'glb' || ext === 'gltf') loadGLBFile(file);
    else if (ext === 'splat' || ext === 'ksplat' || ext === 'spz') loadSplatFile(file);
  }
  e.target.value = '';
});

// Drag-and-drop file import onto the canvas
State.renderer.domElement.addEventListener('dragover', (e) => {
  e.preventDefault();
  e.dataTransfer.dropEffect = 'copy';
});

State.renderer.domElement.addEventListener('drop', async (e) => {
  e.preventDefault();
  const files = [...e.dataTransfer.files];
  const mtlFiles = new Map();
  for (const f of files) {
    if (f.name.toLowerCase().endsWith('.mtl'))
      mtlFiles.set(f.name.toLowerCase(), f);
  }
  for (const file of files) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (ext === 'mtl') continue;
    if (ext === 'json') {
      try {
        document.getElementById('loading').style.display = 'block';
        document.getElementById('loading').textContent = 'Loading scene...';
        const data = await importSceneState(file);
        await restoreScene(data);
        document.getElementById('loading').style.display = 'none';
      } catch (err) {
        console.error('Failed to load scene:', err);
        document.getElementById('loading').innerHTML =
          `<span style="color:#f88">Failed to load scene</span><br>` +
          `<span style="color:#aaa; font-size:0.85em">${err?.message || err}</span>`;
        setTimeout(() => { document.getElementById('loading').style.display = 'none'; }, 3000);
      }
    } else if (ext === 'stl') {
      loadSTLFile(file);
    } else if (ext === 'obj') {
      const mtlRef = file.name.replace(/\.obj$/i, '.mtl').toLowerCase();
      loadOBJFile(file, mtlFiles.get(mtlRef) || null);
    } else if (ext === 'ply') {
      loadPLYFile(file);
    } else if (ext === 'glb' || ext === 'gltf') {
      loadGLBFile(file);
    } else if (ext === 'splat' || ext === 'ksplat' || ext === 'spz') {
      loadSplatFile(file);
    }
  }
});

// Primitive buttons
document.getElementById('addCubeBtn').addEventListener('click',     () => addPrimitive('cube'));
document.getElementById('addSphereBtn').addEventListener('click',   () => addPrimitive('sphere'));
document.getElementById('addCylinderBtn').addEventListener('click', () => addPrimitive('cylinder'));

// STL transform mode buttons
document.getElementById('stlModeT').addEventListener('click',  () => setSTLTransformMode('translate'));
document.getElementById('stlModeR').addEventListener('click',  () => setSTLTransformMode('rotate'));
document.getElementById('stlModeS').addEventListener('click',  () => setSTLTransformMode('scale'));
document.getElementById('stlSpaceBtn').addEventListener('click', () => {
  const ctrl = State.stlTransformControls;
  const isLocal = ctrl.space === 'local';
  ctrl.setSpace(isLocal ? 'world' : 'local');
  const btn = document.getElementById('stlSpaceBtn');
  btn.textContent = isLocal ? 'World' : 'Local';
  btn.classList.toggle('active', !isLocal);
});
document.getElementById('stlDeselect').addEventListener('click', deselectSTL);

document.getElementById('lockAspectCb').addEventListener('change', (e) => {
  State.setLockAspect(e.target.checked);
});

document.getElementById('stlResetPos').addEventListener('click', () => {
  if (!State.selectedSTL) return;
  State.selectedSTL.mesh.position.set(0, 0, 0);
  syncSTLNumericInputs(State.selectedSTL);
});

document.getElementById('stlResetRot').addEventListener('click', () => {
  if (!State.selectedSTL) return;
  State.selectedSTL.mesh.rotation.set(0, 0, 0);
  syncSTLNumericInputs(State.selectedSTL);
});

document.getElementById('stlResetScale').addEventListener('click', () => {
  if (!State.selectedSTL) return;
  const s = State.selectedSTL.importScale;
  State.selectedSTL.mesh.scale.copy(s || new THREE.Vector3(1, 1, 1));
  syncSTLNumericInputs(State.selectedSTL);
});

// STL numeric position/rotation/scale inputs
function _applySTLNumericInputs() {
  const entry = State.selectedSTL;
  if (!entry) return;
  const m  = entry.mesh;
  const x  = parseFloat(document.getElementById('stlPosX').value) || 0;
  const y  = parseFloat(document.getElementById('stlPosY').value) || 0;
  const z  = parseFloat(document.getElementById('stlPosZ').value) || 0;
  const rx = parseFloat(document.getElementById('stlRotX').value) || 0;
  const ry = parseFloat(document.getElementById('stlRotY').value) || 0;
  const rz = parseFloat(document.getElementById('stlRotZ').value) || 0;
  const rawSx = parseFloat(document.getElementById('stlScX').value);
  const rawSy = parseFloat(document.getElementById('stlScY').value);
  const rawSz = parseFloat(document.getElementById('stlScZ').value);
  const sx = isNaN(rawSx) ? m.scale.x : rawSx;
  const sy = isNaN(rawSy) ? m.scale.z : rawSy;
  const sz = isNaN(rawSz) ? m.scale.y : rawSz;
  const d  = Math.PI / 180;
  m.position.set(x / 1000, z / 1000, y / 1000);
  m.rotation.set(rx * d, rz * d, ry * d);
  m.scale.set(sx, sz, sy);
}

['stlPosX', 'stlPosY', 'stlPosZ', 'stlRotX', 'stlRotY', 'stlRotZ', 'stlScX', 'stlScY', 'stlScZ'].forEach(id => {
  const el = document.getElementById(id);
  el.addEventListener('change', _applySTLNumericInputs);
  el.addEventListener('keydown', e => { if (e.key === 'Enter') { _applySTLNumericInputs(); el.blur(); } });
});

State.stlTransformControls.addEventListener('objectChange', () => {
  if (State.selectedSTL) syncSTLNumericInputs(State.selectedSTL);
});

document.getElementById('stlParentSelect').addEventListener('change', (e) => {
  if (State.selectedSTL) setSTLParent(State.selectedSTL, e.target.value, false);
});

// Collision button
const collisionBtn    = document.getElementById('collisionBtn');
const collisionInfoEl = document.getElementById('collision-info');
collisionBtn.addEventListener('click', () => {
  State.setCollisionEnabled(!State.collisionEnabled);
  collisionBtn.textContent = `Collision: ${State.collisionEnabled ? 'ON' : 'OFF'}`;
  collisionBtn.classList.toggle('active', State.collisionEnabled);
  collisionInfoEl.style.display = State.collisionEnabled ? 'block' : 'none';
  if (!State.collisionEnabled) clearCollisionHighlights();
});

// Floor collision toggle
const floorCollisionBtn = document.getElementById('floorCollisionBtn');
floorCollisionBtn.classList.add('active');
floorCollisionBtn.addEventListener('click', () => {
  State.setFloorCollisionEnabled(!State.floorCollisionEnabled);
  floorCollisionBtn.textContent = `Floor Collision: ${State.floorCollisionEnabled ? 'ON' : 'OFF'}`;
  floorCollisionBtn.classList.toggle('active', State.floorCollisionEnabled);
});

// Mesh-select toggle
const stlSelectBtn = document.getElementById('stlSelectBtn');
stlSelectBtn.classList.add('active');
stlSelectBtn.addEventListener('click', () => {
  State.setStlSelectable(!State.stlSelectable);
  stlSelectBtn.textContent = `Mesh Select: ${State.stlSelectable ? 'ON' : 'OFF'}`;
  stlSelectBtn.classList.toggle('active', State.stlSelectable);
});

// Floor size slider
document.getElementById('floorSize').addEventListener('input', (e) => {
  const r = parseFloat(e.target.value);
  document.getElementById('floorSizeVal').textContent = `${r} m`;
  setFloorSize(r);
});

// ============================================================
// Click-to-select STL meshes or activate devices
// ------------------------------------------------------------
// Selection runs on pointer-up only when the pointer did NOT move (a click in
// place), never on a drag. This matters because raycasting a THREE.Points cloud
// tests every point on the main thread — running that at the start of every
// orbit gesture stalled the initial frames of a rotation whenever a point cloud
// was present. A camera drag now skips the raycast entirely.
// ============================================================
const _CLICK_DRAG_PX = 5;
let _ptrDownX = 0, _ptrDownY = 0, _ptrDownValid = false;

State.renderer.domElement.addEventListener('pointerdown', (e) => {
  _ptrDownValid = false;
  if (State.stlTransformControls.dragging || State.transformControls.dragging || State.deviceTransformControls.dragging) return;
  if (e.button !== 0) return;
  _ptrDownX = e.clientX; _ptrDownY = e.clientY; _ptrDownValid = true;
});

State.renderer.domElement.addEventListener('pointerup', (e) => {
  if (!_ptrDownValid || e.button !== 0) return;
  _ptrDownValid = false;
  // Treat anything past the drag threshold as a camera move, not a selection.
  if (Math.hypot(e.clientX - _ptrDownX, e.clientY - _ptrDownY) > _CLICK_DRAG_PX) return;

  mouse.x = (e.clientX / innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, State.activeCamera);

  // Test against imported STL meshes first (if selection enabled). Point clouds
  // and splats are excluded — picking them would raycast every point on the main
  // thread; select those from the object list instead.
  if (State.stlSelectable) {
    const stlMeshes = State.importedSTLs
      .filter(s => s.mesh.visible && !s.isPointCloud && !s.isSplat)
      .map(s => s.mesh);
    const stlHits = raycaster.intersectObjects(stlMeshes, false);

    if (stlHits.length > 0) {
      const hitMesh = stlHits[0].object;
      const entry = State.importedSTLs.find(s => s.mesh === hitMesh);
      if (entry) {
        const listItems = document.querySelectorAll('.stl-item');
        const idx = State.importedSTLs.indexOf(entry);
        selectSTL(entry, listItems[idx] || null);
      }
      return;
    }
  }

  // No STL hit — test device meshes for activation.
  const allDeviceMeshes = [];
  for (const dev of State.devices) {
    for (const link of dev.robotLinkMeshes) {
      for (const mesh of link.meshes) {
        allDeviceMeshes.push(mesh);
      }
    }
    for (const mesh of dev.staticMeshes) {
      allDeviceMeshes.push(mesh);
    }
  }

  const deviceHits = raycaster.intersectObjects(allDeviceMeshes, false);
  if (deviceHits.length > 0) {
    const hitDev = findDeviceForObject(deviceHits[0].object);
    if (hitDev && hitDev !== State.activeDevice) {
      setActiveDevice(hitDev);
    }
  }

  // Clicked away from any STL — deselect (unless the gizmo itself was clicked).
  if (State.selectedSTL) {
    const gizmoHits = raycaster.intersectObjects(State.stlTransformControls.children, true);
    if (gizmoHits.length === 0) deselectSTL();
  }
});

// Keyboard shortcuts for STL transform
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  if (State.moveDeviceActive) {
    if (e.key === 't' || e.key === 'T') {
      State.deviceTransformControls.setMode('translate');
      document.getElementById('devModeT').classList.add('active');
      document.getElementById('devModeR').classList.remove('active');
    } else if (e.key === 'r' || e.key === 'R') {
      State.deviceTransformControls.setMode('rotate');
      document.getElementById('devModeR').classList.add('active');
      document.getElementById('devModeT').classList.remove('active');
    }
  } else if (State.selectedSTL) {
    if (e.key === 't' || e.key === 'T') {
      setSTLTransformMode('translate');
    } else if (e.key === 'r' || e.key === 'R') {
      setSTLTransformMode('rotate');
    } else if (e.key === 's' || e.key === 'S') {
      setSTLTransformMode('scale');
    } else if (e.key === 'Escape') {
      deselectSTL();
    }
  } else if (State.activeDevice && State.activeDevice.ikMode) {
    const tc = State.transformControls;
    if (e.key === 't' || e.key === 'T') {
      tc.setMode('translate');
      tc.showX = true; tc.showY = true; tc.showZ = true;
    } else if (e.key === 'r' || e.key === 'R') {
      // Cycle Rx → Ry → Rz → Rx
      if (tc.mode === 'rotate') {
        if (tc.showX && !tc.showY && !tc.showZ)      { tc.showX = false; tc.showY = true; }
        else if (!tc.showX && tc.showY && !tc.showZ)  { tc.showY = false; tc.showZ = true; }
        else                                           { tc.showX = true; tc.showY = false; tc.showZ = false; }
      } else {
        tc.setMode('rotate');
        tc.showX = true; tc.showY = false; tc.showZ = false;
      }
    }
  }
});

// ============================================================
// Navigation gizmo mouse events
// ============================================================
navCanvas.addEventListener('mousemove', (e) => {
  const rect = navCanvas.getBoundingClientRect();
  const id = navHitTest(e.clientX - rect.left, e.clientY - rect.top);
  if (id !== navHovered) {
    setNavHovered(id);
    navCanvas.style.cursor = id ? 'pointer' : 'default';
    State.requestRender();
  }
});
navCanvas.addEventListener('mouseleave', () => {
  setNavHovered(null); navCanvas.style.cursor = 'default'; State.requestRender();
});

const _snapVec = new THREE.Vector3();

navCanvas.addEventListener('click', (e) => {
  const rect = navCanvas.getBoundingClientRect();
  const id = navHitTest(e.clientX - rect.left, e.clientY - rect.top);
  if (!id) return;
  const ax = NAV_AXES.find(a => a.id === id);
  const dist = State.activeCamera.position.distanceTo(State.orbitControls.target);
  setSnapAnim({
    sp: State.activeCamera.position.clone(),
    ep: State.orbitControls.target.clone().addScaledVector(ax.dir, dist),
    su: State.activeCamera.up.clone(),
    eu: ax.up.clone(),
    t: 0,
  });
  State.orbitControls.enabled = false;
});

// ============================================================
// Resize handler
// ============================================================
window.addEventListener('resize', () => {
  State.camera.aspect = innerWidth / innerHeight;
  State.camera.updateProjectionMatrix();
  if (State.orthoOn) updateOrthoFrustum();
  State.renderer.setPixelRatio(devicePixelRatio);
  State.renderer.setSize(innerWidth, innerHeight);
  State.labelRenderer.setSize(innerWidth, innerHeight);
  State.requestRender();
});

// ============================================================
// On-demand render triggers
// ------------------------------------------------------------
// Any user interaction may change the view or the scene (sliders,
// buttons, list clicks, gizmo drags, hover highlights). Rather than
// instrument every handler, a single set of input listeners requests a
// render whenever the user does something. Camera moves, IK, snap
// animation, async loads and websocket updates request renders at their
// own sources. With nothing happening, the loop draws nothing.
// ============================================================
['pointerdown', 'pointerup', 'wheel'].forEach((ev) =>
  window.addEventListener(ev, () => State.requestRender(), { passive: true }));
window.addEventListener('pointermove', (e) => {
  if (e.buttons) State.requestRender();   // only while dragging
}, { passive: true });
window.addEventListener('keydown', () => State.requestRender());

// ============================================================
// Initialization
// ============================================================
// LidarStudio starts as an empty editor scene; a robot only loads if explicitly
// requested via ?config=<name>. "none" (the default) means no device.
const configParam = new URLSearchParams(window.location.search).get('config') || 'none';

// Try restoring from IndexedDB first, then fall back to localStorage for legacy data
const SCENE_STORAGE_KEY = 'robotvis_scene';
let restoredFromStorage = false;

// Signature of the buffer set already persisted under BUFFERS_KEY. Primed after
// a split-format restore so the first auto-save doesn't needlessly re-clone the
// buffers we just loaded; left null when nothing usable was loaded (incl. legacy
// combined records) so the next auto-save migrates buffers into BUFFERS_KEY.
let _lastSavedBufferSig = null;
// True once a restore has merged buffers from the dedicated BUFFERS_KEY record.
let _loadedSplitBuffers = false;

const MAX_RESTORE_ATTEMPTS = 3;

async function _tryLoadSavedScene() {
  // 1. Prefer IndexedDB (supports large mesh buffers)
  try {
    const dbData = await dbLoad();
    if (dbData && dbData.version && Array.isArray(dbData.devices) && dbData.devices.length > 0) {
      // New split format: stl records carry metadata only — merge the heavy
      // buffers back in from their dedicated record, matched by stl id.
      // Legacy combined records already have inline buffers and skip this.
      if (Array.isArray(dbData.stls) && dbData.stls.some(s => !s.buffer)) {
        try {
          const bufRec = await dbLoad(BUFFERS_KEY);
          if (bufRec && Array.isArray(bufRec.buffers)) {
            const byId = new Map(bufRec.buffers.map(b => [b.id, b.buffer]));
            for (const s of dbData.stls) {
              if (!s.buffer && byId.has(s.id)) s.buffer = byId.get(s.id);
            }
            _loadedSplitBuffers = true;
          }
        } catch (e) {
          console.warn('[Auto-restore] Buffer record read failed:', e);
        }
      }
      return dbData;
    }
  } catch (e) {
    console.warn('[Auto-restore] IndexedDB read failed:', e);
  }
  // 2. Fall back to localStorage (legacy saves or small scenes)
  try {
    const raw = localStorage.getItem(SCENE_STORAGE_KEY);
    if (raw) {
      const data = JSON.parse(raw);
      if (data && data.version && Array.isArray(data.devices) && data.devices.length > 0) {
        console.log('[Auto-restore] Using legacy localStorage save');
        return data;
      }
    }
  } catch (e) {
    console.warn('[Auto-restore] localStorage read failed:', e);
  }
  return null;
}

// LidarStudio loads clouds/splats explicitly (Library / Load Scene / Load
// Workflow), so the page does not auto-restore a saved scene on load. (Disabled
// the inherited robot-viewer auto-restore, which retried up to 3 times.)
const savedData = null;

if (savedData) {
  for (let attempt = 1; attempt <= MAX_RESTORE_ATTEMPTS && !restoredFromStorage; attempt++) {
    try {
      document.getElementById('loading').textContent =
        attempt === 1 ? 'Restoring scene...' : `Restoring scene (attempt ${attempt}/${MAX_RESTORE_ATTEMPTS})...`;
      await restoreScene(savedData);
      restoredFromStorage = true;
      console.log(`[Auto-restore] Scene restored (attempt ${attempt})`);
    } catch (err) {
      console.warn(`[Auto-restore] Attempt ${attempt}/${MAX_RESTORE_ATTEMPTS} failed:`, err);
      for (const dev of [...State.devices]) {
        State.transformControls.detach();
        State.deviceTransformControls.detach();
        State.scene.remove(dev.rootGroup);
      }
      State.devices.length = 0;
      State.resetDeviceIdCounter();
      if (attempt < MAX_RESTORE_ATTEMPTS) {
        await new Promise(r => setTimeout(r, 1000 * attempt));
      }
    }
  }
  if (!restoredFromStorage) {
    console.warn('[Auto-restore] All attempts failed — starting fresh');
  } else if (_loadedSplitBuffers) {
    // Buffers came from BUFFERS_KEY unchanged — record their signature so the
    // first auto-save skips the redundant re-write.
    _lastSavedBufferSig = sceneBufferSignature();
  }
}

if (!restoredFromStorage && configParam && configParam !== 'none') {
  try {
    const initialDevice = await loadDevice(configParam);
    State.devices.push(initialDevice);
    State.setActiveDevice(initialDevice);
    if (initialDevice.type === 'hexapod') updateHexapodPose(initialDevice);
    else updateFK(initialDevice);
    buildControlPanel(initialDevice);
    rebuildDeviceList();
    rebuildPrimaryModelDropdown(configParam);

    // Auto-fit camera
    const fitBox = new THREE.Box3();
    initialDevice.rootGroup.updateWorldMatrix(true, true);
    initialDevice.rootGroup.traverse((child) => {
      if (child.isMesh) fitBox.expandByObject(child);
    });
    if (!fitBox.isEmpty()) {
      const fitCenter = fitBox.getCenter(new THREE.Vector3());
      const fitSize   = fitBox.getSize(new THREE.Vector3());
      const maxDim    = Math.max(fitSize.x, fitSize.y, fitSize.z);
      const fov       = State.camera.fov * (Math.PI / 180);
      const fitDist   = maxDim / (2 * Math.tan(fov / 2)) * 1.2;
      const direction = new THREE.Vector3(1, 0.6, 1).normalize();
      State.camera.position.copy(fitCenter).addScaledVector(direction, fitDist);
      State.orbitControls.target.copy(fitCenter);
      State.camera.updateProjectionMatrix();
      State.orbitControls.update();
    }
  } catch (err) {
    console.error('Failed to load initial device:', err);
    const msg = err?.message || String(err);
    document.getElementById('loading').innerHTML =
      `<span style="color:#f88">Failed to load <b>${configParam}</b></span><br>` +
      `<span style="color:#aaa; font-size:0.85em">${msg}</span><br>` +
      `<span style="color:#aaa; font-size:0.85em">Check the browser console (F12) and server logs for details.</span>`;
  }
}

document.getElementById('loading').style.display = 'none';
State.requestRender();

window.debugHome = () => {
  const d = State.activeDevice;
  if (!d) return null;
  const m = new THREE.Matrix4().makeRotationFromQuaternion(d.homeQuaternion).elements;
  return [[m[0],m[4],m[8]],[m[1],m[5],m[9]],[m[2],m[6],m[10]]];
};
window.debugEE = () => {
  const d = State.activeDevice;
  if (!d) return null;
  State.scene.updateMatrixWorld(true);
  const m = d.eeMarker.matrixWorld.elements;
  const p = d.eeMarker.getWorldPosition(new THREE.Vector3());
  return { pos: [p.x, p.y, p.z],
           R: [[m[0],m[4],m[8]],[m[1],m[5],m[9]],[m[2],m[6],m[10]]] };
};

// Auto-save scene to IndexedDB periodically and on page unload.
// IndexedDB handles large mesh buffers that would overflow localStorage's ~5 MB quota.
//
// The heavy mesh/point-cloud/splat buffers are stored separately and only
// re-written when they actually change (tracked by a cheap signature). Routine
// saves — fired every 30 s and while the camera/joints move — then write only
// the small metadata record, avoiding the multi-MB IndexedDB structured clone
// that previously hitched the frame (and stalled rotation) on every tick.
// (_lastSavedBufferSig is declared up near the restore code so it can be primed
//  after a restore that already loaded the buffers in the split format.)

async function autoSaveScene() {
  if (State.devices.length === 0) return;
  try {
    const sig = sceneBufferSignature();
    if (sig !== _lastSavedBufferSig) {
      // Buffers changed (import/remove/restore) — persist them first so the
      // metadata record never references buffers that aren't on disk yet.
      await dbSave(buildSceneBuffersForDB(), BUFFERS_KEY);
      _lastSavedBufferSig = sig;
    }
    await dbSave(buildSceneMetadataForDB());
  } catch (e) {
    console.warn('[Auto-save] IndexedDB save failed:', e);
  }
}
setInterval(autoSaveScene, 30_000);
window.addEventListener('beforeunload', () => autoSaveScene());
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') autoSaveScene();
});

// Screenshot button — composite WebGL canvas with the control panel
document.getElementById('screenshotBtn').addEventListener('click', async () => {
  State.orbitControls.update();
  State.renderer.render(State.scene, State.activeCamera);

  const glCanvas = State.renderer.domElement;
  const out = document.createElement('canvas');
  out.width = glCanvas.width;
  out.height = glCanvas.height;
  const ctx = out.getContext('2d');
  ctx.drawImage(glCanvas, 0, 0, out.width, out.height);

  const panel = document.getElementById('panel');
  if (panel && typeof html2canvas === 'function') {
    try {
      const scale = glCanvas.width / window.innerWidth;
      const panelCanvas = await html2canvas(panel, {
        backgroundColor: null,
        scale,
        logging: false,
      });
      const rect = panel.getBoundingClientRect();
      ctx.drawImage(panelCanvas, rect.left * scale, rect.top * scale);
    } catch (e) {
      console.warn('[Screenshot] panel capture failed:', e);
    }
  }

  const a = document.createElement('a');
  const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  a.download = `screenshot_${ts}.png`;
  a.href = out.toDataURL('image/png');
  a.click();
});

// Save / Load scene buttons
document.getElementById('saveSceneBtn').addEventListener('click', () => exportSceneState());

document.getElementById('loadSceneBtn').addEventListener('click', () => {
  document.getElementById('loadSceneFile').click();
});

document.getElementById('clearSceneBtn').addEventListener('click', () => {
  if (!confirm('Clear the entire scene? This cannot be undone.')) return;

  // Clear imported STLs
  for (const entry of [...State.importedSTLs]) {
    if (State.selectedSTL === entry) deselectSTL();
    entry.mesh.removeFromParent();
    if (entry.isSplat) {
      if (entry._splatViewer) entry._splatViewer.dispose();
      if (entry._blobUrl) URL.revokeObjectURL(entry._blobUrl);
      if (entry._collisionPoints) {
        entry._collisionPoints.geometry.dispose();
        entry._collisionPoints.material.dispose();
      }
    } else {
      entry.mesh.geometry.dispose();
      entry.mesh.material.dispose();
    }
  }
  State.importedSTLs.length = 0;
  document.getElementById('stl-list').innerHTML = '';

  // Clear devices
  for (const dev of [...State.devices]) {
    State.transformControls.detach();
    State.deviceTransformControls.detach();
    State.scene.remove(dev.rootGroup);
    dev.rootGroup.traverse(child => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
        else child.material.dispose();
      }
    });
    for (const label of dev.meshLabels) label.removeFromParent();
    if (dev.chainLine) State.scene.remove(dev.chainLine);
    for (const s of dev.chainSpheres) State.scene.remove(s);
    if (dev.ikTarget) State.scene.remove(dev.ikTarget);
    if (dev.ikLine) State.scene.remove(dev.ikLine);
  }
  State.devices.length = 0;
  State.resetDeviceIdCounter();

  rebuildDeviceList();
  rebuildPrimaryModelDropdown('meca500_config.json');
});

// ============================================================
// Core scene restore (used by file load and localStorage restore)
// ============================================================
async function restoreScene(data) {
  // Clear existing imported STLs
  for (const entry of [...State.importedSTLs]) {
    if (State.selectedSTL === entry) deselectSTL();
    entry.mesh.removeFromParent();
    if (entry.isSplat) {
      if (entry._splatViewer) entry._splatViewer.dispose();
      if (entry._blobUrl) URL.revokeObjectURL(entry._blobUrl);
      if (entry._collisionPoints) {
        entry._collisionPoints.geometry.dispose();
        entry._collisionPoints.material.dispose();
      }
    } else {
      entry.mesh.geometry.dispose();
      entry.mesh.material.dispose();
    }
  }
  State.importedSTLs.length = 0;
  document.getElementById('stl-list').innerHTML = '';

  // Clear existing devices and reload from state
  for (const dev of [...State.devices]) {
    State.transformControls.detach();
    State.deviceTransformControls.detach();
    State.scene.remove(dev.rootGroup);
    dev.rootGroup.traverse(child => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
        else child.material.dispose();
      }
    });
    for (const label of dev.meshLabels) label.removeFromParent();
    if (dev.chainLine) State.scene.remove(dev.chainLine);
    for (const s of dev.chainSpheres) State.scene.remove(s);
    if (dev.ikTarget) State.scene.remove(dev.ikTarget);
    if (dev.ikLine) State.scene.remove(dev.ikLine);
  }
  State.devices.length = 0;
  State.resetDeviceIdCounter();

  // Reload devices from state
  if (data.devices && data.devices.length > 0) {
    for (const devState of data.devices) {
      const dev = await loadDevice(devState.configFile);
      State.devices.push(dev);
      if (devState.name) dev.name = devState.name;
      if (devState.jointAngles) {
        for (let i = 0; i < devState.jointAngles.length && i < dev.jointAngles.length; i++) {
          dev.jointAngles[i] = devState.jointAngles[i];
        }
      }
      if (devState.platformPose && dev.type === 'hexapod') {
        for (let i = 0; i < 6; i++) dev.platformPose[i] = devState.platformPose[i] || 0;
      }
      if (devState.position) {
        dev.rootGroup.position.set(...devState.position);
      }
      if (devState.rotation) {
        dev.rootGroup.rotation.set(...devState.rotation);
      }
      if (dev.type === 'hexapod') updateHexapodPose(dev);
      else updateFK(dev);
      console.log('[Load Scene] Device:', dev.name, 'id:', dev.id,
        'joints:', dev.jointAngles.map(a => (a * 180 / Math.PI).toFixed(1)),
        'pos:', [dev.rootGroup.position.x, dev.rootGroup.position.y, dev.rootGroup.position.z]);
    }
    // Restore device parent links (must happen after all devices are loaded)
    for (let i = 0; i < data.devices.length; i++) {
      const parentLink = data.devices[i].parentLink;
      if (parentLink && parentLink.includes(':')) {
        const [idxStr, linkName] = parentLink.split(':', 2);
        const parentIdx = parseInt(idxStr, 10);
        let runtimeLink = null;
        if (!isNaN(parentIdx) && parentIdx >= 0 && parentIdx < State.devices.length) {
          runtimeLink = State.devices[parentIdx].id + ':' + linkName;
        } else {
          // Legacy format — search by link name
          for (const dev of State.devices) {
            if (dev !== State.devices[i] && dev.linkToJoint && dev.linkToJoint[linkName] !== undefined) {
              runtimeLink = dev.id + ':' + linkName;
              break;
            }
          }
        }
        if (runtimeLink) setDeviceParent(State.devices[i], runtimeLink, true);
      }
    }
    State.setActiveDevice(State.devices[0]);
    buildControlPanel(State.devices[0]);
  }

  // Restore floor size
  if (data.floorSize != null) {
    setFloorSize(data.floorSize);
    document.getElementById('floorSize').value = data.floorSize;
    document.getElementById('floorSizeVal').textContent = `${data.floorSize} m`;
  }

  // Restore camera
  if (data.camera) {
    if (data.camera.position) State.camera.position.set(...data.camera.position);
    if (data.camera.target) State.orbitControls.target.set(...data.camera.target);
    State.camera.updateProjectionMatrix();
    State.orbitControls.update();
  }

  // Ensure full scene graph is updated before restoring STLs
  State.scene.updateMatrixWorld(true);

  // Restore STLs (two-phase: create, then apply transforms)
  if (data.stls && data.stls.length > 0) {
    await restoreSTLsFromState(data.stls);
  }

  rebuildDeviceList();
  rebuildPrimaryModelDropdown(State.devices[0]?.configFile || 'meca500_config.json');
}

document.getElementById('loadSceneFile').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    document.getElementById('loading').style.display = 'block';
    document.getElementById('loading').textContent = 'Loading scene...';

    const data = await importSceneState(file);
    await restoreScene(data);

    document.getElementById('loading').style.display = 'none';
  } catch (err) {
    console.error('Failed to load scene:', err);
    document.getElementById('loading').innerHTML =
      `<span style="color:#f88">Failed to load scene</span><br>` +
      `<span style="color:#aaa; font-size:0.85em">${err?.message || err}</span>`;
    setTimeout(() => { document.getElementById('loading').style.display = 'none'; }, 3000);
  }
  e.target.value = '';
});

// Start VR support
initVR();

// Start render loop
State.renderer.setAnimationLoop(animate);

// Start WebSocket
wsConnect();
initWsInfoPanel();
