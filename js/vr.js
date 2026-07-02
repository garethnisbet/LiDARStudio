// ============================================================
// js/vr.js — WebXR VR interaction (Meta Quest controller conventions)
//
// Controller mapping (Meta Quest guidelines):
//   Trigger (select)    = ray-select: UI interaction, object selection
//   Grip (squeeze)      = grab objects (meshes, clouds, splats)
//   Left thumbstick     = locomotion: up=teleport arc, L/R=snap turn
//   Right thumbstick    = scroll panel (Y), snap turn (X) when not on panel
//   Thumbstick press    = toggle passthrough (AR camera feed)
//   B / Y button        = toggle / reposition VR panel
// ============================================================
import * as THREE from 'three';
import { XRControllerModelFactory } from 'three/addons/webxr/XRControllerModelFactory.js';
import { HTMLMesh } from 'three/addons/interactive/HTMLMesh.js';
import { InteractiveGroup } from 'three/addons/interactive/InteractiveGroup.js';
import * as State from './state.js';
import { setOrtho } from './scene.js';
import { selectSTL } from './stl.js';
import { dbSaveVRAnchor, dbLoadVRAnchor } from './storage.js';

const _raycaster = new THREE.Raycaster();
const _tempMatrix = new THREE.Matrix4();
const _worldPos = new THREE.Vector3();
const _groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const _intersection = new THREE.Vector3();

let teleportMarker;
let teleportArc;
let savedCamPos, savedCamQuat, savedTarget;
let activeGrab = null;
let bgSphere = null;

// Anchor-based drift correction (survives headset removal/re-don)
let sceneAnchor = null;
let needsAnchor = false;
let lastAnchorPos = null;
let lastAnchorQuat = null;

// Persistent anchor — survives across VR sessions using room environment map
let persistentHandle = null;
let preloadedAnchorData = null;
let savedVRAnchorData = null;
let rigRepositioned = false;
const VR_ANCHOR_SAVE_INTERVAL = 30_000;
let lastAnchorSave = 0;

const controllers = [];

// VR panel
let vrPanelMesh = null;
let interactiveGroup = null;
let panelVisible = false;
let lastPanelRefresh = 0;
const PANEL_REFRESH_INTERVAL = 500;

// Locomotion state
let teleportActive = false;
let teleportController = null;

// Gamepad state for edge-detection
const prevButtons = [{}, {}]; // keyed by button index
const STICK_DEADZONE = 0.2;
const SNAP_ANGLE = Math.PI / 6; // 30 degrees
let snapTurnCooldown = 0;
const SCROLL_SPEED = 400;

// Teleport arc constants
const ARC_SEGMENTS = 30;
const ARC_GRAVITY = -9.8;
const ARC_VELOCITY = 5.0;

export async function initVR() {
  const renderer = State.renderer;

  if (!navigator.xr) return;

  const arSupported = await navigator.xr.isSessionSupported('immersive-ar').catch(() => false);
  const vrSupported = await navigator.xr.isSessionSupported('immersive-vr').catch(() => false);
  const xrMode = arSupported ? 'immersive-ar' : vrSupported ? 'immersive-vr' : null;

  renderer.xr.enabled = !!xrMode;

  if (!xrMode) return;

  try { preloadedAnchorData = await dbLoadVRAnchor(); } catch (e) {}

  let currentSession = null;
  const btn = document.createElement('button');
  btn.textContent = 'ENTER VR';
  btn.style.cssText = 'position:absolute;bottom:20px;left:calc(50% - 50px);width:100px;padding:12px 6px;border:1px solid #fff;border-radius:4px;background:rgba(0,0,0,0.1);color:#fff;font:normal 13px sans-serif;text-align:center;cursor:pointer;opacity:0.5;z-index:9999;transition:opacity 1s ease;';
  btn.onmouseenter = () => { btn.style.opacity = '1.0'; };
  btn.onmouseleave = () => { btn.style.opacity = '0.5'; };

  btn.onclick = async () => {
    if (currentSession) { currentSession.end(); return; }
    const sessionOpts = {
      optionalFeatures: ['local-floor', 'bounded-floor', 'layers', 'hand-tracking', 'anchors', 'persistent-anchors'],
    };
    const session = await navigator.xr.requestSession(xrMode, sessionOpts);
    session.addEventListener('end', () => { currentSession = null; btn.textContent = 'ENTER VR'; });
    // Splat rendering is heavily overdraw-bound on mobile GPUs (Quest). Render
    // the eye buffers at reduced resolution and apply maximum foveation so the
    // periphery shades fewer fragments. Both must be set before setSession().
    renderer.xr.setFramebufferScaleFactor(0.8);
    renderer.xr.setFoveation(1);
    await renderer.xr.setSession(session);
    currentSession = session;
    btn.textContent = 'EXIT VR';
  };
  document.body.appendChild(btn);

  const exitBtn = document.getElementById('exitVRBtn');
  if (exitBtn) {
    exitBtn.addEventListener('click', () => {
      if (currentSession) currentSession.end();
    });
  }

  const rig = new THREE.Group();
  rig.name = 'VRCameraRig';
  State.scene.add(rig);
  rig.add(State.camera);
  State.setVRRig(rig);

  const factory = new XRControllerModelFactory();

  for (let i = 0; i < 2; i++) {
    const controller = renderer.xr.getController(i);
    controller.userData.index = i;
    rig.add(controller);

    const grip = renderer.xr.getControllerGrip(i);
    grip.add(factory.createControllerModel(grip));
    rig.add(grip);

    const rayGeo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 0, -5),
    ]);
    const color = i === 0 ? 0x4488ff : 0x44ff88;
    const rayLine = new THREE.Line(rayGeo,
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.5 }));
    rayLine.name = 'vr-ray';
    controller.add(rayLine);

    // Trigger = ray select (UI, device selection)
    controller.addEventListener('selectstart', () => onTriggerStart(controller));
    controller.addEventListener('selectend', () => onTriggerEnd(controller));

    // Grip = grab objects
    controller.addEventListener('squeezestart', () => onGripStart(controller));
    controller.addEventListener('squeezeend', () => onGripEnd(controller));

    controllers.push(controller);
  }

  // Teleport marker
  const ringGeo = new THREE.RingGeometry(0.05, 0.09, 32);
  ringGeo.rotateX(-Math.PI / 2);
  const discGeo = new THREE.CircleGeometry(0.05, 32);
  discGeo.rotateX(-Math.PI / 2);
  teleportMarker = new THREE.Group();
  teleportMarker.add(new THREE.Mesh(ringGeo,
    new THREE.MeshBasicMaterial({ color: 0x44ff88, side: THREE.DoubleSide })));
  teleportMarker.add(new THREE.Mesh(discGeo,
    new THREE.MeshBasicMaterial({ color: 0x44ff88, transparent: true, opacity: 0.3, side: THREE.DoubleSide })));
  teleportMarker.visible = false;
  State.scene.add(teleportMarker);

  // Teleport arc line
  const arcPositions = new Float32Array(ARC_SEGMENTS * 3);
  const arcGeo = new THREE.BufferGeometry();
  arcGeo.setAttribute('position', new THREE.BufferAttribute(arcPositions, 3));
  teleportArc = new THREE.Line(arcGeo,
    new THREE.LineBasicMaterial({ color: 0x44ff88, transparent: true, opacity: 0.6 }));
  teleportArc.frustumCulled = false;
  teleportArc.visible = false;
  State.scene.add(teleportArc);

  // Interactive group for VR panel (HTMLMesh must be a direct child)
  interactiveGroup = new InteractiveGroup();
  interactiveGroup.listenToXRControllerEvents(controllers[0]);
  interactiveGroup.listenToXRControllerEvents(controllers[1]);
  State.scene.add(interactiveGroup);

  renderer.xr.addEventListener('sessionstart', onSessionStart);
  renderer.xr.addEventListener('sessionend', onSessionEnd);
}

// ============================================================
// Session lifecycle
// ============================================================

function onSessionStart() {
  State.setVRActive(true);
  if (State.orthoOn) setOrtho(false);

  State.scene.background = null;
  setPassthrough(true);

  savedCamPos = State.camera.position.clone();
  savedCamQuat = State.camera.quaternion.clone();
  savedTarget = State.orbitControls.target.clone();

  const rig = State.vrRig;
  rig.position.set(savedTarget.x, 0, savedTarget.z);
  State.camera.position.set(0, 0, 0);
  State.camera.quaternion.identity();
  State.orbitControls.enabled = false;

  sceneAnchor = null;
  lastAnchorPos = null;
  lastAnchorQuat = null;
  persistentHandle = null;
  savedVRAnchorData = preloadedAnchorData;
  rigRepositioned = false;
  lastAnchorSave = 0;
  needsAnchor = true;

  const exitBtn = document.getElementById('exitVRBtn');
  if (exitBtn) exitBtn.style.display = '';

  createVRPanel();
}

function onSessionEnd() {
  State.setVRActive(false);
  State.requestRender();   // resume drawing the desktop view

  saveVRAnchorState();
  if (sceneAnchor && !persistentHandle) sceneAnchor.delete();
  sceneAnchor = null;
  lastAnchorPos = null;
  lastAnchorQuat = null;
  persistentHandle = null;
  needsAnchor = false;

  State.scene.background = new THREE.Color(0x2a2a3a);
  State.setPassthroughOn(false);
  if (bgSphere) bgSphere.visible = false;

  const exitBtn = document.getElementById('exitVRBtn');
  if (exitBtn) exitBtn.style.display = 'none';

  const rig = State.vrRig;
  State.camera.position.copy(savedCamPos);
  State.camera.quaternion.copy(savedCamQuat);
  rig.position.set(0, 0, 0);
  rig.quaternion.identity();

  State.orbitControls.target.copy(savedTarget);
  State.orbitControls.enabled = true;
  State.orbitControls.update();

  teleportMarker.visible = false;
  teleportArc.visible = false;
  teleportActive = false;
  activeGrab = null;
  destroyVRPanel();
}

// ============================================================
// Passthrough toggle
// ============================================================

function setPassthrough(on) {
  State.setPassthroughOn(on);
  if (on) {
    State.scene.background = null;
    if (bgSphere) bgSphere.visible = false;
  } else {
    State.scene.background = null;
    if (!bgSphere) {
      const geo = new THREE.SphereGeometry(20, 32, 16);
      const mat = new THREE.MeshBasicMaterial({ color: 0x2a2a3a, side: THREE.BackSide });
      bgSphere = new THREE.Mesh(geo, mat);
      bgSphere.renderOrder = -1;
      State.scene.add(bgSphere);
    }
    bgSphere.visible = true;
  }
}

function togglePassthrough() {
  setPassthrough(!State.passthroughOn);
}

// ============================================================
// VR Panel (HTMLMesh of the control panel)
// ============================================================

// Overflow styles stripped for html2canvas compatibility; restored on session end
const _vrOverflowSaved = [];

function _stripOverflow() {
  const ids = ['panel', 'stl-list'];
  for (const id of ids) {
    const el = document.getElementById(id);
    if (!el) continue;
    _vrOverflowSaved.push({ el, overflowY: el.style.overflowY, maxHeight: el.style.maxHeight });
    el.style.overflowY = 'visible';
    el.style.maxHeight = 'none';
  }
}

function _restoreOverflow() {
  for (const { el, overflowY, maxHeight } of _vrOverflowSaved) {
    el.style.overflowY = overflowY;
    el.style.maxHeight = maxHeight;
  }
  _vrOverflowSaved.length = 0;
}

function createVRPanel() {
  destroyVRPanel();

  const panelEl = document.getElementById('panel');
  if (!panelEl) return;

  _stripOverflow();
  vrPanelMesh = new HTMLMesh(panelEl);
  vrPanelMesh.scale.setScalar(2);
  interactiveGroup.add(vrPanelMesh);
  positionPanelInFront();
  vrPanelMesh.visible = true;
  panelVisible = true;
}

function destroyVRPanel() {
  if (vrPanelMesh) {
    interactiveGroup.remove(vrPanelMesh);
    vrPanelMesh.dispose();
    vrPanelMesh = null;
  }
  _restoreOverflow();
  panelVisible = false;
}

function positionPanelInFront() {
  if (!vrPanelMesh) return;

  const cam = State.camera;
  const camWorld = new THREE.Vector3();
  const camDir = new THREE.Vector3();
  cam.getWorldPosition(camWorld);
  cam.getWorldDirection(camDir);

  camDir.y = 0;
  camDir.normalize();
  const pos = camWorld.clone().add(camDir.multiplyScalar(1.2));
  pos.y = camWorld.y - 0.2;

  vrPanelMesh.position.copy(pos);
  vrPanelMesh.lookAt(camWorld.x, pos.y, camWorld.z);
}

function toggleVRPanel() {
  if (!vrPanelMesh) {
    createVRPanel();
    return;
  }
  if (panelVisible) {
    panelVisible = false;
    vrPanelMesh.visible = false;
  } else {
    destroyVRPanel();
    createVRPanel();
  }
}

function refreshVRPanel() {
  if (!vrPanelMesh || !panelVisible) return;
  destroyVRPanel();
  createVRPanel();
}

// ============================================================
// Raycasting helpers
// ============================================================

function castRay(controller) {
  _tempMatrix.identity().extractRotation(controller.matrixWorld);
  _raycaster.ray.origin.setFromMatrixPosition(controller.matrixWorld);
  _raycaster.ray.direction.set(0, 0, -1).applyMatrix4(_tempMatrix);
}

function isPointingAtPanel(controller) {
  if (!vrPanelMesh || !panelVisible) return false;
  castRay(controller);
  return _raycaster.intersectObject(vrPanelMesh, false).length > 0;
}

// ============================================================
// Teleport arc (parabolic, Meta Quest style)
// ============================================================

function computeArc(controller) {
  _tempMatrix.identity().extractRotation(controller.matrixWorld);
  const origin = new THREE.Vector3().setFromMatrixPosition(controller.matrixWorld);
  const dir = new THREE.Vector3(0, 0, -1).applyMatrix4(_tempMatrix).normalize();

  const vel = dir.multiplyScalar(ARC_VELOCITY);
  const dt = 0.05;
  const pts = [];
  const pos = origin.clone();

  for (let i = 0; i < ARC_SEGMENTS; i++) {
    pts.push(pos.clone());
    pos.x += vel.x * dt;
    pos.y += vel.y * dt;
    pos.z += vel.z * dt;
    vel.y += ARC_GRAVITY * dt;

    if (pos.y <= 0) {
      pos.y = 0;
      pts.push(pos.clone());
      return { pts, landing: pos.clone() };
    }
  }
  return { pts, landing: null };
}

function updateTeleportArc(controller) {
  const { pts, landing } = computeArc(controller);

  const posAttr = teleportArc.geometry.attributes.position;
  for (let i = 0; i < ARC_SEGMENTS; i++) {
    const p = i < pts.length ? pts[i] : pts[pts.length - 1];
    posAttr.setXYZ(i, p.x, p.y, p.z);
  }
  posAttr.needsUpdate = true;
  teleportArc.geometry.setDrawRange(0, pts.length);
  teleportArc.visible = true;

  if (landing) {
    teleportMarker.position.copy(landing);
    teleportMarker.position.y = 0.005;
    teleportMarker.visible = true;
  } else {
    teleportMarker.visible = false;
  }
}

// ============================================================
// Trigger (select) — ray-based UI & scene interaction
// ============================================================

function onTriggerStart(controller) {
  castRay(controller);

  // Panel interaction — let InteractiveGroup handle it
  if (vrPanelMesh && panelVisible) {
    const panelHits = _raycaster.intersectObject(vrPanelMesh, false);
    if (panelHits.length > 0) {
      controller.userData.hitType = 'panel';
      return;
    }
  }

  // STL mesh selection
  if (State.stlSelectable) {
    const stlMeshes = State.importedSTLs.filter(s => s.mesh.visible).map(s => s.mesh);
    const stlHits = _raycaster.intersectObjects(stlMeshes, true);
    if (stlHits.length > 0 && stlHits[0].distance < 10) {
      const hitMesh = stlHits[0].object;
      const entry = State.importedSTLs.find(s => {
        if (s.mesh === hitMesh) return true;
        let found = false;
        s.mesh.traverse(c => { if (c === hitMesh) found = true; });
        return found;
      });
      if (entry) {
        const listItems = document.querySelectorAll('.stl-item');
        const idx = State.importedSTLs.indexOf(entry);
        selectSTL(entry, listItems[idx] || null);
        refreshVRPanel();
      }
      controller.userData.hitType = 'stl';
      return;
    }
  }

  controller.userData.hitType = 'none';
}

function onTriggerEnd(controller) {
  controller.userData.hitType = null;
}

// ============================================================
// Grip (squeeze) — grab objects with either hand
// ============================================================

function onGripStart(controller) {
  if (activeGrab) return;
  castRay(controller);

  // Collect all grabbable candidates
  const candidates = State.importedSTLs.filter(s => s.mesh.visible).map(s => s.mesh);

  const hits = _raycaster.intersectObjects(candidates, true);
  if (hits.length > 0 && hits[0].distance < 5) {
    const hitObj = hits[0].object;
    let target = null;
    for (const s of State.importedSTLs) {
      if (s.mesh === hitObj) { target = s.mesh; break; }
      let found = false;
      s.mesh.traverse(c => { if (c === hitObj) found = true; });
      if (found) { target = s.mesh; break; }
    }
    if (!target) target = hitObj;

    controller.getWorldPosition(_worldPos);
    const objWorld = new THREE.Vector3();
    target.getWorldPosition(objWorld);

    const ctrlQuat = new THREE.Quaternion();
    controller.getWorldQuaternion(ctrlQuat);

    activeGrab = {
      controller,
      object: target,
      offset: objWorld.clone().sub(_worldPos),
      startCtrlQuat: ctrlQuat,
      startObjQuat: target.quaternion.clone(),
    };
  }
}

function onGripEnd(controller) {
  if (activeGrab?.controller === controller) activeGrab = null;
}

// ============================================================
// Gamepad polling — thumbstick locomotion, scroll, buttons
//
// xr-standard gamepad per XRInputSource (Quest Touch):
//   axes[0,1] = unmapped (always 0)
//   axes[2]   = thumbstick X,  axes[3] = thumbstick Y  (Y: negative=forward)
//   buttons[0] = trigger
//   buttons[1] = squeeze/grip
//   buttons[3] = thumbstick press
//   buttons[4] = X (left) / A (right)
//   buttons[5] = Y (left) / B (right)
// ============================================================

function buttonEdge(idx, btnIndex, pressed) {
  const key = btnIndex;
  const prev = !!prevButtons[idx][key];
  prevButtons[idx][key] = pressed;
  return pressed && !prev;
}

function pollGamepads(dt) {
  const session = State.renderer.xr.getSession();
  if (!session) return;

  const panelEl = document.getElementById('panel');
  let leftStickX = 0, leftStickY = 0;
  let rightStickX = 0, rightStickY = 0;
  let leftController = controllers[0];

  for (let si = 0; si < session.inputSources.length; si++) {
    const source = session.inputSources[si];
    const gp = source.gamepad;
    if (!gp) continue;
    const isLeft = source.handedness === 'left';
    // Map input source to its Three.js controller by array index
    const controller = controllers[si];
    if (!controller) continue;
    const idx = si;

    // xr-standard: thumbstick is axes[2,3] (axes[0,1] are unmapped on Quest)
    const stickX = gp.axes[2] || 0;
    const stickY = gp.axes[3] || 0;

    if (isLeft) {
      leftStickX = stickX;
      leftStickY = stickY;
      leftController = controller;
    } else {
      rightStickX = stickX;
      rightStickY = stickY;
    }

    // B / Y button (index 5) — toggle and reposition VR panel
    if (gp.buttons[5] && buttonEdge(idx, 5, gp.buttons[5].pressed)) {
      toggleVRPanel();
    }

    // Thumbstick press (index 3) — toggle passthrough
    if (gp.buttons[3] && buttonEdge(idx, 3, gp.buttons[3].pressed)) {
      togglePassthrough();
    }
  }

  // Left thumbstick up = teleport arc
  if (leftStickY < -STICK_DEADZONE) {
    teleportActive = true;
    teleportController = leftController;
  } else if (teleportActive && leftStickY >= -STICK_DEADZONE) {
    // Release — execute teleport
    if (teleportMarker.visible) {
      const rig = State.vrRig;
      State.camera.getWorldPosition(_worldPos);
      rig.position.x += teleportMarker.position.x - _worldPos.x;
      rig.position.z += teleportMarker.position.z - _worldPos.z;
      saveVRAnchorState();
    }
    teleportActive = false;
    teleportMarker.visible = false;
    teleportArc.visible = false;
  }

  // Snap turn: left stick L/R or right stick L/R (when not scrolling)
  if (snapTurnCooldown > 0) {
    snapTurnCooldown -= dt;
  } else {
    let turnInput = 0;
    if (Math.abs(leftStickX) > 0.6) turnInput = leftStickX;
    else if (Math.abs(rightStickX) > 0.6 && !isPointingAtPanelAny()) turnInput = rightStickX;

    if (Math.abs(turnInput) > 0.6) {
      const angle = turnInput > 0 ? -SNAP_ANGLE : SNAP_ANGLE;
      State.vrRig.rotateY(angle);
      snapTurnCooldown = 0.3;
      saveVRAnchorState();
    }
  }

  // Right thumbstick Y = scroll panel (when pointing at it)
  if (panelEl && panelVisible && Math.abs(rightStickY) > STICK_DEADZONE) {
    if (isPointingAtPanelAny()) {
      panelEl.scrollTop += rightStickY * SCROLL_SPEED * dt;
    }
  }
}

function isPointingAtPanelAny() {
  for (const ctrl of controllers) {
    if (isPointingAtPanel(ctrl)) return true;
  }
  return false;
}

// ============================================================
// Anchor-based drift correction
// ============================================================

async function createSceneAnchor(frame) {
  needsAnchor = false;
  const refSpace = State.renderer.xr.getReferenceSpace();
  if (!refSpace) return;

  // Try restoring a persistent anchor from a previous session
  if (savedVRAnchorData?.anchorUUID) {
    const session = State.renderer.xr.getSession();
    if (session.restorePersistentAnchor) {
      try {
        sceneAnchor = await session.restorePersistentAnchor(savedVRAnchorData.anchorUUID);
        persistentHandle = savedVRAnchorData.anchorUUID;
        rigRepositioned = false;
        lastAnchorPos = null;
        lastAnchorQuat = null;
        console.log('[VR] Restored persistent anchor:', persistentHandle);
        return;
      } catch (e) {
        console.warn('[VR] Persistent anchor restore failed, creating new:', e);
        savedVRAnchorData = null;
      }
    } else {
      savedVRAnchorData = null;
    }
  }

  // Create a new anchor at the rig position
  if (!frame.createAnchor) return;
  const rig = State.vrRig;
  const pose = new XRRigidTransform(
    { x: rig.position.x, y: rig.position.y, z: rig.position.z, w: 1 },
    { x: rig.quaternion.x, y: rig.quaternion.y, z: rig.quaternion.z, w: rig.quaternion.w }
  );
  try {
    sceneAnchor = await frame.createAnchor(pose, refSpace);
    lastAnchorPos = null;
    lastAnchorQuat = null;
    rigRepositioned = true;

    // Request a persistent handle so this anchor survives across sessions
    if (sceneAnchor.requestPersistentHandle) {
      try {
        persistentHandle = await sceneAnchor.requestPersistentHandle();
        console.log('[VR] Created persistent anchor:', persistentHandle);
        saveVRAnchorState();
      } catch (e) {
        console.warn('[VR] Persistent handle unavailable:', e);
      }
    }
  } catch (e) {
    console.warn('[VR] Anchor creation failed:', e);
  }
}

function applyAnchorDriftCorrection(frame) {
  if (!sceneAnchor) return;
  if (!frame.trackedAnchors || !frame.trackedAnchors.has(sceneAnchor)) return;

  const refSpace = State.renderer.xr.getReferenceSpace();
  const anchorPose = frame.getPose(sceneAnchor.anchorSpace, refSpace);
  if (!anchorPose) return;

  const ap = anchorPose.transform.position;
  const ao = anchorPose.transform.orientation;
  const curPos = new THREE.Vector3(ap.x, ap.y, ap.z);
  const curQuat = new THREE.Quaternion(ao.x, ao.y, ao.z, ao.w);

  // Reposition rig on first tracked frame after restoring a persistent anchor
  if (!rigRepositioned && savedVRAnchorData) {
    const off = savedVRAnchorData.rigOffset;
    const qOff = savedVRAnchorData.rigQuatOffset;
    // Offset was saved in anchor-local space; rotate back to world
    const worldOffset = new THREE.Vector3(off.x, off.y, off.z).applyQuaternion(curQuat);
    State.vrRig.position.copy(curPos).add(worldOffset);
    const offsetQuat = new THREE.Quaternion(qOff.x, qOff.y, qOff.z, qOff.w);
    State.vrRig.quaternion.copy(offsetQuat.multiply(curQuat));
    rigRepositioned = true;
    savedVRAnchorData = null;
    console.log('[VR] Rig repositioned from persistent anchor');
  }

  // Ongoing drift correction
  if (lastAnchorPos) {
    const drift = curPos.clone().sub(lastAnchorPos);
    if (drift.lengthSq() > 1e-8) {
      State.vrRig.position.add(drift);
    }
    const rotDrift = curQuat.clone().multiply(lastAnchorQuat.clone().invert());
    if (Math.abs(rotDrift.w) < 0.99999) {
      State.vrRig.quaternion.premultiply(rotDrift);
    }
  }

  lastAnchorPos = curPos;
  lastAnchorQuat = curQuat;
}

// ============================================================
// Persistent anchor save — stores rig-to-anchor offset in IndexedDB
// ============================================================

function saveVRAnchorState() {
  if (!persistentHandle || !lastAnchorPos || !lastAnchorQuat) return;
  const rig = State.vrRig;
  // Store offset in anchor-local space so it survives reference-space rotation
  const worldOffset = rig.position.clone().sub(lastAnchorPos);
  const localOffset = worldOffset.applyQuaternion(lastAnchorQuat.clone().invert());
  const quatOffset = rig.quaternion.clone().multiply(lastAnchorQuat.clone().invert());
  const data = {
    anchorUUID: persistentHandle,
    rigOffset: { x: localOffset.x, y: localOffset.y, z: localOffset.z },
    rigQuatOffset: { x: quatOffset.x, y: quatOffset.y, z: quatOffset.z, w: quatOffset.w },
  };
  preloadedAnchorData = data;
  dbSaveVRAnchor(data).catch(e => console.warn('[VR] Anchor save failed:', e));
}

// ============================================================
// updateVR — called each frame from animate()
// ============================================================

let lastFrameTime = 0;

export function updateVR(frame) {
  if (!State.vrActive) return;

  if (frame) {
    if (needsAnchor && !sceneAnchor) createSceneAnchor(frame);
    applyAnchorDriftCorrection(frame);
  }

  const now = performance.now();
  const dt = lastFrameTime ? Math.min((now - lastFrameTime) / 1000, 0.1) : 0.016;
  lastFrameTime = now;

  // Teleport arc preview (left thumbstick up)
  if (teleportActive && teleportController) {
    updateTeleportArc(teleportController);
  }

  // Grab — move and rotate grabbed object with controller
  if (activeGrab) {
    activeGrab.controller.getWorldPosition(_worldPos);
    const newWorld = _worldPos.clone().add(activeGrab.offset);
    if (activeGrab.object.parent) activeGrab.object.parent.worldToLocal(newWorld);
    activeGrab.object.position.copy(newWorld);

    if (activeGrab.startCtrlQuat) {
      const currentCtrlQuat = new THREE.Quaternion();
      activeGrab.controller.getWorldQuaternion(currentCtrlQuat);
      const deltaQuat = currentCtrlQuat.multiply(activeGrab.startCtrlQuat.clone().invert());
      const newQuat = deltaQuat.multiply(activeGrab.startObjQuat);
      activeGrab.object.quaternion.copy(newQuat);
    }
  }

  // Gamepad: thumbstick locomotion, scroll, face buttons
  pollGamepads(dt);

  // Periodic panel texture refresh
  if (vrPanelMesh && panelVisible) {
    if (now - lastPanelRefresh > PANEL_REFRESH_INTERVAL) {
      lastPanelRefresh = now;
      const tex = vrPanelMesh.material.map;
      if (tex && typeof tex.update === 'function') {
        tex.update();
      }
    }
  }

  // Periodic persistent anchor save
  if (persistentHandle && now - lastAnchorSave > VR_ANCHOR_SAVE_INTERVAL) {
    lastAnchorSave = now;
    saveVRAnchorState();
  }
}
