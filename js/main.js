// ============================================================
// js/main.js — animate loop, event handlers, initialisation
// ============================================================
import * as THREE from 'three';
import { acceleratedRaycast } from 'three-mesh-bvh';

// Patch Three.js Mesh to use BVH-accelerated raycasting
THREE.Mesh.prototype.raycast = acceleratedRaycast;

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
  buildSceneMetadataForDB, buildSceneBuffersForDB, sceneBufferSignature,
  exportSceneState, importSceneState, restoreSTLsFromState,
  loadSTLFile, loadOBJFile, loadPLYFile, loadGLBFile, loadSplatFile,
  updateSplatClip,
  setPointSize, setPointShape,
  addPrimitive,
  selectSTL, deselectSTL, setSTLTransformMode, syncSTLNumericInputs,
} from './stl.js';
import { dbSave, BUFFERS_KEY } from './storage.js';
import { initVR, updateVR } from './vr.js';
import { initLidarPanel } from './lidar.js';

try { initLidarPanel(); } catch (e) { console.error('LiDAR panel init failed:', e); }

// ============================================================
// Raycaster (for click-to-select)
// ============================================================
const raycaster = new THREE.Raycaster();
raycaster.params.Points = { threshold: 0.005 };
const mouse = new THREE.Vector2();

// ============================================================
// Animate loop
// ============================================================
function animate(time, frame) {
  updateVR(frame);

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

  // Foreground splat clip tracks camera distance, so refresh it on each
  // drawn frame before rendering.
  updateSplatClip();

  State.renderer.render(State.scene, State.activeCamera);

  if (!State.vrActive) {
    State.labelRenderer.render(State.scene, State.activeCamera);
    renderNavGizmo();
  }
}

// ============================================================
// Button event handlers
// ============================================================

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

// Mesh import button
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

// Object transform mode buttons
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

// Object numeric position/rotation/scale inputs
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

// Point size slider (mm in the UI; PointsMaterial.size is world metres)
document.getElementById('pointSize').addEventListener('input', (e) => {
  const mm = parseFloat(e.target.value);
  document.getElementById('pointSizeVal').textContent = `${mm} mm`;
  setPointSize(mm / 1000);
});

// Point shape selector (round / square / soft)
document.getElementById('pointShape').addEventListener('change', (e) => {
  setPointShape(e.target.value);
});

// ============================================================
// Click-to-select imported meshes
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
  if (State.stlTransformControls.dragging) return;
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

  // Test against imported meshes (if selection enabled). Point clouds and
  // splats are excluded — picking them would raycast every point on the main
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

  // Clicked away from any object — deselect (unless the gizmo itself was clicked).
  if (State.selectedSTL) {
    const gizmoHits = raycaster.intersectObjects(State.stlTransformControls.children, true);
    if (gizmoHits.length === 0) deselectSTL();
  }
});

// Keyboard shortcuts for object transform
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  if (State.selectedSTL) {
    if (e.key === 't' || e.key === 'T') {
      setSTLTransformMode('translate');
    } else if (e.key === 'r' || e.key === 'R') {
      setSTLTransformMode('rotate');
    } else if (e.key === 's' || e.key === 'S') {
      setSTLTransformMode('scale');
    } else if (e.key === 'Escape') {
      deselectSTL();
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
// render whenever the user does something. Camera moves, snap
// animation and async loads request renders at their own sources.
// With nothing happening, the loop draws nothing.
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
// LidarStudio starts as an empty editor scene; clouds/splats are loaded
// explicitly (Library / Load Scene / Load Workflow).

document.getElementById('loading').style.display = 'none';
State.requestRender();

// Auto-save scene to IndexedDB periodically and on page unload.
// IndexedDB handles large mesh buffers that would overflow localStorage's ~5 MB quota.
//
// The heavy mesh/point-cloud/splat buffers are stored separately and only
// re-written when they actually change (tracked by a cheap signature). Routine
// saves — fired every 30 s and while the camera moves — then write only
// the small metadata record, avoiding the multi-MB IndexedDB structured clone
// that previously hitched the frame (and stalled rotation) on every tick.
let _lastSavedBufferSig = null;

async function autoSaveScene() {
  if (State.importedSTLs.length === 0) return;
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

function _clearImportedObjects() {
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
}

document.getElementById('clearSceneBtn').addEventListener('click', () => {
  if (!confirm('Clear the entire scene? This cannot be undone.')) return;
  _clearImportedObjects();
  State.requestRender();
});

// ============================================================
// Core scene restore (used by file load / drag-drop)
// ============================================================
async function restoreScene(data) {
  _clearImportedObjects();

  // Restore floor size
  if (data.floorSize != null) {
    setFloorSize(data.floorSize);
    document.getElementById('floorSize').value = data.floorSize;
    document.getElementById('floorSizeVal').textContent = `${data.floorSize} m`;
  }

  // Restore point size
  if (data.pointSize != null) {
    setPointSize(data.pointSize);
    const mm = Math.round(data.pointSize * 10000) / 10;
    document.getElementById('pointSize').value = mm;
    document.getElementById('pointSizeVal').textContent = `${mm} mm`;
  }

  // Restore point shape
  if (data.pointShape) {
    setPointShape(data.pointShape);
    document.getElementById('pointShape').value = data.pointShape;
  }

  // Restore camera
  if (data.camera) {
    if (data.camera.position) State.camera.position.set(...data.camera.position);
    if (data.camera.target) State.orbitControls.target.set(...data.camera.target);
    State.camera.updateProjectionMatrix();
    State.orbitControls.update();
  }

  // Ensure full scene graph is updated before restoring objects
  State.scene.updateMatrixWorld(true);

  // Restore objects (two-phase: create, then apply transforms)
  if (data.stls && data.stls.length > 0) {
    await restoreSTLsFromState(data.stls);
  }
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
