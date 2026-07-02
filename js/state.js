// ============================================================
// js/state.js — all shared mutable globals
// ============================================================
// All values start as undefined/null and are initialised by
// scene.js (for Three.js objects) or by direct assignment from
// other modules via the setter helpers below.

// Three.js core
export let scene         = null;
export let camera        = null;
export let renderer      = null;
export let labelRenderer = null;

// Cameras
export let orthoCamera  = null;
export let activeCamera = null;
export let orthoOn      = false;

// Splat foreground clip (0 = off, 1 = clip everything up to ~2x orbit
// distance). Applied in both perspective and orthographic views.
export let splatClipFraction = 0;

// Controls
export let orbitControls        = null;
export let stlTransformControls = null;

// UI flags
export let labelsOn = false;

// Floor
export let floorSize = 2;

// Imported objects (meshes, point clouds, splats)
export const importedSTLs = [];
export let stlColorIdx    = 0;

// Object selection
export let selectedSTL      = null;
export let selectedListItem = null;
export let lockAspect       = false;
export let stlSelectable    = true;

// ============================================================
// Setters — used when a let-export must be reassigned from
// another module (ES modules cannot reassign a foreign binding)
// ============================================================

export function initCoreObjects(s, c, r, lr) {
  scene = s; camera = c; renderer = r; labelRenderer = lr;
}

export function initCameras(ortho, active) {
  orthoCamera = ortho; activeCamera = active;
}

export function initControls(orbit, stlTransform) {
  orbitControls        = orbit;
  stlTransformControls = stlTransform;
}

export function setActiveCamera(cam)  { activeCamera = cam; }
export function setOrthoOn(v)         { orthoOn = v; }
export function setSplatClipFraction(v) { splatClipFraction = v; }

export function setLabelsOn(v)          { labelsOn = v; }

export function setFloorSize(v)         { floorSize = v; }

export function setStlColorIdx(v)       { stlColorIdx = v; }
export function setSelectedSTL(e)       { selectedSTL = e; }
export function setSelectedListItem(i)  { selectedListItem = i; }
export function setLockAspect(v)        { lockAspect = v; }
export function setStlSelectable(v)     { stlSelectable = v; }

// VR
export let vrActive       = false;
export let vrRig          = null;
export let passthroughOn  = false;

export function setVRActive(v)       { vrActive = v; }
export function setVRRig(rig)        { vrRig = rig; }
export function setPassthroughOn(v)  { passthroughOn = v; }

// ============================================================
// On-demand rendering
// ------------------------------------------------------------
// The animation loop only draws a frame when something has
// changed. `requestRender()` flags that the next loop iteration
// must render. For things that animate continuously (e.g. a
// Gaussian-splat viewer that re-sorts every frame) register a
// key with setContinuousRender(key, true) to keep drawing until
// it is released. VR always renders (driven by WebXR).
// ============================================================
export let needsRender = true;
export function requestRender() { needsRender = true; }
export function clearNeedsRender() { needsRender = false; }

const _continuousKeys = new Set();
export function setContinuousRender(key, on) {
  if (on) _continuousKeys.add(key); else _continuousKeys.delete(key);
  if (on) needsRender = true;
}
export function shouldRenderContinuously() { return _continuousKeys.size > 0; }
