// js/lidar.js — LiDAR workflow panel for LidarStudio.
// Drives the /api/* endpoints (cloud/splat generation + project outputs) added
// by the Python backend (lidar_jobs.py) and loads results straight into the
// three.js + GaussianSplats3D viewer via the existing PLY loader.

import { loadPLYFile, deselectSTL, selectSTL, setVisibilityClip } from './stl.js';
import * as State from './state.js';
import * as THREE from 'three';
import { TransformControls } from 'three/addons/controls/TransformControls.js';

const DEFAULT_PROJECT = '/home/gareth';

// Maps an in-scene object name -> its server-side .ply path, so edits can be
// run on the full file in Python (objects loaded from disk only).
const objPaths = {};

// Reused to detect an untransformed object (nothing to bake).
const _IDENTITY = new THREE.Matrix4();

// ── Saved workflow state (localStorage) ──
// The panel's form inputs (project/scan paths, generate + edit parameters,
// option toggles) are saved/restored on demand via the Save/Load Workflow
// buttons — not auto-restored.
const LS_KEY = 'lidarStudio.workflow';
function loadState() {
  try { return JSON.parse(localStorage.getItem(LS_KEY)) || {}; } catch { return {}; }
}
function saveWorkflowState(state) {
  try { localStorage.setItem(LS_KEY, JSON.stringify(state)); } catch { /* quota/private mode */ }
}

async function api(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

// Fetch a server-side .ply by absolute path and hand it to the viewer's PLY
// loader (which auto-detects splat vs point cloud vs mesh).
async function loadPlyFromServer(path, name, transforms = null) {
  const url = `/api/scan/file?path=${encodeURIComponent(path)}`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`load failed: ${resp.status}`);
  // Straight to ArrayBuffer: wrapping the response in a Blob + File and
  // re-reading it through FileReader held three full copies of a multi-GB
  // splat in tab memory, which is what made big library loads fail as 'err'.
  const buffer = await resp.arrayBuffer();
  const fname = name || path.split('/').pop();
  objPaths[fname.replace(/\.ply$/i, '')] = path;   // entry.name is the basename
  // Library loads pass no pose, so restore it from the edit's sidecar (edits
  // keep the file in its local frame; the pose is saved alongside).
  if (!transforms) transforms = await fetchPose(path);
  const entry = await loadPLYFile(buffer, transforms, fname);
  State.requestRender && State.requestRender();
  return entry;
}

// Fetch the <ply>.pose.json sidecar written by an edit, if present.
async function fetchPose(path) {
  try {
    const r = await fetch(`/api/scan/file?path=${encodeURIComponent(path + '.pose.json')}`);
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}

// Re-select an entry in the object list (keeps it the active selection after an
// edit, so follow-up edits and box placement keep working).
function reselect(entry) {
  if (!entry) return;
  const idx = State.importedSTLs.indexOf(entry);
  const li = document.querySelectorAll('.stl-item')[idx] || null;
  selectSTL(entry, li);
}

// Snapshot a mesh's local pose so an edited result can be reloaded in the same
// place (edits keep the file's local coords, so they'd otherwise spring back to
// the untransformed origin).
function poseOf(mesh, entry) {
  if (!mesh) return null;
  return {
    position: mesh.position.toArray(),
    rotation: [mesh.rotation.x, mesh.rotation.y, mesh.rotation.z],
    scale: mesh.scale.toArray(),
    visible: mesh.visible,
    parentLink: entry?.parentLink ?? null,
  };
}

// World matrix that maps the object's *file* coordinates to world — the same
// frame the visibility clip tests against. For splats that's the inner splat
// mesh (it carries the viewer's internal transform); for clouds it's the mesh
// itself. Box crop/delete must use this so the region matches what's shown.
function cloudWorldMatrix(entry) {
  const m = (entry && entry.isSplat && entry._splatViewer && entry._splatViewer.splatMesh)
    ? entry._splatViewer.splatMesh : entry.mesh;
  m.updateWorldMatrix(true, false);
  return m.matrixWorld;
}

// Describe an eraser primitive for the server: {type, matrix} where matrix maps
// a cloud-file point into the primitive's canonical unit frame (cube/sphere
// half-extent or radius 0.5). cloudWorld maps the cloud's file coords to world.
function eraserMatrix(prim, cloudWorld) {
  prim.mesh.updateWorldMatrix(true, false);
  prim.mesh.geometry.computeBoundingBox();
  const bb = prim.mesh.geometry.boundingBox;
  const half = Math.max(bb.max.x - bb.min.x, bb.max.y - bb.min.y, bb.max.z - bb.min.z) / 2 || 0.025;
  const M = new THREE.Matrix4().makeScale(0.5 / half, 0.5 / half, 0.5 / half)
    .multiply(new THREE.Matrix4().copy(prim.mesh.matrixWorld).invert())
    .multiply(cloudWorld);
  return { type: prim.primType, matrix: Array.from(M.elements) };
}

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'style') n.style.cssText = v;
    else if (k === 'class') n.className = v;
    else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const kid of kids) n.append(kid?.nodeType ? kid : document.createTextNode(kid ?? ''));
  return n;
}

export function initLidarPanel() {
  if (document.getElementById('lidar-panel')) return;

  const css = `
  #lidar-head{cursor:pointer;user-select:none;display:flex;justify-content:space-between;
    align-items:center;font:700 12px system-ui;letter-spacing:.04em;text-transform:uppercase;
    color:#9cf;padding:6px 0}
  #lidar-panel{color:#cdd6e3;font:13px system-ui;padding-bottom:10px;margin-bottom:8px;
    border-bottom:1px solid #2a3344}
  #lidar-panel h4{margin:10px 0 6px;font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#8aa}
  #lidar-panel input,#lidar-panel select{width:100%;box-sizing:border-box;background:#0f1620;color:#dfe;
    border:1px solid #324a5e;border-radius:5px;padding:6px;margin:3px 0;font:12px monospace}
  #lidar-panel button.act{width:100%;background:#2a6df0;color:#fff;border:0;border-radius:6px;
    padding:8px;margin-top:6px;font-weight:600;cursor:pointer}
  #lidar-panel button.act:disabled{opacity:.5;cursor:default}
  #lidar-panel .row{display:flex;gap:6px}
  #lidar-panel .item{display:flex;justify-content:space-between;align-items:center;gap:6px;
    padding:5px 6px;border:1px solid #283447;border-radius:5px;margin:3px 0;background:#121a26}
  #lidar-panel .item>span{flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  #lidar-panel .item button{flex:0 0 auto;width:auto;margin:0;background:#2c7;border:0;color:#04210f;border-radius:4px;padding:3px 9px;
    font-weight:700;cursor:pointer;font-size:11px}
  #lidar-panel .muted{color:#7d8aa0;font-size:11px}
  #ls-bar-label{display:none;justify-content:space-between;align-items:baseline;gap:8px;
    margin-top:8px;font-size:11px;color:#aeb9cc}
  #ls-bar-label .ls-bar-msg{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  #ls-bar-label .ls-bar-pct{font-weight:700;color:#cdd8ec;font-variant-numeric:tabular-nums;flex:0 0 auto}
  #ls-bar-wrap{height:10px;background:#0f1620;border-radius:5px;overflow:hidden;margin-top:4px;display:none}
  #ls-bar{height:100%;width:0;background:linear-gradient(90deg,#2a6df0,#4f9bff);
    border-radius:5px;transition:width .25s ease}
  #ls-bar.indeterminate{width:35%!important;animation:ls-indet 1.1s ease-in-out infinite}
  @keyframes ls-indet{0%{margin-left:-35%}100%{margin-left:100%}}
  #ls-log{font:10px/1.4 monospace;color:#9fb;background:#0c121b;border-radius:5px;padding:6px;
    margin-top:6px;max-height:120px;overflow:auto;white-space:pre-wrap;display:none}`;
  document.head.append(el('style', {}, css));

  // ── Library (existing outputs) ──
  const projectInput = el('input', { id: 'ls-project', value: DEFAULT_PROJECT });
  const outList = el('div', { id: 'ls-outputs' });
  const refresh = async () => {
    outList.replaceChildren(el('div', { class: 'muted' }, 'Loading…'));
    try {
      const data = await api('/api/project/outputs', { path: projectInput.value.trim() });
      const items = [
        ...(data.splats || []).map(f => ({ ...f, kind: 'splat' })),
        ...(data.pointclouds || []).map(f => ({ ...f, kind: 'cloud' })),
      ];
      // Offer every project cloud as a splat seed, keeping the current choice
      // across refreshes when it still exists.
      const seedCur = seedSel.value;
      seedSel.replaceChildren(
        el('option', { value: '' }, 'seed cloud: auto'),
        ...(data.pointclouds || []).map(f => el('option', { value: f.path }, `seed: ${f.name}`)));
      seedSel.value = [...seedSel.options].some(o => o.value === seedCur) ? seedCur : '';
      outList.replaceChildren(...(items.length ? items.map(f =>
        el('div', { class: 'item' },
          el('span', { title: f.name }, `${f.kind === 'splat' ? '🟣' : '⚪'} ${f.name} `,
            el('span', { class: 'muted' }, `${f.size_mb}MB`)),
          el('button', { onclick: async (e) => {
            e.target.disabled = true; e.target.textContent = '…';
            try { await loadPlyFromServer(f.path, f.name); e.target.textContent = '✓'; }
            catch (err) { e.target.textContent = 'err'; console.error(err); }
          } }, 'Load'))
      ) : [el('div', { class: 'muted' }, 'No outputs in this project.')]));
    } catch (err) {
      outList.replaceChildren(el('div', { class: 'muted' }, 'Error: ' + err.message));
    }
  };

  // ── Generate ──
  const scanInput = el('input', { id: 'ls-scan', placeholder: '/path/to/scan_folder' });
  const typeSel = el('select', {}, ...['splat', 'pointcloud'].map(t => el('option', { value: t }, t)));
  const methodSel = el('select', {}, ...['surfel', 'trained', 'bootstrap'].map(m => el('option', { value: m }, m)));
  const voxelInput = el('input', { type: 'number', step: '0.005', min: '0.005', value: '0.01' });
  const sorInput = el('input', { type: 'number', step: '0.25', min: '1', value: '2' });
  // Seed cloud for splat generation: 'auto' keeps the job's default (surfel:
  // the scan's dense cloud; trained: the latest project cloud); any other
  // entry pins the splat to that exact cloud — e.g. an edited one.
  const seedSel = el('select', {}, el('option', { value: '' }, 'seed cloud: auto'));
  const pSeed = el('div', { class: 'row' }, seedSel);
  // Bootstrap splat Gaussian radius (m) — smaller = finer splats, less blobby.
  const splatSizeInput = el('input', { type: 'number', step: '0.005', min: '0.001', value: '0.02' });
  const pSplatSize = el('div', { class: 'row' },
    el('label', { class: 'muted', style: 'flex:1' }, 'blob size (m)', splatSizeInput));
  // Blob size only applies to the bootstrap splat builder.
  const showGen = () => {
    pSplatSize.style.display = (typeSel.value === 'splat' && methodSel.value === 'bootstrap') ? 'flex' : 'none';
    pSeed.style.display = typeSel.value === 'splat' ? 'flex' : 'none';
  };
  typeSel.addEventListener('change', showGen);
  methodSel.addEventListener('change', showGen);
  showGen();
  const barMsg = el('span', { class: 'ls-bar-msg' }, 'Starting…');
  const barPct = el('span', { class: 'ls-bar-pct' }, '');
  const barLabel = el('div', { id: 'ls-bar-label' }, barMsg, barPct);
  const barFill = el('div', { id: 'ls-bar' });
  const barWrap = el('div', { id: 'ls-bar-wrap' }, barFill);
  const logBox = el('div', { id: 'ls-log' });
  const genBtn = el('button', { class: 'act' }, 'Generate');

  let lastPct = null;
  // Drive the progress bar. pct === undefined ⇒ keep current % (log-only update);
  // before the first real % arrives the bar runs an indeterminate animation.
  const setBar = (pct, msg) => {
    barLabel.style.display = 'flex';
    barWrap.style.display = 'block';
    if (typeof pct === 'number') {
      lastPct = pct;
      barFill.classList.remove('indeterminate');
      barFill.style.width = `${pct}%`;
      barPct.textContent = `${Math.round(pct)}%`;
    } else if (lastPct === null) {           // unknown progress → indeterminate
      barFill.classList.add('indeterminate');
      barPct.textContent = '';
    }
    if (msg) {
      barMsg.textContent = msg;
      logBox.style.display = 'block';
      logBox.textContent = msg + '\n' + logBox.textContent;
    }
  };
  // Reset the bar to a clean indeterminate state at the start of a new job.
  const resetBar = () => {
    lastPct = null;
    barFill.style.width = '0%'; barFill.classList.remove('indeterminate');
    barPct.textContent = ''; barMsg.textContent = 'Starting…';
  };

  genBtn.onclick = async () => {
    const scan = scanInput.value.trim();
    if (!scan) { setBar(0, 'Enter a scan folder path'); return; }
    genBtn.disabled = true; logBox.textContent = ''; logBox.style.display = 'block';
    resetBar();
    const type = typeSel.value;
    const options = type === 'splat'
      ? { splat_mode: methodSel.value, splat_voxel: parseFloat(voxelInput.value),
          surfel_sor: parseFloat(sorInput.value), splat_size: parseFloat(splatSizeInput.value),
          ...(seedSel.value ? { pointcloud: seedSel.value } : {}) }
      : { voxel_size: parseFloat(voxelInput.value) };
    try {
      const { job_id } = await api('/api/process/start',
        { type, project_path: projectInput.value.trim(), scan_path: scan, options });
      const es = new EventSource(`/api/process/events/${job_id}`);
      es.addEventListener('done', () => { es.close(); genBtn.disabled = false; refresh(); });
      es.onmessage = async (ev) => {
        const d = JSON.parse(ev.data);
        if (d.event === 'progress') setBar(d.percent, d.message);
        else if (d.event === 'log') setBar(undefined, d.message);
        else if (d.event === 'error') { setBar(0, 'ERROR: ' + d.message); es.close(); genBtn.disabled = false; }
        else if (d.event === 'result') {
          setBar(100, `Done: ${d.filename} — loading…`);
          try { await loadPlyFromServer(d.path, d.filename); } catch (e) { console.error(e); }
        }
      };
      es.onerror = () => { es.close(); genBtn.disabled = false; };
    } catch (err) { setBar(0, 'ERROR: ' + err.message); genBtn.disabled = false; }
  };

  // ── Edit (operates on the selected cloud/splat) ──
  const editOp = el('select', {},
    el('option', { value: 'decimate' }, 'Decimate (keep 1-in-N)'),
    el('option', { value: 'denoise_sor' }, 'Denoise (remove outliers)'),
    el('option', { value: 'crop' }, 'Crop to box'),
    el('option', { value: 'recolour' }, 'Recolour from scan photos'),
    el('option', { value: 'transform' }, 'Save transformed (bake pose)'));
  const facInput = el('input', { type: 'number', min: '2', step: '1', value: '2' });
  const sorNb = el('input', { type: 'number', min: '4', step: '1', value: '20' });
  const sorStd = el('input', { type: 'number', min: '0.5', step: '0.25', value: '2' });
  const cMin = ['x', 'y', 'z'].map(() => el('input', { type: 'number', step: '0.1' }));
  const cMax = ['x', 'y', 'z'].map(() => el('input', { type: 'number', step: '0.1' }));
  const invCb = el('input', { type: 'checkbox', style: 'width:auto;flex:0 0 auto;margin:0' });
  const editStat = el('div', { class: 'muted' }, 'Select a loaded object to edit.');

  // ── 3D crop-box gizmo (dedicated TransformControls, isolated from selection) ──
  let cropGizmo = null, cropBox = null, cropTargetPath = null, cropTargetLi = null, cropTargetMesh = null;

  // Which box (visibility or crop) the T/R/S keys currently drive. Switched by
  // clicking a box or grabbing its gizmo; the active box's edges show in full
  // colour, the other's are dimmed.
  let activeBox = null;
  function setActiveBox(box) {
    activeBox = box;
    for (const b of [visBox, cropBox]) {
      const ls = b && b.children && b.children[0];
      if (ls && ls.material) ls.material.color.set(b === activeBox ? (b.userData.edgeColor ?? 0xffffff) : 0x52627a);
    }
    State.requestRender();
  }

  function ensureGizmo() {
    if (cropGizmo) return cropGizmo;
    const g = new TransformControls(State.activeCamera || State.camera, State.renderer.domElement);
    g.setSize(0.7);
    g.addEventListener('dragging-changed', (e) => { State.orbitControls.enabled = !e.value; if (e.value) setActiveBox(cropBox); });
    g.addEventListener('change', () => State.requestRender());
    State.scene.add(g);
    cropGizmo = g;
    return g;
  }
  function removeCropBox(render = true) {
    const wasActive = activeBox === cropBox;
    if (cropGizmo && cropGizmo.object) cropGizmo.detach();
    if (cropBox) {
      State.scene.remove(cropBox);
      cropBox.traverse(o => { o.geometry?.dispose?.(); o.material?.dispose?.(); });
      cropBox = null;
    }
    if (wasActive) setActiveBox(visBox);   // hand focus to the visibility box if present
    if (render) State.requestRender();
  }
  function placeCropBox() {
    const e = State.selectedSTL;
    if (!e) { editStat.textContent = 'Select a loaded object first.'; return; }
    if (!objPaths[e.name]) { editStat.textContent = 'Crop needs a file: load from Library / generate.'; return; }
    const b = new THREE.Box3().setFromObject(e.mesh);
    if (b.isEmpty()) { editStat.textContent = 'Could not read object bounds.'; return; }
    cropTargetPath = objPaths[e.name];
    cropTargetLi = State.selectedListItem;
    cropTargetMesh = e.mesh;             // remember its in-scene transform
    deselectSTL();                       // free the shared gizmo / selection
    removeCropBox(false);
    const c = b.getCenter(new THREE.Vector3()), s = b.getSize(new THREE.Vector3());
    const geo = new THREE.BoxGeometry(1, 1, 1);
    cropBox = new THREE.Mesh(geo, new THREE.MeshBasicMaterial(
      { color: 0x33ff99, transparent: true, opacity: 0.12, depthWrite: false }));
    cropBox.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({ color: 0x33ff99 })));
    cropBox.userData.edgeColor = 0x33ff99;
    cropBox.position.copy(c);
    cropBox.scale.set(Math.max(s.x, 0.05), Math.max(s.y, 0.05), Math.max(s.z, 0.05));
    State.scene.add(cropBox);
    ensureGizmo().attach(cropBox);
    setActiveBox(cropBox);
    State.requestRender();
    editStat.textContent = 'Drag the green box (Move/Rotate/Scale), then Apply edit.';
  }

  // ── Visibility clip box (non-destructive view aid) ──
  // A blue box that hides everything inside or outside it, so internal
  // structure can be inspected/edited. Uses its own gizmo so it never
  // disturbs the selection or the crop box.
  let visGizmo = null, visBox = null, visModeOutside = false;
  // The object the visibility box was placed on. Remembered because clicking the
  // box deselects the cloud (the box has its own gizmo), so the delete can't
  // rely on State.selectedSTL.
  let visTargetEntry = null;
  const visStat = el('div', { class: 'muted' }, 'Reveal internal structure without deleting anything.');
  function ensureVisGizmo() {
    if (visGizmo) return visGizmo;
    const g = new TransformControls(State.activeCamera || State.camera, State.renderer.domElement);
    g.setSize(0.7);
    g.addEventListener('dragging-changed', (e) => { State.orbitControls.enabled = !e.value; if (e.value) setActiveBox(visBox); });
    g.addEventListener('change', () => { pushVisClip(); State.requestRender(); });
    State.scene.add(g);
    visGizmo = g;
    return g;
  }
  function pushVisClip() {
    if (!visBox) return;
    visBox.updateWorldMatrix(true, false);
    setVisibilityClip({ enabled: true, mode: visModeOutside ? 'outside' : 'inside',
      matrix: Array.from(visBox.matrixWorld.elements) });
  }
  function removeVisBox(render = true) {
    const wasActive = activeBox === visBox;
    if (visGizmo && visGizmo.object) visGizmo.detach();
    if (visBox) {
      State.scene.remove(visBox);
      visBox.traverse(o => { o.geometry?.dispose?.(); o.material?.dispose?.(); });
      visBox = null;
    }
    setVisibilityClip({ enabled: false });
    if (wasActive) setActiveBox(cropBox);   // hand focus to the crop box if present
    if (render) State.requestRender();
  }
  function placeVisBox() {
    const e = State.selectedSTL;
    const b = e && e.mesh ? new THREE.Box3().setFromObject(e.mesh) : null;
    if (!b || b.isEmpty()) { visStat.textContent = 'Select a loaded object first.'; return; }
    removeVisBox(false);
    visTargetEntry = e;   // remember the object this box edits
    const c = b.getCenter(new THREE.Vector3()), s = b.getSize(new THREE.Vector3());
    const geo = new THREE.BoxGeometry(1, 1, 1);
    visBox = new THREE.Mesh(geo, new THREE.MeshBasicMaterial(
      { color: 0x33aaff, transparent: true, opacity: 0.10, depthWrite: false }));
    visBox.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({ color: 0x33aaff })));
    visBox.userData.edgeColor = 0x33aaff;
    visBox.position.copy(c);
    // Start at half the object's size so there's something to reveal.
    visBox.scale.set(Math.max(s.x * 0.5, 0.05), Math.max(s.y * 0.5, 0.05), Math.max(s.z * 0.5, 0.05));
    State.scene.add(visBox);
    ensureVisGizmo().attach(visBox);
    setActiveBox(visBox);
    pushVisClip();
    State.requestRender();
    visStat.textContent = 'Drag the blue box; toggle inside/outside.';
  }
  // Keeps the inside/outside toggle and the delete button labelled in sync, so
  // it's always clear which side the delete will remove.
  const updateVisLabels = () => {
    const side = visModeOutside ? 'outside' : 'inside';
    visModeBtn.textContent = `Showing: ${side}`;
    delBtn.textContent = `Delete shown points (${side})`;
  };
  const visModeBtn = el('button', { class: 'act', style: 'flex:1', onclick: () => {
    visModeOutside = !visModeOutside;
    updateVisLabels();
    pushVisClip();
  } }, 'Showing: inside');
  // Delete the currently-shown side of the box (100% removal — what decimate
  // can't do). Showing inside deletes the inside; showing outside deletes the
  // outside.
  async function deleteInBox() {
    if (!visBox) { editStat.textContent = 'Place a visibility box first.'; return; }
    // Use the object the box was placed on (clicking the box clears the live
    // selection), falling back to the current selection.
    const entry = visTargetEntry || State.selectedSTL;
    if (!entry || !State.importedSTLs.includes(entry)) {
      editStat.textContent = 'Re-place the visibility box on an object first.'; return;
    }
    const srcPath = objPaths[entry.name];
    if (!srcPath) { editStat.textContent = 'Delete needs a file: load it from the Library.'; return; }
    const li = document.querySelectorAll('.stl-item')[State.importedSTLs.indexOf(entry)] || null;
    visBox.updateWorldMatrix(true, false);
    // Map file coords → box-local using the same frame the clip uses, so the
    // deleted region matches what's shown. Delete the *shown* side: showing
    // inside → invert=true keeps outside (deletes inside), and vice-versa.
    const Q = new THREE.Matrix4().copy(visBox.matrixWorld).invert().multiply(cloudWorldMatrix(entry));
    const params = { matrix: Array.from(Q.elements), invert: !visModeOutside };
    editStat.textContent = `Deleting points ${visModeOutside ? 'outside' : 'inside'} box…`;
    const newEntry = await runEdit('/api/edit/apply', { path: srcPath, op: 'crop', params }, li,
      r => `Kept ${r.kept.toLocaleString()} / ${r.total.toLocaleString()} — reloading`,
      poseOf(entry.mesh, entry));
    if (newEntry) visTargetEntry = newEntry;   // keep editing the reloaded result
    // The shown side is now empty; flip the view to the kept side so the result
    // is visible, and keep the box in place for further edits.
    visModeOutside = !visModeOutside;
    updateVisLabels();
    pushVisClip();
  }
  const delBtn = el('button', { class: 'act', style: 'margin-top:4px;background:#a33', onclick: deleteInBox },
    'Delete shown points (inside)');
  const pVis = el('div', {},
    visStat,
    el('button', { class: 'act', style: 'margin-top:2px', onclick: placeVisBox }, 'Place visibility box'),
    el('div', { class: 'row' },
      el('button', { class: 'act', style: 'flex:1', onclick: () => visGizmo?.setMode('translate') }, 'Move'),
      el('button', { class: 'act', style: 'flex:1', onclick: () => visGizmo?.setMode('rotate') }, 'Rotate'),
      el('button', { class: 'act', style: 'flex:1', onclick: () => visGizmo?.setMode('scale') }, 'Scale')),
    el('div', { class: 'muted', style: 'font-size:10px' }, 'or press T / R / S'),
    el('div', { class: 'row' },
      visModeBtn,
      el('button', { class: 'act', style: 'flex:1', onclick: () => removeVisBox() }, 'Clear')),
    delBtn);

  const pDec = el('div', { class: 'row' }, el('label', { class: 'muted', style: 'flex:1' }, 'N', facInput));
  const pSor = el('div', { class: 'row' },
    el('label', { class: 'muted', style: 'flex:1' }, 'neighbours', sorNb),
    el('label', { class: 'muted', style: 'flex:1' }, 'std', sorStd));
  const fillBtn = el('button', { class: 'act', style: 'margin-top:2px',
    onclick: () => {
      const e = State.selectedSTL;
      if (!e) { editStat.textContent = 'Select an object first.'; return; }
      const b = new THREE.Box3().setFromObject(e.mesh);
      if (b.isEmpty()) { editStat.textContent = 'Could not read bounds; enter manually.'; return; }
      cMin.forEach((inp, i) => inp.value = b.min.getComponent(i).toFixed(2));
      cMax.forEach((inp, i) => inp.value = b.max.getComponent(i).toFixed(2));
    } }, 'Fill bounds from selection');
  const pCrop = el('div', {},
    el('button', { class: 'act', style: 'margin-top:2px', onclick: placeCropBox }, 'Place 3D crop box'),
    el('div', { class: 'row' },
      el('button', { class: 'act', style: 'flex:1', onclick: () => cropGizmo?.setMode('translate') }, 'Move'),
      el('button', { class: 'act', style: 'flex:1', onclick: () => cropGizmo?.setMode('rotate') }, 'Rotate'),
      el('button', { class: 'act', style: 'flex:1', onclick: () => cropGizmo?.setMode('scale') }, 'Scale')),
    el('div', { class: 'muted', style: 'font-size:10px' }, 'or press T / R / S'),
    el('div', { class: 'muted', style: 'margin-top:4px' }, 'or set bounds manually:'),
    el('div', { class: 'row' }, el('span', { class: 'muted', style: 'width:28px' }, 'min'), ...cMin),
    el('div', { class: 'row' }, el('span', { class: 'muted', style: 'width:28px' }, 'max'), ...cMax),
    el('label', { class: 'muted' }, invCb, ' keep outside (delete inside)'),
    fillBtn);
  const pRecol = el('div', { class: 'muted' },
    'Re-projects the photos from the Scan folder (Generate section) onto the '
    + 'selected cloud using its saved trajectory. Set that field first.');
  const pTransform = el('div', { class: 'muted' },
    'Bakes the selected object\'s current move/rotate/scale into a new file '
    + '(saved as *_edited.ply), so it reopens already placed. Splats also rotate '
    + 'their gaussians; non-uniform scale is approximate.');
  // Optional: restrict decimate/denoise to the visibility box region.
  const regionCb = el('input', { type: 'checkbox', style: 'width:auto;flex:0 0 auto;margin:0' });
  const pRegion = el('label', { class: 'muted', style: 'display:flex;align-items:center;gap:6px;margin-top:4px' },
    regionCb, 'Limit to visibility box region');
  const regionScoped = (op) => op === 'decimate' || op === 'denoise_sor';
  const showOp = () => {
    pDec.style.display = editOp.value === 'decimate' ? 'flex' : 'none';
    pSor.style.display = editOp.value === 'denoise_sor' ? 'flex' : 'none';
    pCrop.style.display = editOp.value === 'crop' ? 'block' : 'none';
    pRecol.style.display = editOp.value === 'recolour' ? 'block' : 'none';
    pTransform.style.display = editOp.value === 'transform' ? 'block' : 'none';
    pRegion.style.display = regionScoped(editOp.value) ? 'flex' : 'none';
  };
  editOp.addEventListener('change', showOp);
  const applyEditBtn = el('button', { class: 'act' }, 'Apply edit');

  async function runEdit(endpoint, body, li, doneMsg, transforms = null) {
    applyEditBtn.disabled = true;
    let entry = null;
    try {
      // Persist the object's pose so a later Library reload restores placement.
      const r = await api(endpoint, { ...body, pose: transforms });
      editStat.textContent = doneMsg(r);
      entry = await loadPlyFromServer(r.output, r.output.split('/').pop(), transforms);
      if (li) li.querySelector('button[title="Remove"]')?.click();  // drop the pre-edit object
      reselect(entry);   // keep the result selected for follow-up edits
    } catch (e) { editStat.textContent = 'Error: ' + e.message; }
    applyEditBtn.disabled = false;
    return entry;
  }

  applyEditBtn.onclick = async () => {
    const op = editOp.value;

    // Crop via the 3D box uses the remembered target (placing the box deselects).
    if (op === 'crop' && cropBox) {
      cropBox.updateWorldMatrix(true, false);
      // The server crops the file's *local* point coords, so fold the cloud's
      // in-scene transform into the matrix: Q = inv(boxWorld) · cloudWorld maps
      // a local point straight into box-local space (test |xyz| <= 0.5 there).
      const cropEntry = State.importedSTLs.find(s => s.mesh === cropTargetMesh);
      const Q = new THREE.Matrix4().copy(cropBox.matrixWorld).invert();
      if (cropEntry) Q.multiply(cloudWorldMatrix(cropEntry));
      const params = { matrix: Array.from(Q.elements), invert: invCb.checked };
      editStat.textContent = 'Cropping to box…';
      const li = cropTargetLi;
      await runEdit('/api/edit/apply', { path: cropTargetPath, op: 'crop', params },
        li, r => `Kept ${r.kept.toLocaleString()} / ${r.total.toLocaleString()} — reloading`,
        poseOf(cropTargetMesh, cropEntry));
      removeCropBox();
      return;
    }

    const entry = State.selectedSTL;
    if (!entry) { editStat.textContent = 'Select an object in the list first.'; return; }
    const srcPath = objPaths[entry.name];
    if (!srcPath) { editStat.textContent = 'Edit needs a file: load it from the Library or generate it.'; return; }

    const li = State.selectedListItem;

    if (op === 'recolour') {
      const scan = scanInput.value.trim();
      if (!scan) { editStat.textContent = 'Set the Scan folder (Generate section) first.'; return; }
      editStat.textContent = 'Recolouring (multi-view)…';
      return runEdit('/api/edit/recolour', { path: srcPath, scan_path: scan }, li,
        r => `Coloured ${r.coloured.toLocaleString()} / ${r.total.toLocaleString()} — reloading`,
        poseOf(entry.mesh, entry));
    }

    if (op === 'transform') {
      entry.mesh.updateWorldMatrix(true, false);
      if (entry.mesh.matrixWorld.equals(_IDENTITY)) {
        editStat.textContent = 'Object is not transformed — nothing to bake.'; return;
      }
      const params = { matrix: Array.from(entry.mesh.matrixWorld.elements) };
      editStat.textContent = 'Baking transform…';
      return runEdit('/api/edit/apply', { path: srcPath, op: 'transform', params }, li,
        r => `Saved transformed ${r.kind} (${r.total.toLocaleString()} pts) — reloading`);
    }

    let params = {};
    if (op === 'decimate') params = { factor: parseInt(facInput.value) || 2 };
    else if (op === 'denoise_sor') params = { nb_neighbors: parseInt(sorNb.value) || 20, std_ratio: parseFloat(sorStd.value) || 2 };
    else if (op === 'crop') {
      params = { min: cMin.map(i => parseFloat(i.value)), max: cMax.map(i => parseFloat(i.value)), invert: invCb.checked };
      if (params.min.concat(params.max).some(v => !isFinite(v))) {
        editStat.textContent = 'Place a 3D crop box or fill in the bounds first.'; return;
      }
    }

    // Scope decimate/denoise to the visibility box when requested. The server
    // works on the file's local coords, so fold in the cloud's transform:
    // Q = inv(boxWorld) · cloudWorld. region_invert follows the box's view side.
    if (regionScoped(op) && regionCb.checked) {
      if (!visBox) { editStat.textContent = 'Place a visibility box first, or untick "Limit to visibility box".'; return; }
      visBox.updateWorldMatrix(true, false);
      const Q = new THREE.Matrix4().copy(visBox.matrixWorld).invert().multiply(cloudWorldMatrix(entry));
      params.matrix = Array.from(Q.elements);
      params.region_invert = visModeOutside;
    }
    editStat.textContent = `Applying ${op}…`;
    return runEdit('/api/edit/apply', { path: srcPath, op, params }, li,
      r => `Kept ${r.kept.toLocaleString()} / ${r.total.toLocaleString()} — reloading`,
      poseOf(entry.mesh, entry));
  };

  // Non-destructive revert: reload the original file the edits derived from.
  const revertBtn = el('button', { class: 'act', style: 'background:#553', onclick: async () => {
    const entry = State.selectedSTL;
    if (!entry || !objPaths[entry.name]) { editStat.textContent = 'Select an edited object first.'; return; }
    const cur = objPaths[entry.name];
    const orig = cur.replace(/(_edited|_recoloured)+\.ply$/i, '.ply');
    if (orig === cur) { editStat.textContent = 'This is already an original.'; return; }
    editStat.textContent = 'Reverting to original…';
    const li = State.selectedListItem;
    try {
      await loadPlyFromServer(orig, orig.split('/').pop(), poseOf(entry.mesh, entry));
      if (li) li.querySelector('button[title="Remove"]')?.click();
    } catch (e) { editStat.textContent = 'Original not found: ' + e.message; }
  } }, 'Revert to original');

  // ── Save As (export the selected cloud/splat under a new name) ──
  const saveAsInput = el('input', { placeholder: 'new_name.ply' });
  const bakeCb = el('input', { type: 'checkbox', style: 'width:auto;flex:0 0 auto;margin:0' });
  async function saveAs() {
    const entry = State.selectedSTL;
    if (!entry) { editStat.textContent = 'Select an object to save first.'; return; }
    const srcPath = objPaths[entry.name];
    if (!srcPath) { editStat.textContent = 'Save As needs a loaded file (Library/generate).'; return; }
    let name = saveAsInput.value.trim();
    if (!name) { editStat.textContent = 'Enter a name for Save As.'; return; }
    if (!/\.ply$/i.test(name)) name += '.ply';
    name = name.replace(/[/\\]/g, '_');                  // keep it a plain filename
    const output = srcPath.replace(/\/[^/]+$/, '') + '/' + name;   // same folder as source
    // If the cloud has live-erased points, write the survivors (drop those
    // indices) rather than copying the untouched file.
    const erA = entry.isPointCloud ? entry.mesh.geometry.getAttribute('aErased') : null;
    const erased = [];
    if (erA) { const a = erA.array; for (let i = 0; i < a.length; i++) if (a[i] > 0.5) erased.push(i); }

    saveAsBtn.disabled = true;
    editStat.textContent = `Saving as ${name}…`;
    try {
      if (erased.length) {
        // Commit live erasures: keep all but the erased indices, into the new file.
        await api('/api/edit/apply', { path: srcPath, op: 'drop', params: { drop: erased },
          output, pose: poseOf(entry.mesh, entry) });
        editStat.textContent = `Saved as ${name} (${erased.length.toLocaleString()} erased)`;
      } else {
        // Bake = self-contained file with the transform baked into the coords;
        // otherwise a lossless copy plus a pose sidecar that reloads here.
        const body = { path: srcPath, output };
        if (bakeCb.checked) body.matrix = Array.from(cloudWorldMatrix(entry).elements);
        else body.pose = poseOf(entry.mesh, entry);
        await api('/api/edit/save_as', body);
        editStat.textContent = `Saved as ${name}${bakeCb.checked ? ' (baked)' : ''}`;
      }
      refresh();
    } catch (e) { editStat.textContent = 'Error: ' + e.message; }
    saveAsBtn.disabled = false;
  }
  const saveAsBtn = el('button', { class: 'act', style: 'margin-top:2px', onclick: saveAs }, 'Save As');
  const pSaveAs = el('label', { class: 'muted', style: 'display:flex;align-items:center;gap:6px;margin-top:4px' },
    bakeCb, 'Bake transform into coordinates (portable)');

  // ── Erase with primitives (Cube/Sphere/Cylinder act as eraser volumes) ──
  // Point clouds: turn the eraser on, then drag a primitive through the target
  // cloud — points vanish live as the volume sweeps over them (saved on Save As).
  // Splats can't be edited live, so they use a one-shot "Erase now".
  const eraseStat = el('div', { class: 'muted' },
    'Add a Cube/Sphere/Cylinder, then erase.');
  const eraseUndo = [];                 // unified undo stack (capped)
  let eraserOn = false, eraserTarget = null;
  function refreshUndoBtn() { if (undoEraseBtn) undoEraseBtn.disabled = eraseUndo.length === 0; }

  // Live eraser (point clouds) — driven by dragging a primitive's gizmo.
  const _erTmp = new THREE.Vector3();
  let _sweepPending = false;
  function requestSweep() {
    if (_sweepPending) return;
    _sweepPending = true;
    requestAnimationFrame(() => { _sweepPending = false; eraseSweep(); });
  }
  function eraseSweep() {
    if (!eraserOn || !eraserTarget || !State.importedSTLs.includes(eraserTarget)) return;
    const geo = eraserTarget.mesh.geometry;
    const pos = geo.getAttribute('position'), erA = geo.getAttribute('aErased');
    if (!pos || !erA) return;
    const prims = State.importedSTLs.filter(s => s.primType && s.mesh.visible);
    if (!prims.length) return;
    const cloudWorld = cloudWorldMatrix(eraserTarget);
    // Per primitive: matrix mapping a cloud-file point straight to canonical space.
    const mats = prims.map(p => {
      const e = eraserMatrix(p, cloudWorld);
      return { type: e.type, m: new THREE.Matrix4().fromArray(e.matrix) };
    });
    const arr = erA.array, p = pos.array;
    let changed = false;
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] > 0.5) continue;
      const o = i * 3;
      for (const { type, m } of mats) {
        const v = _erTmp.set(p[o], p[o + 1], p[o + 2]).applyMatrix4(m);
        const inside = type === 'sphere' ? (v.x * v.x + v.y * v.y + v.z * v.z) <= 0.25
          : type === 'cylinder' ? (v.x * v.x + v.z * v.z) <= 0.25 && Math.abs(v.y) <= 0.5
          : Math.abs(v.x) <= 0.5 && Math.abs(v.y) <= 0.5 && Math.abs(v.z) <= 0.5;
        if (inside) { arr[i] = 1; changed = true; break; }
      }
    }
    if (changed) { erA.needsUpdate = true; State.requestRender(); }
  }

  // One-shot erase (splats, or a non-interactive cloud erase): delete points
  // inside the current primitives server-side.
  async function eraseOneShot() {
    const entry = State.selectedSTL;
    if (!entry || (!entry.isPointCloud && !entry.isSplat)) {
      eraseStat.textContent = 'Select the cloud/splat to erase from (in the list).'; return; }
    const srcPath = objPaths[entry.name];
    if (!srcPath) { eraseStat.textContent = 'Erase needs a loaded file (Library/generate).'; return; }
    const prims = State.importedSTLs.filter(s => s.primType && s.mesh.visible);
    if (!prims.length) { eraseStat.textContent = 'Add a Cube/Sphere/Cylinder to use as an eraser.'; return; }
    const cloudWorld = cloudWorldMatrix(entry);
    const erasers = prims.map(p => eraserMatrix(p, cloudWorld));
    const snap = { kind: 'oneshot', buffer: entry._buffer, name: entry.name + '.ply',
      pose: poseOf(entry.mesh, entry), path: srcPath };
    eraseOneShotBtn.disabled = true;
    const newEntry = await runEdit('/api/edit/apply', { path: srcPath, op: 'erase', params: { erasers } },
      State.selectedListItem,
      r => `Erased ${(r.total - r.kept).toLocaleString()} — ${r.kept.toLocaleString()} left`,
      poseOf(entry.mesh, entry));
    eraseOneShotBtn.disabled = false;
    if (newEntry && snap.buffer) {
      eraseUndo.push({ ...snap, newEntry }); if (eraseUndo.length > 8) eraseUndo.shift(); refreshUndoBtn();
    }
  }

  async function undoLastErase() {
    const u = eraseUndo.pop();
    if (!u) { eraseStat.textContent = 'Nothing to undo.'; return; }
    refreshUndoBtn();
    if (u.kind === 'live') {
      // Restore the cloud's pre-stroke erased mask.
      if (State.importedSTLs.includes(u.target)) {
        const erA = u.target.mesh.geometry.getAttribute('aErased');
        if (erA) { erA.array.set(u.snapshot); erA.needsUpdate = true; State.requestRender(); }
      }
      eraseStat.textContent = 'Eraser stroke undone.';
      return;
    }
    // One-shot: remove the erased result, re-add the pre-erase object from buffer.
    if (u.newEntry && State.importedSTLs.includes(u.newEntry)) {
      const idx = State.importedSTLs.indexOf(u.newEntry);
      document.querySelectorAll('.stl-item')[idx]?.querySelector('button[title="Remove"]')?.click();
    }
    objPaths[u.name.replace(/\.ply$/i, '')] = u.path;
    const restored = await loadPLYFile(new File([u.buffer], u.name, { type: 'application/octet-stream' }), u.pose);
    reselect(restored);
    State.requestRender();
    eraseStat.textContent = 'Erase undone.';
  }

  const eraserToggle = el('button', { class: 'act', style: 'flex:1', onclick: () => {
    if (!eraserOn) {
      const e = State.selectedSTL;
      if (e && e.isPointCloud) eraserTarget = e;
      if (!eraserTarget) { eraseStat.textContent = 'Select a point cloud first, then turn the eraser on.'; return; }
      eraserOn = true;
    } else { eraserOn = false; }
    eraserToggle.textContent = eraserOn ? 'Live eraser: ON' : 'Live eraser: OFF';
    eraserToggle.style.background = eraserOn ? '#a33' : '';
    eraseStat.textContent = eraserOn
      ? `Erasing "${eraserTarget.name}" — drag a primitive through it.`
      : 'Eraser off.';
  } }, 'Live eraser: OFF');
  const eraseOneShotBtn = el('button', { class: 'act', style: 'flex:1', onclick: eraseOneShot }, 'Erase now');
  const undoEraseBtn = el('button', { class: 'act', disabled: true, onclick: undoLastErase }, 'Undo erase');
  const pErase = el('div', {},
    eraseStat,
    el('div', { class: 'row' }, eraserToggle, eraseOneShotBtn),
    undoEraseBtn);

  // Live-erase hooks: snapshot at drag start, sweep continuously while dragging
  // a primitive. Guarded so normal object transforms are unaffected.
  const _stlTC = State.stlTransformControls;
  if (_stlTC) {
    _stlTC.addEventListener('dragging-changed', (ev) => {
      if (!ev.value || !eraserOn || !eraserTarget) return;
      if (!State.selectedSTL || !State.selectedSTL.primType) return;
      const erA = eraserTarget.mesh.geometry?.getAttribute('aErased');
      if (erA) { eraseUndo.push({ kind: 'live', target: eraserTarget, snapshot: erA.array.slice() });
        if (eraseUndo.length > 8) eraseUndo.shift(); refreshUndoBtn(); }
    });
    _stlTC.addEventListener('objectChange', () => {
      if (eraserOn && State.selectedSTL && State.selectedSTL.primType) requestSweep();
    });
  }

  // Workflow save/load (explicit — no auto-restore).
  const wfStat = el('div', { class: 'muted' }, '');
  const saveWfBtn = el('button', { class: 'act', style: 'flex:1' }, 'Save Workflow');
  const loadWfBtn = el('button', { class: 'act', style: 'flex:1' }, 'Load Workflow');

  const panel = el('div', { id: 'lidar-panel' },
    el('h4', {}, 'Workflow'),
    el('div', { class: 'row' }, saveWfBtn, loadWfBtn),
    wfStat,
    el('h4', {}, 'Library'),
    projectInput,
    el('button', { class: 'act', onclick: refresh }, 'Refresh outputs'),
    outList,
    el('h4', {}, 'Generate'),
    el('div', { class: 'muted' }, 'Scan folder (raw bags):'), scanInput,
    el('div', { class: 'row' }, typeSel, methodSel),
    pSeed,
    el('div', { class: 'row' },
      el('label', { class: 'muted', style: 'flex:1' }, 'voxel', voxelInput),
      el('label', { class: 'muted', style: 'flex:1' }, 'noise σ', sorInput)),
    pSplatSize,
    genBtn, barLabel, barWrap, logBox,
    el('h4', {}, 'Visibility box'),
    pVis,
    el('h4', {}, 'Erase (primitives)'),
    pErase,
    el('h4', {}, 'Edit selected'),
    editStat, editOp, pDec, pSor, pRegion, pCrop, pRecol, pTransform, applyEditBtn, revertBtn,
    el('div', { class: 'muted', style: 'margin-top:6px' }, 'Save selected as:'),
    saveAsInput, pSaveAs, saveAsBtn);

  // Gather/apply the whole workflow form — driven by the Save/Load buttons
  // (no auto-restore; the user controls when state is saved or loaded).
  function collectWorkflow() {
    return {
      project: projectInput.value, scan: scanInput.value,
      type: typeSel.value, method: methodSel.value,
      voxel: voxelInput.value, noise: sorInput.value, splatSize: splatSizeInput.value,
      editOp: editOp.value, factor: facInput.value,
      sorNb: sorNb.value, sorStd: sorStd.value,
      cropInvert: invCb.checked, regionLimit: regionCb.checked,
      showOutside: visModeOutside,
    };
  }
  function applyWorkflow(s) {
    if (!s) return false;
    const setVal = (elm, k) => { if (s[k] != null) elm.value = s[k]; };
    setVal(projectInput, 'project'); setVal(scanInput, 'scan');
    setVal(typeSel, 'type'); setVal(methodSel, 'method');
    setVal(voxelInput, 'voxel'); setVal(sorInput, 'noise'); setVal(splatSizeInput, 'splatSize');
    setVal(editOp, 'editOp'); setVal(facInput, 'factor');
    setVal(sorNb, 'sorNb'); setVal(sorStd, 'sorStd');
    if (typeof s.cropInvert === 'boolean') invCb.checked = s.cropInvert;
    if (typeof s.regionLimit === 'boolean') regionCb.checked = s.regionLimit;
    if (typeof s.showOutside === 'boolean') { visModeOutside = s.showOutside; updateVisLabels(); pushVisClip(); }
    showOp(); showGen();
    return true;
  }
  saveWfBtn.onclick = () => {
    saveWorkflowState(collectWorkflow());
    wfStat.textContent = 'Workflow saved.';
  };
  loadWfBtn.onclick = () => {
    if (applyWorkflow(loadState())) { wfStat.textContent = 'Workflow loaded.'; refresh(); }
    else wfStat.textContent = 'No saved workflow.';
  };

  showOp();

  // Mount inside the existing right-side control panel as a collapsible section
  // (no separate floating bar).
  const arrow = el('span', {}, '▾');
  const head = el('div', { id: 'lidar-head', onclick: () => {
    const hidden = panel.style.display === 'none';
    panel.style.display = hidden ? 'block' : 'none';
    arrow.textContent = hidden ? '▾' : '▸';
  } }, el('span', {}, 'LiDAR Workflow'), arrow);

  const host = document.getElementById('panel') || document.body;
  host.insertBefore(panel, host.firstChild);
  host.insertBefore(head, panel);

  // Resolve the gizmo for the currently-active box (falling back to whichever
  // box exists if none has been explicitly focused yet).
  const activeGizmo = () => {
    if (activeBox === visBox && visBox) return visGizmo;
    if (activeBox === cropBox && cropBox) return cropGizmo;
    return (visBox && visGizmo) ? visGizmo : (cropBox && cropGizmo) ? cropGizmo : null;
  };

  // T / R / S switch the active box gizmo's mode, exactly like imported
  // objects. While a box is up it takes priority, so the capture-phase
  // listener stops the global handler (main.js) from also moving the
  // selected object. With no box active, keys fall through unchanged.
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    const g = activeGizmo();
    if (!g || !g.object) return;
    const mode = (e.key === 't' || e.key === 'T') ? 'translate'
               : (e.key === 'r' || e.key === 'R') ? 'rotate'
               : (e.key === 's' || e.key === 'S') ? 'scale' : null;
    if (!mode) return;
    g.setMode(mode);
    e.stopImmediatePropagation();
    State.requestRender();
  }, true);

  // Clicking a box (not a drag) makes it the active T/R/S target.
  const dom = State.renderer.domElement;
  const _boxRay = new THREE.Raycaster();
  let _bdX = 0, _bdY = 0, _bdValid = false;
  dom.addEventListener('pointerdown', (e) => { _bdValid = e.button === 0; _bdX = e.clientX; _bdY = e.clientY; });
  dom.addEventListener('pointerup', (e) => {
    if (!_bdValid || e.button !== 0) { _bdValid = false; return; }
    _bdValid = false;
    if (Math.hypot(e.clientX - _bdX, e.clientY - _bdY) > 5) return;   // a drag, not a click
    const boxes = [visBox, cropBox].filter(Boolean);
    if (boxes.length < 2) return;   // nothing to switch between
    const r = dom.getBoundingClientRect();
    const m = new THREE.Vector2(
      ((e.clientX - r.left) / r.width) * 2 - 1,
      -((e.clientY - r.top) / r.height) * 2 + 1);
    _boxRay.setFromCamera(m, State.activeCamera || State.camera);
    const hit = _boxRay.intersectObjects(boxes, false)[0];
    if (hit) setActiveBox(boxes.find(b => b === hit.object));
  });

  refresh();
}
