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

// Folder chooser: try the server's native (tkinter) picker first, and fall back
// to an in-browser directory navigator when the server has no display/tkinter.
// Resolves with the chosen absolute path, or null if the user cancelled.
async function browseFolder(title, initial) {
  try {
    const d = await api('/api/browse', { title, initial: initial || '' });
    if (d.path) return d.path;
    if (d.cancelled) return null;      // native dialog opened, user cancelled
    // else: no tkinter — fall through to the in-browser navigator
  } catch { /* server error — fall through */ }
  return pickFolderInBrowser(initial);
}

// In-browser directory navigator (fallback picker). Lists subdirectories via
// /api/browse/dir and resolves with the chosen path, or null if cancelled.
function pickFolderInBrowser(initial) {
  return new Promise((resolve) => {
    let cur = initial || '', parent = null;
    const crumb = el('div', { class: 'ls-nav-crumb' });
    const list = el('div', { class: 'ls-nav-list' });
    const upBtn = el('button', { class: 'act ls-nav-btn' }, '⬆ Up');
    const selBtn = el('button', { class: 'act ls-nav-btn' }, 'Select this folder');
    const cancelBtn = el('button', { class: 'act ls-nav-btn ls-nav-cancel' }, 'Cancel');
    const overlay = el('div', { class: 'ls-nav-overlay' },
      el('div', { class: 'ls-nav-box' },
        el('div', { class: 'ls-nav-head' }, crumb),
        list,
        el('div', { class: 'ls-nav-foot' }, upBtn, selBtn, cancelBtn)));
    const done = (v) => { overlay.remove(); resolve(v); };
    async function go(path) {
      list.replaceChildren(el('div', { class: 'muted' }, 'Loading…'));
      try {
        const d = await api('/api/browse/dir', path ? { path } : {});
        cur = d.path; parent = d.parent;
        crumb.textContent = d.path;
        upBtn.disabled = !d.parent;
        list.replaceChildren(...(d.items.length
          ? d.items.map((it) => {
              const row = el('div', { class: 'ls-nav-item' }, '📁 ' + it.name);
              row.addEventListener('click', () => go(it.path));
              return row;
            })
          : [el('div', { class: 'muted', style: 'padding:6px' }, '(no subfolders)')]));
      } catch (e) {
        list.replaceChildren(el('div', { class: 'muted', style: 'padding:6px' }, 'Error: ' + e.message));
      }
    }
    upBtn.addEventListener('click', () => { if (parent) go(parent); });
    selBtn.addEventListener('click', () => done(cur));
    cancelBtn.addEventListener('click', () => done(null));
    overlay.addEventListener('click', (e) => { if (e.target === overlay) done(null); });
    document.body.appendChild(overlay);
    go(cur);   // empty → server defaults to the home directory
  });
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

// Return a server-side file path for an object, uploading it first if needed.
// Library/generated objects already have one (loadPlyFromServer registers it);
// files opened via the picker/drag-drop only exist as a browser buffer, so the
// server edit ops have nothing to run on. The first edit uploads those bytes
// into the Library folder (dir) and caches the returned path, after which the
// imported object behaves exactly like a Library-loaded one. Returns the path;
// throws Error with a specific, user-visible reason on failure.
async function ensureObjPath(entry, dir) {
  if (!entry) throw new Error('no object selected');
  if (objPaths[entry.name]) return objPaths[entry.name];
  let buf = entry._buffer;
  let len = buf ? (buf.byteLength ?? buf.length ?? 0) : 0;
  // The splat viewer can detach the original ArrayBuffer when it hands it to a
  // worker, leaving _buffer zero-length. The import kept a blob URL holding a
  // copy of the original bytes — recover them from there.
  if (!len && entry._blobUrl) {
    try {
      buf = await (await fetch(entry._blobUrl)).arrayBuffer();
      len = buf.byteLength;
    } catch { /* fall through to the error below */ }
  }
  if (!len) throw new Error('no data bytes to upload — reload the file');
  // Land the file in the project's type subfolder (splats/ or pointclouds/),
  // the same layout generate uses and the Library lists — not the project root.
  const sub = entry.isSplat ? 'splats' : entry.isPointCloud ? 'pointclouds' : 'exports';
  const target = dir ? dir.replace(/\/+$/, '') + '/' + sub : '';
  const q = new URLSearchParams({ dir: target, name: entry.name + '.ply' });
  let resp;
  try {
    resp = await fetch(`/api/edit/import?${q.toString()}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: buf,
    });
  } catch (e) { throw new Error('upload request failed: ' + e.message); }
  if (resp.status === 404) throw new Error('server missing /api/edit/import — restart the server');
  if (!resp.ok) throw new Error(`upload failed (HTTP ${resp.status})`);
  let j;
  try { j = await resp.json(); } catch { throw new Error('bad server response to upload'); }
  if (j.error || !j.output) throw new Error('server: ' + (j.error || 'no output path'));
  objPaths[entry.name] = j.output;
  return j.output;
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

// The pose a splat should inherit from the cloud it was generated from, so it
// reloads exactly where that seed sits. A splat is built in its seed's own
// coordinate frame, but the seed's placement is a viewer-only transform the
// fresh file wouldn't otherwise pick up. Prefer the seed's live viewer pose;
// fall back to its saved .pose.json (an edited seed that isn't loaded here).
async function seedPose(seedPath) {
  if (!seedPath) return null;                       // 'auto' seed → no known frame
  const entry = (State.importedSTLs || []).find((e) => objPaths[e.name] === seedPath);
  const pose = entry && entry.mesh ? poseOf(entry.mesh, entry) : await fetchPose(seedPath);
  if (pose) pose.visible = true;                    // a freshly generated splat should show
  return pose;
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

// One shared explanation panel, docked just left of the control panel. Info
// icons write into it rather than popping their own bubble (which the panel's
// overflow clipping + label re-click behaviour made unreliable).
let _infoPanel = null;
function getInfoPanel() {
  if (_infoPanel) return _infoPanel;
  const body = el('div', { class: 'ls-info-body' });
  const x = el('span', { class: 'ls-info-x', title: 'Close' }, '✕');
  const heading = el('span', {}, 'Field info');
  const panel = el('div', { id: 'ls-info-panel' },
    el('div', { class: 'ls-info-title' }, heading, x), body);
  const close = () => {
    panel.style.display = 'none';
    panel._for = null;
    document.querySelectorAll('.ls-info.on').forEach((i) => i.classList.remove('on'));
  };
  x.addEventListener('click', close);
  document.body.appendChild(panel);
  Object.assign(panel, { _body: body, _heading: heading, _close: close });
  _infoPanel = panel;
  return panel;
}

// A clickable ⓘ that shows `text` (headed by `title`) in the shared side panel.
// Clicking the same icon again closes it; clicking another switches content.
function infoIcon(text, title = 'Field info') {
  const icon = el('span', { class: 'ls-info', title: 'What is this?' }, 'ⓘ');
  icon.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    const panel = getInfoPanel();
    if (panel.style.display === 'block' && panel._for === icon) { panel._close(); return; }
    document.querySelectorAll('.ls-info.on').forEach((i) => i.classList.remove('on'));
    icon.classList.add('on');
    panel._heading.textContent = title;
    panel._body.textContent = text;
    panel._for = icon;
    panel.style.display = 'block';
  });
  return icon;
}

// Startup hardware check: a green/amber/red light for CPU, RAM and GPU. If all
// three are green it self-dismisses after 1s; otherwise it stays until the user
// acknowledges (so a weak machine can't be missed).
async function showStartupHealth() {
  let data;
  try {
    const r = await fetch('/api/system');
    if (!r.ok) return;
    data = await r.json();
  } catch { return; }                       // endpoint unavailable → skip silently
  const COLOR = { green: '#3fbf5f', amber: '#e0a02a', red: '#d94a3d' };
  const row = (label, info) => el('div',
    { style: 'display:flex;align-items:center;gap:11px;margin:8px 0' },
    el('span', { style:
      `flex:0 0 auto;width:13px;height:13px;border-radius:50%;` +
      `background:${COLOR[info && info.grade] || '#777'};` +
      `box-shadow:0 0 7px ${COLOR[info && info.grade] || '#777'}` }),
    el('span', { style: 'flex:0 0 40px;font-weight:600' }, label),
    el('span', { style: 'flex:1;color:#9fb0c8;font-size:12px' }, (info && info.detail) || '—'));
  const allGreen = ['cpu', 'ram', 'gpu'].every((k) => data[k] && data[k].grade === 'green');
  const card = el('div', { style:
    'position:fixed;top:22px;left:50%;transform:translateX(-50%);z-index:4000;' +
    'min-width:290px;background:rgba(18,22,30,0.97);border:1px solid #37506e;' +
    'border-radius:10px;padding:14px 18px;color:#e6ecf5;font:13px system-ui;' +
    'box-shadow:0 8px 40px rgba(0,0,0,0.6);transition:opacity .4s' },
    el('div', { style: 'font-weight:700;letter-spacing:.03em;margin-bottom:6px' }, 'System check'),
    row('CPU', data.cpu), row('RAM', data.ram), row('GPU', data.gpu));
  document.body.appendChild(card);
  if (allGreen) {
    setTimeout(() => { card.style.opacity = '0'; }, 3000);
    setTimeout(() => card.remove(), 3450);
  } else {
    card.append(
      el('div', { style: 'margin-top:8px;color:#d6b24a;font-size:11px;max-width:300px' },
        'Some resources are limited — high-quality splats may run slowly or run out ' +
        'of GPU memory. Lower the quality slider (e.g. downscale 2, fewer splats) if so.'),
      el('button', { style:
        'margin-top:10px;width:100%;background:#2a6df0;color:#fff;border:0;border-radius:6px;' +
        'padding:8px;font-weight:600;cursor:pointer', onclick: () => card.remove() }, 'OK'));
  }
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
  #lidar-panel input[type=checkbox]{width:auto;flex:0 0 auto;margin:0;padding:0;cursor:pointer}
  #lidar-panel button.act{width:100%;background:#2a6df0;color:#fff;border:0;border-radius:6px;
    padding:8px;margin-top:6px;font-weight:600;cursor:pointer}
  #lidar-panel button.act:disabled{opacity:.5;cursor:default}
  #lidar-panel .row{display:flex;gap:6px}
  #lidar-panel .item{display:flex;justify-content:space-between;align-items:center;gap:6px;
    padding:5px 6px;border:1px solid #283447;border-radius:5px;margin:3px 0;background:#121a26}
  #lidar-panel .item>span{flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  #lidar-panel .item button{flex:0 0 auto;width:auto;margin:0;background:#2c7;border:0;color:#04210f;border-radius:4px;padding:3px 9px;
    font-weight:700;cursor:pointer;font-size:11px}
  #lidar-panel .item button.del{background:#a33;color:#fff}
  #lidar-panel .item button:disabled{opacity:.5;cursor:default}
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
    margin-top:6px;max-height:120px;overflow:auto;white-space:pre-wrap}
  #ls-log-wrap{display:none;position:relative}
  #ls-log-copy{display:none;position:absolute;top:9px;right:6px;width:auto;margin:0;padding:2px 6px;
    background:#243247;color:#cdd8ec;border:1px solid #3a4a63;border-radius:4px;
    font-size:12px;line-height:1;cursor:pointer;opacity:.85}
  #ls-log-copy:hover{opacity:1;background:#2c3d55}
  #lidar-panel .ls-info{flex:0 0 auto;cursor:pointer;color:#5a86c0;
    font-size:12px;margin-left:4px;line-height:1;user-select:none}
  #lidar-panel .ls-info:hover{color:#8fb4e8}
  #lidar-panel .ls-info.on{color:#8fb4e8}
  #ls-info-panel{display:none;position:fixed;top:10px;right:280px;width:250px;z-index:3000;
    max-height:calc(100vh - 20px);overflow:auto;background:rgba(12,19,32,0.97);
    border:1px solid #37506e;border-radius:8px;padding:12px 14px;color:#cdd6e3;
    font:400 12px/1.6 system-ui;box-shadow:0 6px 30px rgba(0,0,0,.55)}
  #ls-info-panel .ls-info-title{display:flex;justify-content:space-between;align-items:center;
    gap:8px;font-weight:700;color:#9cf;font-size:11px;letter-spacing:.04em;
    text-transform:uppercase;margin-bottom:8px}
  #ls-info-panel .ls-info-x{cursor:pointer;color:#7d8aa0;font-size:15px;line-height:1;padding:0 2px}
  #ls-info-panel .ls-info-x:hover{color:#cdd6e3}
  #lidar-panel button.browse{width:auto;flex:0 0 auto;margin:3px 0;padding:6px 10px;font-size:13px;line-height:1}
  .ls-nav-overlay{position:fixed;inset:0;z-index:4000;background:rgba(4,8,14,.6);
    display:flex;align-items:center;justify-content:center}
  .ls-nav-box{width:min(560px,92vw);max-height:80vh;display:flex;flex-direction:column;
    background:#0f1620;border:1px solid #37506e;border-radius:10px;
    box-shadow:0 10px 40px rgba(0,0,0,.6);color:#cdd6e3;font:13px system-ui;overflow:hidden}
  .ls-nav-head{padding:10px 14px;border-bottom:1px solid #2a3344;background:#121a26}
  .ls-nav-crumb{font:11px monospace;color:#9cf;word-break:break-all}
  .ls-nav-list{flex:1 1 auto;overflow:auto;padding:4px}
  .ls-nav-item{padding:7px 10px;border-radius:5px;cursor:pointer;white-space:nowrap;
    overflow:hidden;text-overflow:ellipsis}
  .ls-nav-item:hover{background:#1b2636}
  .ls-nav-foot{display:flex;gap:8px;padding:10px 14px;border-top:1px solid #2a3344;background:#121a26}
  button.ls-nav-btn{flex:1 1 auto;width:auto;margin:0;background:#2a6df0;color:#fff;border:0;
    border-radius:6px;padding:8px;font-weight:600;cursor:pointer}
  button.ls-nav-btn:disabled{opacity:.5;cursor:default}
  button.ls-nav-cancel{background:#3a4a63}`;
  document.head.append(el('style', {}, css));

  // ── Library (existing outputs) ──
  const projectInput = el('input', { id: 'ls-project', value: DEFAULT_PROJECT });
  // The two working paths (library + scan folder) persist on their own — auto-
  // saved on every change and restored on load — independent of the explicit
  // Save/Load Workflow buttons.
  const PATHS_KEY = 'lidarStudio.paths';
  const persistPaths = () => {
    try {
      localStorage.setItem(PATHS_KEY, JSON.stringify(
        { project: projectInput.value, scan: scanInput.value }));
    } catch { /* quota/private mode */ }
  };
  const browseInto = async (input, title, after) => {
    const p = await browseFolder(title, input.value.trim());
    if (p) { input.value = p; persistPaths(); if (after) after(); }
  };
  const projBrowse = el('button', { class: 'act browse', title: 'Browse for library folder' }, '📁');
  projBrowse.addEventListener('click', () =>
    browseInto(projectInput, 'Select Library / Project Folder', refresh));
  projectInput.addEventListener('change', persistPaths);
  const outList = el('div', { id: 'ls-outputs' });
  const refresh = async () => {
    outList.replaceChildren(el('div', { class: 'muted' }, 'Loading…'));
    try {
      const data = await api('/api/project/outputs', { path: projectInput.value.trim() });
      const items = [
        ...(data.splats || []).map(f => ({ ...f, kind: 'splat' })),
        ...(data.meshes || []).map(f => ({ ...f, kind: 'mesh' })),
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
          el('span', { title: f.name }, `${f.kind === 'splat' ? '🟣' : f.kind === 'mesh' ? '🔷' : '⚪'} ${f.name} `,
            el('span', { class: 'muted' }, `${f.size_mb}MB`)),
          el('button', { onclick: async (e) => {
            e.target.disabled = true; e.target.textContent = '…';
            try { await loadPlyFromServer(f.path, f.name); e.target.textContent = '✓'; }
            catch (err) { e.target.textContent = 'err'; console.error(err); }
          } }, 'Load'),
          el('button', { class: 'del', title: 'Delete file (and its sidecars)', onclick: async (e) => {
            if (!confirm(`Delete ${f.name}?\nThis permanently removes the file and its .pose/.traj sidecars.`)) return;
            e.target.disabled = true; e.target.textContent = '…';
            try { await api('/api/project/delete', { path: f.path }); refresh(); }
            catch (err) { e.target.textContent = 'err'; e.target.disabled = false; console.error(err); }
          } }, '🗑'))
      ) : [el('div', { class: 'muted' }, 'No outputs in this project.')]));
    } catch (err) {
      outList.replaceChildren(el('div', { class: 'muted' }, 'Error: ' + err.message));
    }
  };

  // ── Generate ──
  const scanInput = el('input', { id: 'ls-scan', placeholder: '/path/to/scan_folder' });
  const scanBrowse = el('button', { class: 'act browse', title: 'Browse for scan folder' }, '📁');
  scanBrowse.addEventListener('click', () => browseInto(scanInput, 'Select Scan Folder'));
  scanInput.addEventListener('change', persistPaths);
  // Restore the persisted library + scan paths (overriding the defaults).
  try {
    const sp = JSON.parse(localStorage.getItem(PATHS_KEY) || '{}');
    if (sp.project) projectInput.value = sp.project;
    if (sp.scan) scanInput.value = sp.scan;
  } catch { /* ignore */ }
  const typeSel = el('select', {}, ...['splat', 'pointcloud', 'mesh'].map(t => el('option', { value: t }, t)));
  const voxelInput = el('input', { type: 'number', step: '0.005', min: '0.005', value: '0.01' });
  // voxel is used only by the point-cloud job (splats are always GPU-trained).
  const lblVoxel = el('label', { class: 'muted', style: 'flex:1' }, 'voxel',
    infoIcon('Point-cloud voxel size in metres. Smaller = denser, finer cloud but bigger '
      + 'and slower. 0.01 m is dense; 0.05 m is a light preview cloud.', 'Voxel'),
    voxelInput);
  const pVoxel = el('div', { class: 'row' }, lblVoxel);
  const monoCheck = el('input', { type: 'checkbox' });
  const lblMono = el('label', { class: 'muted', style: 'flex:1;display:flex;align-items:center;gap:6px' },
    monoCheck, 'monochromatic',
    infoIcon('Skip the slow multi-view photo colouring and shade points by LiDAR '
      + 'intensity (grey if unavailable). Much faster and needs no images — good for '
      + 'a quick geometry preview.', 'Monochromatic'));
  const pMono = el('div', { class: 'row' }, lblMono);
  const handleCheck = el('input', { type: 'checkbox', checked: true });
  const lblHandle = el('label', { class: 'muted', style: 'flex:1;display:flex;align-items:center;gap:6px' },
    handleCheck, 'remove handle',
    infoIcon('Drop the scanner\'s own handle/mount — a fixed cluster in the sensor '
      + 'frame that otherwise smears into a "snake" along the path. On by default; '
      + 'uncheck to keep it.', 'Remove handle'));
  const pHandle = el('div', { class: 'row' }, lblHandle);

  // ── Trained-splat quality controls ──────────────────────────────────────
  // One master 'quality' slider drives a set of individual controls (which
  // stay editable for fine-tuning). Presets follow the splat-quality campaign
  // arc: draft (fast preview) → max (the ds1 / 3M / 30k / sharpen-1.3 champion).
  // The shape/opacity recipe constants live as process_splat.py defaults; only
  // the resolution / count / sharpness knobs scale with quality here.
  // Caps stop at 3M: at downscale-1 that is the largest count that fits the
  // 24 GB campaign card (6M/ds1 never completed — see the scale-clamp note in
  // process_splat.py). 'max' is the validated champion recipe; it is the default.
  const QUALITY = [
    { name: 'draft',    iterations: 3000,  downscale: 4, cap: 1, dropBlurry: 0,    undistortScale: 1.0 },
    { name: 'balanced', iterations: 7000,  downscale: 2, cap: 3, dropBlurry: 0.15, undistortScale: 1.0 },
    { name: 'high',     iterations: 15000, downscale: 1, cap: 3, dropBlurry: 0.15, undistortScale: 1.0 },
    { name: 'max',      iterations: 30000, downscale: 1, cap: 3, dropBlurry: 0.15, undistortScale: 1.3 },
  ];
  const qSlider = el('input', { type: 'range', min: '1', max: '4', step: '1', value: '4', style: 'flex:1' });
  const qName = el('span', { class: 'muted', style: 'min-width:64px;text-align:right' }, 'max');
  const iterInput = el('input', { type: 'number', step: '1000', min: '500', value: '30000' });
  const downInput = el('input', { type: 'number', step: '1', min: '1', max: '8', value: '1' });
  const capInput = el('input', { type: 'number', step: '0.5', min: '0.5', value: '3' });      // millions
  const dropInput = el('input', { type: 'number', step: '0.05', min: '0', max: '0.9', value: '0.15' });
  const usInput = el('input', { type: 'number', step: '0.1', min: '1', max: '2', value: '1.3' });
  // Anisotropy cap: independent expert override (NOT driven by the quality
  // preset). 20:1 is the champion; raise for cables/thin structure.
  const anisoInput = el('input', { type: 'number', step: '5', min: '2', max: '200', value: '20' });
  const sfmInput = el('input', { placeholder: 'SfM poses .npz (optional, biggest quality lever)' });
  // Repopulate the individual controls from a quality preset (1-based index).
  const applyQuality = (idx) => {
    const q = QUALITY[Math.min(3, Math.max(0, idx - 1))];
    qName.textContent = q.name;
    iterInput.value = q.iterations; downInput.value = q.downscale;
    capInput.value = q.cap; dropInput.value = q.dropBlurry; usInput.value = q.undistortScale;
  };
  qSlider.addEventListener('input', () => applyQuality(parseInt(qSlider.value)));
  const qLbl = (t, inp, info) =>
    el('label', { class: 'muted', style: 'flex:1' }, t, infoIcon(info, t), inp);
  const pTrained = el('div', {},
    el('div', { class: 'row', style: 'align-items:center' },
      el('span', { class: 'muted', style: 'min-width:52px' }, 'quality'), qSlider, qName,
      infoIcon('Master preset, from draft (fast preview) to max — the campaign champion: '
        + 'full resolution, 3M splats, 30k iterations, sharpen 1.3 (~3–4 h; the largest '
        + 'count that fits a 24 GB card at full res). Defaults to max. Moving the slider '
        + 'sets the individual controls below, which you can still fine-tune by hand.', 'Quality')),
    el('div', { class: 'row' },
      qLbl('iterations', iterInput,
        'How many training steps run. More = sharper and better-converged but slower '
        + '(roughly linear in time). ~7k is a good balance; 30k is diminishing returns.'),
      qLbl('downscale', downInput,
        'Training-image shrink factor. 1 = full resolution (sharpest, most RAM/time); '
        + '2–4 are faster previews. If a result looks soft, drop this toward 1.')),
    el('div', { class: 'row' },
      qLbl('max splats (M)', capInput,
        'Cap on how many Gaussians (in millions) the trainer may grow to. More = finer '
        + 'detail but larger files and more GPU memory. ~3M is the most that fits a 24 GB '
        + 'card at full resolution (downscale 1); go higher only on a bigger GPU.'),
      qLbl('drop blurry', dropInput,
        'Fraction of the blurriest source photos to discard before training (ranked by '
        + 'Laplacian sharpness). 0.15 drops the shakiest ~15%; 0 keeps every frame. '
        + 'Removes motion-blurred photos that would soften the result.')),
    el('div', { class: 'row' },
      qLbl('sharpen ×', usInput,
        "Undistorted-canvas size vs the source photo. Above 1 preserves the fisheye "
        + "centre's native detail that a 1:1 wide-angle pinhole under-samples. 1.3 sharpens "
        + 'the centre at ~1.7× training cost.'),
      qLbl('aniso cap', anisoInput,
        'Max stretch (long-axis : short-axis) of a single Gaussian. 20:1 is the champion — '
        + 'it bounds each splat’s footprint (and VRAM). Raise toward 50–100 for scenes with '
        + 'cables / thin structures so they can form needle-shaped splats; the absolute-scale '
        + 'clamp still keeps needles from blowing up GPU memory. Not changed by the quality slider.')),
    el('div', { class: 'row', style: 'align-items:center' },
      el('label', { class: 'muted', style: 'flex:1' }, 'SfM poses', sfmInput),
      infoIcon('Optional .npz of camera poses from Structure-from-Motion (COLMAP/GLOMAP, '
        + 'aligned to the LiDAR frame). The single biggest quality lever (~+4 dB, '
        + 'walls-to-parity) — it corrects per-frame camera drift. Leave empty to use the '
        + 'LiDAR-odometry poses.', 'SfM poses')));
  // Seed cloud: 'auto' uses the latest project cloud; any other entry pins the
  // splat to that exact cloud — e.g. an edited one.
  const seedSel = el('select', {}, el('option', { value: '' }, 'seed cloud: auto'));
  const pSeed = el('div', { class: 'row', style: 'align-items:center' }, seedSel,
    infoIcon("Which point cloud the splat is trained from. 'auto' uses the latest project "
      + 'cloud; choose a specific or edited cloud to pin the splat to it.', 'Seed cloud'));
  const showGen = () => {
    const type = typeSel.value;
    // Seed cloud applies to splats and meshes (both reconstruct from a cloud);
    // the trained-quality panel is splat-only; voxel is point-cloud-only.
    pSeed.style.display = (type === 'splat' || type === 'mesh') ? 'flex' : 'none';
    pTrained.style.display = type === 'splat' ? 'block' : 'none';
    pVoxel.style.display = type === 'pointcloud' ? 'flex' : 'none';
    pMono.style.display = type === 'pointcloud' ? 'flex' : 'none';
    pHandle.style.display = type === 'pointcloud' ? 'flex' : 'none';
  };
  typeSel.addEventListener('change', showGen);
  showGen();
  const barMsg = el('span', { class: 'ls-bar-msg' }, 'Starting…');
  const barPct = el('span', { class: 'ls-bar-pct' }, '');
  const barLabel = el('div', { id: 'ls-bar-label' }, barMsg, barPct);
  const barFill = el('div', { id: 'ls-bar' });
  const barWrap = el('div', { id: 'ls-bar-wrap' }, barFill);
  const logBox = el('div', { id: 'ls-log' });
  // Copy icon overlaid on the log so an error is one click to copy.
  const logCopyBtn = el('button', { id: 'ls-log-copy', title: 'Copy output',
    onclick: async () => {
      try {
        await navigator.clipboard.writeText(logBox.textContent.trim());
        logCopyBtn.textContent = '✓';
        setTimeout(() => { logCopyBtn.textContent = '📋'; }, 1200);
      } catch (e) { console.error('copy failed', e); }
    } }, '📋');
  const logWrap = el('div', { id: 'ls-log-wrap' }, logBox, logCopyBtn);
  const genBtn = el('button', { class: 'act' }, 'Generate');
  // Stop button: interrupts the running job (shown only while one runs).
  const stopBtn = el('button', { class: 'act', style: 'display:none;background:#a33' }, 'Stop');
  let currentJobId = null;
  const endRun = () => {
    genBtn.disabled = false;
    stopBtn.style.display = 'none';
    stopBtn.disabled = false; stopBtn.textContent = 'Stop';
    currentJobId = null;
  };
  stopBtn.onclick = async () => {
    if (!currentJobId) return;
    stopBtn.disabled = true; stopBtn.textContent = 'Stopping…';
    try { await api('/api/process/cancel', { job_id: currentJobId }); }
    catch (e) { console.error('cancel failed', e); }
  };

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
      logWrap.style.display = 'block';
      logBox.textContent = msg + '\n' + logBox.textContent;
    }
  };
  // Reset the bar to a clean indeterminate state at the start of a new job.
  const resetBar = () => {
    lastPct = null;
    barFill.style.width = '0%'; barFill.classList.remove('indeterminate');
    barPct.textContent = ''; barMsg.textContent = 'Starting…';
  };

  // ── Bag media viewer: one large closable floating panel over the main
  // display, shared by Scan photos and LiDAR sweeps (the control panel only
  // hosts the Load buttons — inline images there were too small to read).
  const mvTitle = el('span', { style: 'font-weight:600' }, '');
  const mvImg = el('img', { style:
    'max-width:100%;max-height:calc(92vh - 110px);display:block;margin:0 auto;' +
    'object-fit:contain;background:#000;border-radius:4px' });
  const mvSlider = el('input', { type: 'range', min: '0', max: '0', value: '0', style: 'flex:1' });
  const mvNum = el('span', { class: 'muted', style: 'min-width:86px;text-align:right' }, '–');
  let mvSource = null, mvTimer = null;   // {title, count, url(i)}
  const mvShow = () => {
    if (!mvSource) return;
    const i = Math.min(mvSource.count - 1, Math.max(0, parseInt(mvSlider.value) || 0));
    mvNum.textContent = `${i} / ${mvSource.count - 1}`;
    // Debounce while scrubbing so we don't request every intermediate frame.
    clearTimeout(mvTimer);
    mvTimer = setTimeout(() => { mvImg.src = mvSource.url(i); }, 120);
  };
  mvSlider.addEventListener('input', mvShow);
  const mvStep = (d) => { mvSlider.value = (parseInt(mvSlider.value) || 0) + d; mvShow(); };
  const mvPanel = el('div', { style:
    'position:fixed;left:16px;top:16px;z-index:2000;display:none;' +
    'width:min(74vw,1200px);background:rgba(18,20,26,0.95);border:1px solid #555;' +
    'border-radius:8px;padding:10px;box-shadow:0 6px 30px rgba(0,0,0,0.5)' },
    el('div', { class: 'row', style: 'align-items:center;margin-bottom:6px' },
      mvTitle,
      el('span', { style: 'flex:1' }),
      el('button', { class: 'act', style: 'width:34px', onclick: () => mvStep(-1) }, '◀'),
      el('button', { class: 'act', style: 'width:34px', onclick: () => mvStep(1) }, '▶'),
      el('button', { class: 'act', style: 'width:30px;margin-left:8px;background:#a33',
        onclick: () => { mvPanel.style.display = 'none'; } }, '✕')),
    mvImg,
    el('div', { class: 'row', style: 'margin-top:6px' }, mvSlider, mvNum));
  document.body.appendChild(mvPanel);
  const mvOpen = (source) => {
    mvSource = source;
    mvTitle.textContent = source.title;
    mvSlider.max = String(Math.max(0, source.count - 1));
    if (parseInt(mvSlider.value) >= source.count) mvSlider.value = '0';
    mvPanel.style.display = 'block';
    mvShow();
  };

  // ── Scan photos: raw camera JPEGs from the IMAGE bag (server rotates to
  // portrait + downscales for display). ──
  const photoStat = el('div', { class: 'muted' }, 'Uses the scan folder above.');
  const photoLoadBtn = el('button', { class: 'act', onclick: async () => {
    const scan = scanInput.value.trim();
    if (!scan) { photoStat.textContent = 'Enter a scan folder path above.'; return; }
    photoStat.textContent = 'Indexing bag…';
    try {
      const r = await api('/api/scan/photos', { path: scan });
      photoStat.textContent = `${r.count} photos (${r.topic})`;
      mvOpen({
        title: `Scan photos — ${r.count} frames`,
        count: r.count,
        url: (i) => `/api/scan/photo?path=${encodeURIComponent(scan)}&index=${i}&rot=ccw&width=1600`,
      });
    } catch (e) { photoStat.textContent = 'Error: ' + e.message; }
  } }, 'Load photos');
  const pPhotos = el('div', {}, photoStat, el('div', { class: 'row' }, photoLoadBtn));

  // ── LiDAR sweeps: per-sweep clouds server-rendered as top-down + side
  // height-coloured panels. ──
  const sweepStat = el('div', { class: 'muted' }, 'Uses the scan folder above.');
  const sweepLoadBtn = el('button', { class: 'act', onclick: async () => {
    const scan = scanInput.value.trim();
    if (!scan) { sweepStat.textContent = 'Enter a scan folder path above.'; return; }
    sweepStat.textContent = 'Indexing bag…';
    try {
      const r = await api('/api/scan/sweeps', { path: scan });
      sweepStat.textContent = `${r.count} sweeps (${r.topic})`;
      mvOpen({
        title: `LiDAR sweeps — ${r.count} sweeps`,
        count: r.count,
        url: (i) => `/api/scan/sweep?path=${encodeURIComponent(scan)}&index=${i}`,
      });
    } catch (e) { sweepStat.textContent = 'Error: ' + e.message; }
  } }, 'Load sweeps');
  const pSweeps = el('div', {}, sweepStat, el('div', { class: 'row' }, sweepLoadBtn));

  genBtn.onclick = async () => {
    const scan = scanInput.value.trim();
    if (!scan) { setBar(0, 'Enter a scan folder path'); return; }
    genBtn.disabled = true; logBox.textContent = ''; logWrap.style.display = 'block';
    logCopyBtn.style.display = 'none';   // copy icon appears only on error
    resetBar();
    const type = typeSel.value;
    // Remember the chosen seed so the finished splat can reload at the seed's
    // placement (captured now, since the dropdown may change during the run).
    const seedSelPath = type === 'splat' ? seedSel.value : '';
    const capM = parseFloat(capInput.value) || 6;
    const options = type === 'splat'
      ? { iterations: parseInt(iterInput.value) || 7000,
          downscale: parseInt(downInput.value) || 1,
          cap_max: Math.round(capM * 1e6),
          max_init_points: Math.round(Math.min(3, capM) * 1e6),
          drop_blurry: parseFloat(dropInput.value) || 0,
          undistort_scale: parseFloat(usInput.value) || 1.0,
          aniso_cap: parseFloat(anisoInput.value) || 20,
          ...(sfmInput.value.trim() ? { sfm_poses: sfmInput.value.trim() } : {}),
          ...(seedSel.value ? { pointcloud: seedSel.value } : {}) }
      : type === 'mesh'
      // Surface-mesh defaults land the champion recipe (outlier-clean + Taubin
      // smooth); a chosen seed pins which cloud it reconstructs from.
      ? { depth: 10, smooth: 15, density_quantile: 0.03,
          ...(seedSel.value ? { pointcloud: seedSel.value } : {}) }
      : { voxel_size: parseFloat(voxelInput.value), mono: monoCheck.checked,
          keep_self_view: !handleCheck.checked };
    try {
      const { job_id } = await api('/api/process/start',
        { type, project_path: projectInput.value.trim(), scan_path: scan, options });
      currentJobId = job_id; stopBtn.style.display = 'block';
      const es = new EventSource(`/api/process/events/${job_id}`);
      es.addEventListener('done', () => { es.close(); endRun(); refresh(); });
      es.onmessage = async (ev) => {
        const d = JSON.parse(ev.data);
        if (d.event === 'progress') setBar(d.percent, d.message);
        else if (d.event === 'log') setBar(undefined, d.message);
        else if (d.event === 'error') {
          const m = String(d.message || '').replace(/^\s*ERROR:\s*/i, '');
          setBar(0, 'ERROR: ' + m); logCopyBtn.style.display = 'block';
          es.close(); endRun();
        }
        else if (d.event === 'cancelled') { setBar(0, d.message); es.close(); endRun(); }
        else if (d.event === 'result') {
          setBar(100, `Done: ${d.filename} — loading…`);
          // Reload the new splat at the seed cloud's position/rotation so it
          // lands where the seed sits (rather than springing to the origin).
          try {
            const pose = await seedPose(seedSelPath);
            await loadPlyFromServer(d.path, d.filename, pose);
          } catch (e) { console.error(e); }
        }
      };
      es.onerror = () => { es.close(); endRun(); };
    } catch (err) { setBar(0, 'ERROR: ' + err.message); logCopyBtn.style.display = 'block'; endRun(); }
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
    let srcPath;
    try { srcPath = await ensureObjPath(entry, projectInput.value); }
    catch (e) { editStat.textContent = 'Delete: ' + e.message; return; }
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
    let srcPath;
    try { srcPath = await ensureObjPath(entry, projectInput.value); }
    catch (e) { eraseStat.textContent = 'Erase: ' + e.message; return; }
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
    el('div', { class: 'row' }, projectInput, projBrowse),
    el('button', { class: 'act', onclick: refresh }, 'Refresh outputs'),
    outList,
    el('h4', {}, 'Generate'),
    el('div', { class: 'muted' }, 'Scan folder (raw bags):'),
    el('div', { class: 'row' }, scanInput, scanBrowse),
    el('div', { class: 'row', style: 'align-items:center' }, typeSel,
      infoIcon('Output kind. splat = photoreal GPU-trained Gaussians (uses the quality '
        + 'slider below); pointcloud = raw coloured LiDAR points.', 'Output type')),
    pSeed,
    pTrained,
    pVoxel,
    pMono,
    pHandle,
    genBtn, stopBtn, barLabel, barWrap, logWrap,
    el('h4', {}, 'Scan photos'),
    pPhotos,
    el('h4', {}, 'LiDAR sweeps'),
    pSweeps,
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
      type: typeSel.value, voxel: voxelInput.value,
      quality: qSlider.value, iterations: iterInput.value, downscale: downInput.value,
      cap: capInput.value, dropBlurry: dropInput.value, undistortScale: usInput.value,
      anisoCap: anisoInput.value, sfm: sfmInput.value,
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
    setVal(typeSel, 'type'); setVal(voxelInput, 'voxel');
    // Restore saved knobs verbatim (don't re-derive from the quality preset —
    // the user may have hand-tuned individual controls before saving).
    setVal(qSlider, 'quality'); setVal(iterInput, 'iterations'); setVal(downInput, 'downscale');
    setVal(capInput, 'cap'); setVal(dropInput, 'dropBlurry'); setVal(usInput, 'undistortScale');
    setVal(anisoInput, 'anisoCap'); setVal(sfmInput, 'sfm');
    if (s.quality != null) qName.textContent = (QUALITY[parseInt(s.quality) - 1] || {}).name || '';
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

  showStartupHealth();   // one-time hardware traffic-light panel

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
