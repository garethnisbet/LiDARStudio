// js/lidar.js — LiDAR workflow panel for LidarStudio.
// Drives the /api/* endpoints (cloud/splat generation + project outputs) added
// by the Python backend (lidar_jobs.py) and loads results straight into the
// three.js + GaussianSplats3D viewer via the existing PLY loader.

import { loadPLYFile, deselectSTL } from './stl.js';
import * as State from './state.js';
import * as THREE from 'three';
import { TransformControls } from 'three/addons/controls/TransformControls.js';

const DEFAULT_PROJECT = '/home/gareth';

// Maps an in-scene object name -> its server-side .ply path, so edits can be
// run on the full file in Python (objects loaded from disk only).
const objPaths = {};

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
async function loadPlyFromServer(path, name) {
  const url = `/api/scan/file?path=${encodeURIComponent(path)}`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`load failed: ${resp.status}`);
  const blob = await resp.blob();
  const fname = name || path.split('/').pop();
  const file = new File([blob], fname, { type: 'application/octet-stream' });
  objPaths[fname.replace(/\.ply$/i, '')] = path;   // entry.name is the basename
  loadPLYFile(file);
  State.requestRender && State.requestRender();
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
  #lidar-panel .item button{background:#2c7;border:0;color:#04210f;border-radius:4px;padding:3px 9px;
    font-weight:700;cursor:pointer;font-size:11px}
  #lidar-panel .muted{color:#7d8aa0;font-size:11px}
  #ls-bar-wrap{height:6px;background:#0f1620;border-radius:4px;overflow:hidden;margin-top:8px;display:none}
  #ls-bar{height:100%;width:0;background:#2a6df0;transition:width .2s}
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
      outList.replaceChildren(...(items.length ? items.map(f =>
        el('div', { class: 'item' },
          el('span', {}, `${f.kind === 'splat' ? '🟣' : '⚪'} ${f.name} `,
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
  const barWrap = el('div', { id: 'ls-bar-wrap' }, el('div', { id: 'ls-bar' }));
  const logBox = el('div', { id: 'ls-log' });
  const genBtn = el('button', { class: 'act' }, 'Generate');

  const setBar = (pct, msg) => {
    barWrap.style.display = 'block';
    barWrap.firstChild.style.width = `${pct || 0}%`;
    if (msg) { logBox.style.display = 'block'; logBox.textContent = msg + '\n' + logBox.textContent; }
  };

  genBtn.onclick = async () => {
    const scan = scanInput.value.trim();
    if (!scan) { setBar(0, 'Enter a scan folder path'); return; }
    genBtn.disabled = true; logBox.textContent = ''; logBox.style.display = 'block';
    const type = typeSel.value;
    const options = type === 'splat'
      ? { splat_mode: methodSel.value, splat_voxel: parseFloat(voxelInput.value), surfel_sor: parseFloat(sorInput.value) }
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
    el('option', { value: 'recolour' }, 'Recolour from scan photos'));
  const facInput = el('input', { type: 'number', min: '2', step: '1', value: '2' });
  const sorNb = el('input', { type: 'number', min: '4', step: '1', value: '20' });
  const sorStd = el('input', { type: 'number', min: '0.5', step: '0.25', value: '2' });
  const cMin = ['x', 'y', 'z'].map(() => el('input', { type: 'number', step: '0.1' }));
  const cMax = ['x', 'y', 'z'].map(() => el('input', { type: 'number', step: '0.1' }));
  const invCb = el('input', { type: 'checkbox' });
  const editStat = el('div', { class: 'muted' }, 'Select a loaded object to edit.');

  // ── 3D crop-box gizmo (dedicated TransformControls, isolated from selection) ──
  let cropGizmo = null, cropBox = null, cropTargetPath = null, cropTargetLi = null;
  function ensureGizmo() {
    if (cropGizmo) return cropGizmo;
    const g = new TransformControls(State.activeCamera || State.camera, State.renderer.domElement);
    g.setSize(0.7);
    g.addEventListener('dragging-changed', (e) => { State.orbitControls.enabled = !e.value; });
    g.addEventListener('change', () => State.requestRender());
    State.scene.add(g);
    cropGizmo = g;
    return g;
  }
  function removeCropBox(render = true) {
    if (cropGizmo && cropGizmo.object) cropGizmo.detach();
    if (cropBox) {
      State.scene.remove(cropBox);
      cropBox.traverse(o => { o.geometry?.dispose?.(); o.material?.dispose?.(); });
      cropBox = null;
    }
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
    deselectSTL();                       // free the shared gizmo / selection
    removeCropBox(false);
    const c = b.getCenter(new THREE.Vector3()), s = b.getSize(new THREE.Vector3());
    const geo = new THREE.BoxGeometry(1, 1, 1);
    cropBox = new THREE.Mesh(geo, new THREE.MeshBasicMaterial(
      { color: 0x33ff99, transparent: true, opacity: 0.12, depthWrite: false }));
    cropBox.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({ color: 0x33ff99 })));
    cropBox.position.copy(c);
    cropBox.scale.set(Math.max(s.x, 0.05), Math.max(s.y, 0.05), Math.max(s.z, 0.05));
    State.scene.add(cropBox);
    ensureGizmo().attach(cropBox);
    State.requestRender();
    editStat.textContent = 'Drag the green box (Move/Scale), then Apply edit.';
  }

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
      el('button', { class: 'act', style: 'flex:1', onclick: () => cropGizmo?.setMode('translate') }, 'Move box'),
      el('button', { class: 'act', style: 'flex:1', onclick: () => cropGizmo?.setMode('scale') }, 'Scale box')),
    el('div', { class: 'muted', style: 'margin-top:4px' }, 'or set bounds manually:'),
    el('div', { class: 'row' }, el('span', { class: 'muted', style: 'width:28px' }, 'min'), ...cMin),
    el('div', { class: 'row' }, el('span', { class: 'muted', style: 'width:28px' }, 'max'), ...cMax),
    el('label', { class: 'muted' }, invCb, ' keep outside (delete inside)'),
    fillBtn);
  const pRecol = el('div', { class: 'muted' },
    'Re-projects the photos from the Scan folder (Generate section) onto the '
    + 'selected cloud using its saved trajectory. Set that field first.');
  const showOp = () => {
    pDec.style.display = editOp.value === 'decimate' ? 'flex' : 'none';
    pSor.style.display = editOp.value === 'denoise_sor' ? 'flex' : 'none';
    pCrop.style.display = editOp.value === 'crop' ? 'block' : 'none';
    pRecol.style.display = editOp.value === 'recolour' ? 'block' : 'none';
  };
  editOp.addEventListener('change', showOp);
  const applyEditBtn = el('button', { class: 'act' }, 'Apply edit');

  async function runEdit(endpoint, body, li, doneMsg) {
    applyEditBtn.disabled = true;
    try {
      const r = await api(endpoint, body);
      editStat.textContent = doneMsg(r);
      await loadPlyFromServer(r.output, r.output.split('/').pop());
      if (li) li.querySelector('button[title="Remove"]')?.click();  // drop the pre-edit object
    } catch (e) { editStat.textContent = 'Error: ' + e.message; }
    applyEditBtn.disabled = false;
  }

  applyEditBtn.onclick = async () => {
    const op = editOp.value;

    // Crop via the 3D box uses the remembered target (placing the box deselects).
    if (op === 'crop' && cropBox) {
      cropBox.updateWorldMatrix(true, false);
      const b = new THREE.Box3().setFromObject(cropBox);
      const params = { min: [b.min.x, b.min.y, b.min.z], max: [b.max.x, b.max.y, b.max.z], invert: invCb.checked };
      editStat.textContent = 'Cropping to box…';
      const li = cropTargetLi;
      await runEdit('/api/edit/apply', { path: cropTargetPath, op: 'crop', params },
        li, r => `Kept ${r.kept.toLocaleString()} / ${r.total.toLocaleString()} — reloading`);
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
        r => `Coloured ${r.coloured.toLocaleString()} / ${r.total.toLocaleString()} — reloading`);
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
    editStat.textContent = `Applying ${op}…`;
    return runEdit('/api/edit/apply', { path: srcPath, op, params }, li,
      r => `Kept ${r.kept.toLocaleString()} / ${r.total.toLocaleString()} — reloading`);
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
      await loadPlyFromServer(orig, orig.split('/').pop());
      if (li) li.querySelector('button[title="Remove"]')?.click();
    } catch (e) { editStat.textContent = 'Original not found: ' + e.message; }
  } }, 'Revert to original');

  const panel = el('div', { id: 'lidar-panel' },
    el('h4', {}, 'Library'),
    projectInput,
    el('button', { class: 'act', onclick: refresh }, 'Refresh outputs'),
    outList,
    el('h4', {}, 'Generate'),
    el('div', { class: 'muted' }, 'Scan folder (raw bags):'), scanInput,
    el('div', { class: 'row' }, typeSel, methodSel),
    el('div', { class: 'row' },
      el('label', { class: 'muted', style: 'flex:1' }, 'voxel', voxelInput),
      el('label', { class: 'muted', style: 'flex:1' }, 'noise σ', sorInput)),
    genBtn, barWrap, logBox,
    el('h4', {}, 'Edit selected'),
    editStat, editOp, pDec, pSor, pCrop, pRecol, applyEditBtn, revertBtn);
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
  refresh();
}
