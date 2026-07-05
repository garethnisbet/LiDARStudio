// ============================================================
// js/stl.js — STL/OBJ/PLY/GLB/splat import, file-based scene save/load,
//             primitive creation, object list UI,
//             selection, transform mode
// ============================================================
import * as THREE from 'three';
import { STLLoader }  from 'three/addons/loaders/STLLoader.js';
import { OBJLoader }  from 'three/addons/loaders/OBJLoader.js';
import { MTLLoader }  from 'three/addons/loaders/MTLLoader.js';
import { PLYLoader }  from 'three/addons/loaders/PLYLoader.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { MeshBVH } from 'three-mesh-bvh';
import { DropInViewer, SceneFormat } from 'gaussian-splats-3d';

import * as State from './state.js';

// ============================================================
// Loaders & colour palette
// ============================================================
const stlLoader         = new STLLoader();
const objLoader         = new OBJLoader();
const plyLoader         = new PLYLoader();
const gltfImportLoader  = new GLTFLoader();

export const stlColors = [0x44aaff, 0xff6644, 0x44ff88, 0xffaa22, 0xcc44ff, 0xff4488, 0x22ddcc, 0xaadd22];

// Helper: consume the next color slot and return the color
function nextColor() {
  const idx = State.stlColorIdx;
  State.setStlColorIdx(idx + 1);
  return stlColors[idx % stlColors.length];
}

// ============================================================
// Scene state save/load (file-based)
// ============================================================

function _arrayBufferToBase64(buf) {
  const bytes = new Uint8Array(buf);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function _base64ToArrayBuffer(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

function _buildCameraPayload() {
  const cam = State.camera;
  const ctrl = State.orbitControls;
  return {
    position: [cam.position.x, cam.position.y, cam.position.z],
    target: [ctrl.target.x, ctrl.target.y, ctrl.target.z],
  };
}

function _buildSTLPayload(entry, bufferFn, includeSplatBuffers) {
  const m = entry.mesh;
  const rec = {
    id: entry.stlId,
    name: entry.name,
    color: entry.color,
    opacity: entry.opacity,
    fileType: entry.fileType || 'stl',
    isPointCloud: entry.isPointCloud || false,
    isSplat: entry.isSplat || false,
    position: [m.position.x, m.position.y, m.position.z],
    rotation: [m.rotation.x, m.rotation.y, m.rotation.z],
    scale: [m.scale.x, m.scale.y, m.scale.z],
    visible: m.visible,
  };
  // bufferFn === null builds a metadata-only record (buffers saved separately).
  if (bufferFn && (!entry.isSplat || includeSplatBuffers)) rec.buffer = bufferFn(entry._buffer);
  if (entry.isSplat && entry._fileName) rec.splatFile = entry._fileName;
  if (entry.isSplat) rec.splatTint = entry._splatTint ?? 0xffffff;
  return rec;
}

export function buildScenePayload() {
  const stls = State.importedSTLs.map(entry => _buildSTLPayload(entry, _arrayBufferToBase64, false));
  return { version: 1, stls, camera: _buildCameraPayload(), floorSize: State.floorSize };
}

// DB variant — stores raw ArrayBuffers (no base64), used by IndexedDB auto-save.
export function buildScenePayloadForDB() {
  const stls = State.importedSTLs.map(entry => _buildSTLPayload(entry, buf => buf, true));
  return { version: 1, stls, camera: _buildCameraPayload(), floorSize: State.floorSize };
}

// Metadata-only DB payload — omits the heavy mesh/point-cloud/splat buffers,
// which are saved separately (keyed by stl id) and merged back on restore.
// Cheap enough to clone into IndexedDB on every auto-save tick.
export function buildSceneMetadataForDB() {
  const stls = State.importedSTLs.map(entry => _buildSTLPayload(entry, null, false));
  return { version: 1, stls, camera: _buildCameraPayload(), floorSize: State.floorSize };
}

// The heavy buffers only, keyed by stl id. Re-written only when the buffer set
// actually changes (see sceneBufferSignature).
export function buildSceneBuffersForDB() {
  const buffers = State.importedSTLs
    .filter(e => e._buffer)
    .map(e => ({ id: e.stlId, buffer: e._buffer }));
  return { version: 1, buffers };
}

// Cheap fingerprint of the heavy buffer set (ids + byte lengths). While this is
// unchanged between auto-saves, the buffers record is left untouched and only
// the lightweight metadata is re-written — avoiding the multi-MB structured
// clone that otherwise hitched the frame every 30 s.
export function sceneBufferSignature() {
  return State.importedSTLs
    .map(e => e.stlId + ':' + (e._buffer ? e._buffer.byteLength : 0))
    .join(',');
}

export async function exportSceneState() {
  const payload = buildScenePayload();
  console.log('[Save Scene]', payload.stls.length, 'objects');
  for (const s of payload.stls) console.log('  object:', s.name, 'pos:', s.position, 'rot:', s.rotation);

  const json = JSON.stringify(payload, null, 2);
  const blob = new Blob([json], { type: 'application/json' });

  if (window.showSaveFilePicker) {
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName: 'scene_state.json',
        types: [{
          description: 'Scene JSON',
          accept: { 'application/json': ['.json'] },
        }],
      });
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      return;
    } catch (e) {
      if (e.name === 'AbortError') return; // user cancelled
    }
  }
  // Fallback for browsers without File System Access API
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'scene_state.json';
  a.click();
  URL.revokeObjectURL(url);
}

export async function importSceneState(file) {
  const text = await file.text();
  const data = JSON.parse(text);
  if (!data.version || !data.stls) throw new Error('Invalid scene file');
  return data;
}

async function _promptForSplatFiles(expectedNames) {
  const extensions = [...new Set(expectedNames.map(n => '.' + n.split('.').pop().toLowerCase()))];
  const map = new Map();

  if (window.showOpenFilePicker) {
    try {
      const handles = await window.showOpenFilePicker({
        multiple: true,
        types: [{
          description: 'Gaussian Splat files',
          accept: { 'application/octet-stream': extensions },
        }],
      });
      for (const handle of handles) {
        const file = await handle.getFile();
        map.set(file.name, file);
        map.set(file.name.toLowerCase(), file);
      }
    } catch { /* user cancelled */ }
  } else {
    await new Promise((resolve) => {
      const input = document.createElement('input');
      input.type = 'file';
      input.multiple = true;
      input.accept = extensions.join(',');
      input.style.display = 'none';
      document.body.appendChild(input);
      let done = false;
      const cleanup = () => {
        if (done) return;
        done = true;
        if (input.parentNode) input.parentNode.removeChild(input);
        resolve();
      };
      input.addEventListener('change', () => {
        for (const file of input.files) {
          map.set(file.name, file);
          map.set(file.name.toLowerCase(), file);
        }
        cleanup();
      });
      window.addEventListener('focus', function onFocus() {
        setTimeout(() => {
          if (!done && input.files.length === 0) cleanup();
          window.removeEventListener('focus', onFocus);
        }, 500);
      });
      input.click();
    });
  }

  return map;
}

export async function restoreSTLsFromState(records) {
  console.log('[Load Scene v3] Restoring', records.length, 'objects — two-phase restore');

  // ── Pre-phase: resolve missing splat files by prompting the user ──
  const missingSplats = records.filter(r => r.isSplat && !r.buffer && r.splatFile);
  if (missingSplats.length > 0) {
    const names = missingSplats.map(r => r.splatFile);
    console.log('[Load Scene] Need splat files:', names.join(', '));
    const loadingEl = document.getElementById('loading');
    if (loadingEl) {
      loadingEl.style.display = 'block';
      loadingEl.textContent = 'Select splat file' + (names.length > 1 ? 's' : '') + ': ' + names.join(', ');
    }
    const fileMap = await _promptForSplatFiles(names);
    for (const rec of missingSplats) {
      const file = fileMap.get(rec.splatFile) || fileMap.get(rec.splatFile.toLowerCase());
      if (file) {
        rec.buffer = await file.arrayBuffer();
        console.log('[Load Scene] Resolved splat file:', rec.splatFile);
      }
    }
    if (loadingEl) loadingEl.textContent = 'Loading scene...';
  }

  // ── Phase 1: create meshes WITHOUT transforms (default positions) ──
  const created = [];
  for (const rec of records) {
    if (rec.isSplat && !rec.buffer) {
      console.log('[Load Scene] Skipping splat (no buffer):', rec.name, '— re-import the file to restore');
      created.push({ rec, entry: null });
      continue;
    }
    // Buffer may be a raw ArrayBuffer (IndexedDB) or a base64 string (file export)
    const buffer = rec.buffer instanceof ArrayBuffer ? rec.buffer : _base64ToArrayBuffer(rec.buffer);
    let entry = null;
    const fileType = rec.fileType || 'stl';
    if (fileType === 'stl') {
      entry = createSTLFromBuffer(buffer, rec.name, rec.color, rec.id, null);
    } else if (fileType === 'ply' && (rec.isSplat || _isPLYGaussianSplat(buffer))) {
      entry = _addSplatToScene(buffer, 'ply', rec.name, rec.color, rec.id, null, rec.splatFile);
    } else if (fileType === 'ply') {
      const geometry = plyLoader.parse(buffer);
      if (rec.isPointCloud || _isPLYPointCloud(buffer)) {
        entry = _addPointsToScene(geometry, buffer, rec.name, rec.color, rec.id, null);
      } else {
        geometry.computeVertexNormals();
        entry = _addMeshToScene(geometry, buffer, 'ply', rec.name, rec.color, rec.id, null);
      }
    } else if (fileType === 'obj') {
      const objText = new TextDecoder().decode(buffer);
      const group = objLoader.parse(objText);
      const geometry = _mergeObject3D(group);
      entry = _addMeshToScene(geometry, buffer, 'obj', rec.name, rec.color, rec.id, null);
    } else if (fileType === 'glb') {
      try {
        const gltf = await new Promise((resolve, reject) =>
          gltfImportLoader.parse(buffer, '', resolve, reject));
        const geometry = _mergeObject3D(gltf.scene);
        entry = _addMeshToScene(geometry, buffer, 'glb', rec.name, rec.color, rec.id, null);
      } catch (e) {
        console.warn('Failed to restore GLB mesh:', rec.name, e);
      }
    } else if (rec.isSplat || (_splatFormatMap[fileType] && fileType !== 'ply')) {
      entry = _addSplatToScene(buffer, fileType, rec.name, rec.color, rec.id, null, rec.splatFile);
    }
    created.push({ rec, entry });
  }

  // ── Phase 2: apply transforms and parent links explicitly ──
  console.log('[Load Scene v3] Phase 2: applying transforms & parents');
  for (const { rec, entry } of created) {
    if (!entry) continue;
    const m = entry.mesh;

    // Apply saved transforms
    if (rec.position) m.position.set(rec.position[0], rec.position[1], rec.position[2]);
    if (rec.rotation) m.rotation.set(rec.rotation[0], rec.rotation[1], rec.rotation[2]);
    if (rec.scale)    m.scale.set(rec.scale[0], rec.scale[1], rec.scale[2]);
    if (rec.visible !== undefined) m.visible = rec.visible;
    if (rec.opacity !== undefined && !entry.isSplat) {
      m.material.opacity = rec.opacity;
      entry.opacity = rec.opacity;
    }
    // Splats apply opacity/tint through shader uniforms (no material.opacity),
    // so restore them onto the entry; updateSplatClip() pushes them each frame.
    if (entry.isSplat) {
      if (rec.opacity !== undefined) entry.opacity = rec.opacity;
      if (rec.splatTint !== undefined) entry._splatTint = rec.splatTint;
    }

    console.log('[Restore]', rec.name,
      'pos:', [m.position.x.toFixed(4), m.position.y.toFixed(4), m.position.z.toFixed(4)],
      'rot:', [m.rotation.x.toFixed(4), m.rotation.y.toFixed(4), m.rotation.z.toFixed(4)],
      'scale:', [m.scale.x.toFixed(4), m.scale.y.toFixed(4), m.scale.z.toFixed(4)]);
  }
}

// ============================================================
// Internal geometry helpers
// ============================================================
export function _mergeObject3D(object3D) {
  const chunks = [];
  const colorChunks = [];
  let totalVerts = 0;
  let hasNonWhite = false;
  object3D.updateWorldMatrix(true, true);
  object3D.traverse(child => {
    if (!child.isMesh || !child.geometry) return;
    let geo = child.geometry.clone();
    child.updateWorldMatrix(true, false);
    geo.applyMatrix4(child.matrixWorld);
    if (geo.index) geo = geo.toNonIndexed();
    const pos = geo.getAttribute('position');
    if (!pos || pos.count === 0) return;
    const arr = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
      arr[i * 3]     = pos.getX(i);
      arr[i * 3 + 1] = pos.getY(i);
      arr[i * 3 + 2] = pos.getZ(i);
    }
    chunks.push(arr);

    const colors = new Float32Array(pos.count * 3);
    const existingColors = geo.getAttribute('color');
    if (existingColors) {
      for (let i = 0; i < pos.count; i++) {
        colors[i * 3]     = existingColors.getX(i);
        colors[i * 3 + 1] = existingColors.getY(i);
        colors[i * 3 + 2] = existingColors.getZ(i);
      }
      hasNonWhite = true;
    } else {
      const mat = child.material;
      const c = (mat && mat.color) ? mat.color : new THREE.Color(1, 1, 1);
      if (c.r < 0.99 || c.g < 0.99 || c.b < 0.99) hasNonWhite = true;
      for (let i = 0; i < pos.count; i++) {
        colors[i * 3]     = c.r;
        colors[i * 3 + 1] = c.g;
        colors[i * 3 + 2] = c.b;
      }
    }
    colorChunks.push(colors);

    totalVerts += pos.count;
  });
  const merged = new THREE.BufferGeometry();
  if (totalVerts > 0) {
    const positions = new Float32Array(totalVerts * 3);
    let offset = 0;
    for (const chunk of chunks) {
      positions.set(chunk, offset);
      offset += chunk.length;
    }
    merged.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    if (hasNonWhite) {
      const colorsArr = new Float32Array(totalVerts * 3);
      let cOffset = 0;
      for (const cc of colorChunks) {
        colorsArr.set(cc, cOffset);
        cOffset += cc.length;
      }
      merged.setAttribute('color', new THREE.BufferAttribute(colorsArr, 3));
    }
  }
  merged.computeVertexNormals();
  return merged;
}

export function _isPLYPointCloud(buffer) {
  const header = new TextDecoder().decode(new Uint8Array(buffer, 0, Math.min(4096, buffer.byteLength)));
  const faceMatch = header.match(/element\s+face\s+(\d+)/);
  if (!faceMatch) return true;
  return parseInt(faceMatch[1], 10) === 0;
}

export function _isPLYGaussianSplat(buffer) {
  const header = new TextDecoder().decode(new Uint8Array(buffer, 0, Math.min(4096, buffer.byteLength)));
  return header.includes('f_dc_0') || header.includes('rot_0') || header.includes('scale_0');
}

export function _addPointsToScene(geometry, buffer, name, color, stlId, transforms) {
  const hasVertexColors = geometry.hasAttribute('color');
  const matColor = hasVertexColors ? 0xffffff : color;
  const material = new THREE.PointsMaterial({
    color: matColor, size: 0.003, sizeAttenuation: true,
    vertexColors: hasVertexColors,
    transparent: true, opacity: 0.9,
  });
  // Per-point flag for the live eraser (0 = visible, 1 = erased). The patched
  // material discards erased points; lidar.js sets these as a primitive sweeps.
  if (!geometry.getAttribute('aErased')) {
    geometry.setAttribute('aErased',
      new THREE.BufferAttribute(new Float32Array(geometry.getAttribute('position').count), 1));
  }
  _applyClipToPointsMaterial(material);
  const points = new THREE.Points(geometry, material);

  if (transforms) {
    points.position.set(...transforms.position);
    points.rotation.set(...transforms.rotation);
    points.scale.set(...transforms.scale);
    points.visible = transforms.visible;
  }

  State.scene.add(points);
  const box = new THREE.Box3().setFromObject(points);
  const div = document.createElement('div');
  div.className = 'mesh-label';
  div.textContent = name;
  const label = new CSS2DObject(div);
  label.visible = State.labelsOn;
  const center = box.getCenter(new THREE.Vector3());
  points.worldToLocal(center);
  label.position.copy(center);
  points.add(label);

  const entry = { mesh: points, label, name, color: matColor, opacity: material.opacity, stlId, _buffer: buffer, fileType: 'ply', isPointCloud: true, parentLink: null, importScale: points.scale.clone() };
  State.importedSTLs.push(entry);
  State.setStlColorIdx(Math.max(State.stlColorIdx, stlColors.indexOf(color) + 1));
  addSTLListItem(entry);

  State.requestRender();
  return entry;
}

export function _addMeshToScene(geometry, buffer, fileType, name, color, stlId, transforms) {
  geometry.boundsTree = new MeshBVH(geometry);

  const hasVertexColors = geometry.hasAttribute('color');
  const material = new THREE.MeshStandardMaterial({
    color: hasVertexColors ? 0xffffff : color,
    metalness: 0.3, roughness: 0.6,
    transparent: true, opacity: 0.85,
    vertexColors: hasVertexColors,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;

  if (transforms) {
    mesh.position.set(...transforms.position);
    mesh.rotation.set(...transforms.rotation);
    mesh.scale.set(...transforms.scale);
    mesh.visible = transforms.visible;
  } else {
    const box = new THREE.Box3().setFromObject(mesh);
    const size = box.getSize(new THREE.Vector3());
    if (size.length() > 1) mesh.scale.setScalar(0.001);
  }

  State.scene.add(mesh);
  const box = new THREE.Box3().setFromObject(mesh);
  const div = document.createElement('div');
  div.className = 'mesh-label';
  div.textContent = name;
  const label = new CSS2DObject(div);
  label.visible = State.labelsOn;
  const center = box.getCenter(new THREE.Vector3());
  mesh.worldToLocal(center);
  label.position.copy(center);
  mesh.add(label);

  const entry = { mesh, label, name, color, opacity: material.opacity, stlId, _buffer: buffer, fileType, parentLink: null, importScale: mesh.scale.clone() };
  State.importedSTLs.push(entry);
  State.setStlColorIdx(Math.max(State.stlColorIdx, stlColors.indexOf(color) + 1));
  addSTLListItem(entry);

  State.requestRender();
  return entry;
}

export function createSTLFromBuffer(buffer, name, color, stlId, transforms) {
  const geometry = stlLoader.parse(buffer);
  geometry.computeVertexNormals();
  return _addMeshToScene(geometry, buffer, 'stl', name, color, stlId, transforms);
}

// ============================================================
// File loaders (wired to input events in main.js)
// ============================================================
export function loadSTLFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const buffer = e.target.result;
    const baseName = file.name.replace(/\.stl$/i, '');
    const color = nextColor();
    const stlId = Date.now() + '_' + Math.random().toString(36).slice(2, 8);
    createSTLFromBuffer(buffer, baseName, color, stlId, null);
  };
  reader.readAsArrayBuffer(file);
}

export function loadOBJFile(file, mtlFile) {
  const readText = f => new Promise(resolve => {
    const r = new FileReader();
    r.onload = e => resolve(e.target.result);
    r.readAsText(f);
  });

  const doLoad = async () => {
    const text = await readText(file);
    const loader = new OBJLoader();

    if (mtlFile) {
      const mtlText = await readText(mtlFile);
      const mtlLoader = new MTLLoader();
      const materials = mtlLoader.parse(mtlText);
      materials.preload();
      loader.setMaterials(materials);
    }

    const group = loader.parse(text);
    const geometry = _mergeObject3D(group);
    const buffer = new TextEncoder().encode(text).buffer;
    const baseName = file.name.replace(/\.obj$/i, '');
    const color = nextColor();
    const stlId = Date.now() + '_' + Math.random().toString(36).slice(2, 8);
    _addMeshToScene(geometry, buffer, 'obj', baseName, color, stlId, null);
  };
  doLoad();
}

export function loadPLYFile(file, transforms = null, name = null) {
  // Resolves with the created scene entry so callers can re-select the result.
  // ``file`` may be a File/Blob or a raw ArrayBuffer (server-side library
  // loads pass the fetched buffer directly to avoid an extra multi-GB copy);
  // for an ArrayBuffer, ``name`` supplies the filename.
  return new Promise((resolve, reject) => {
    const process = (buffer, fname) => {
      try {
        const baseName = fname.replace(/\.ply$/i, '');
        const color = nextColor();
        const stlId = Date.now() + '_' + Math.random().toString(36).slice(2, 8);
        let entry;
        if (_isPLYGaussianSplat(buffer)) {
          entry = _addSplatToScene(buffer, 'ply', baseName, color, stlId, transforms, fname);
        } else if (_isPLYPointCloud(buffer)) {
          const geometry = plyLoader.parse(buffer);
          entry = _addPointsToScene(geometry, buffer, baseName, color, stlId, transforms);
        } else {
          const geometry = plyLoader.parse(buffer);
          geometry.computeVertexNormals();
          entry = _addMeshToScene(geometry, buffer, 'ply', baseName, color, stlId, transforms);
        }
        resolve(entry);
      } catch (err) { reject(err); }
    };
    if (file instanceof ArrayBuffer) {
      process(file, name || 'untitled.ply');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => process(e.target.result, file.name);
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(file);
  });
}

export async function loadGLBFile(file) {
  const buffer = await file.arrayBuffer();
  const baseName = file.name.replace(/\.(glb|gltf)$/i, '');
  const color = nextColor();
  const stlId = Date.now() + '_' + Math.random().toString(36).slice(2, 8);
  try {
    const gltf = await new Promise((resolve, reject) =>
      gltfImportLoader.parse(buffer, '', resolve, reject));
    const geometry = _mergeObject3D(gltf.scene);
    _addMeshToScene(geometry, buffer, 'glb', baseName, color, stlId, null);
  } catch (err) {
    console.error('GLB load error:', err);
    alert(`Failed to load GLB/GLTF file: ${err.message}`);
  }
}

// ============================================================
// Gaussian Splat loading (.splat, .ksplat, .spz)
// ============================================================
const _splatFormatMap = {
  'splat':  SceneFormat.Splat,
  'ksplat': SceneFormat.KSplat,
  'spz':    SceneFormat.Spz,
  'ply':    SceneFormat.Ply,
};

// ── Visibility clip box (non-destructive) ──────────────────
// An *oriented* box (so it can be translated, rotated and scaled) that hides
// geometry either inside or outside it, letting internal structure of a
// cloud/splat be inspected while editing — without deleting anything. The box
// is a unit cube ([-0.5,0.5]³) placed by a world matrix; the inverse of that
// matrix maps any world point into box-local space, where the test is just
// |xyz| ≤ 0.5. Point-cloud materials share these uniforms so every cloud
// updates live; splats are pushed per-frame from updateSplatClip() (their raw
// shader has no modelMatrix, so the matrix is composed per splat).
const _clipUniforms = {
  uClipEnabled: { value: false },
  uClipInv:     { value: new THREE.Matrix4() },  // world → box-local
  uClipMode:    { value: 1.0 },                  // 1 = show inside, 0 = show outside
};
let _clipEnabled = false, _clipMode = 1.0;
const _clipInv = new THREE.Matrix4();
const _clipSplatMat = new THREE.Matrix4();

// Update the active clip box. opts: { enabled, mode:'inside'|'outside',
// matrix:number[16] } — matrix is the unit-cube box's world matrix (column
// major, e.g. box.matrixWorld.elements). Call with { enabled:false } to clear.
export function setVisibilityClip(opts = {}) {
  if ('enabled' in opts) _clipEnabled = !!opts.enabled;
  if (opts.mode) _clipMode = opts.mode === 'outside' ? 0.0 : 1.0;
  if (opts.matrix) _clipInv.fromArray(opts.matrix).invert();
  _clipUniforms.uClipEnabled.value = _clipEnabled;
  _clipUniforms.uClipMode.value = _clipMode;
  _clipUniforms.uClipInv.value.copy(_clipInv);
  State.requestRender && State.requestRender();
}

// Inject the box test into a freshly-compiled PointsMaterial. Shares the
// module uniform objects so setVisibilityClip() drives every cloud at once.
function _applyClipToPointsMaterial(material) {
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uClipEnabled = _clipUniforms.uClipEnabled;
    shader.uniforms.uClipInv     = _clipUniforms.uClipInv;
    shader.uniforms.uClipMode    = _clipUniforms.uClipMode;
    shader.vertexShader =
      'uniform bool uClipEnabled;\nuniform mat4 uClipInv;\nuniform float uClipMode;\nattribute float aErased;\n'
      + shader.vertexShader.replace(
          '#include <project_vertex>',
          '#include <project_vertex>\n'
          + 'if (aErased > 0.5) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); }\n'   // live eraser
          + 'else if (uClipEnabled) {\n'
          + '  vec3 _l = (uClipInv * modelMatrix * vec4(transformed, 1.0)).xyz;\n'
          + '  bool _in = all(lessThanEqual(abs(_l), vec3(0.5)));\n'
          + '  if ((uClipMode > 0.5) ? !_in : _in) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);\n'
          + '}');
  };
}

// Box clip for splats. splatCenter is in the splat mesh's model space, so the
// uniform matrix supplied per-frame is (worldToBox · splatModelMatrix), taking
// splatCenter straight into box-local space.
function _patchSplatBoxClip(material) {
  if (!material || material.userData._boxClipPatched) return;
  if (!material.vertexShader || !material.vertexShader.includes('uniform float orthoZoom;')) return;
  if (!material.vertexShader.includes('vec4 viewCenter = transformModelViewMatrix * vec4(splatCenter, 1.0);')) return;

  material.vertexShader = material.vertexShader
    .replace('uniform float orthoZoom;',
      'uniform float orthoZoom;\nuniform bool uBoxClipEnabled;\nuniform mat4 uBoxClipInv;\nuniform float uBoxClipMode;')
    .replace('vec4 viewCenter = transformModelViewMatrix * vec4(splatCenter, 1.0);',
      'if (uBoxClipEnabled) {\n'
      + '    vec3 _l = (uBoxClipInv * vec4(splatCenter, 1.0)).xyz;\n'
      + '    bool _in = all(lessThanEqual(abs(_l), vec3(0.5)));\n'
      + '    if ((uBoxClipMode > 0.5) ? !_in : _in) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }\n'
      + '}\n'
      + '            vec4 viewCenter = transformModelViewMatrix * vec4(splatCenter, 1.0);');

  material.uniforms.uBoxClipEnabled = { value: false };
  material.uniforms.uBoxClipInv = { value: new THREE.Matrix4() };
  material.uniforms.uBoxClipMode = { value: 1.0 };
  material.userData._boxClipPatched = true;
  material.needsUpdate = true;
}

// ── Foreground clip for splats (ortho cutaway) ──────────────
// The gaussian-splats-3d vertex shader already computes the eye-space
// splat centre (`viewCenter`). We inject a discard so any splat closer
// to the camera than `foregroundClipDist` (world metres) is dropped,
// letting the interior of a scan be seen in orthographic mode. Only the
// splat material is patched, so ordinary meshes are never clipped.
function _patchSplatClipMaterial(material) {
  if (!material || material.userData._fgClipPatched) return;
  if (!material.vertexShader || !material.vertexShader.includes('uniform float orthoZoom;')) return;

  material.vertexShader = material.vertexShader
    .replace(
      'uniform float orthoZoom;',
      'uniform float orthoZoom;\nuniform float foregroundClipDist;'
    )
    .replace(
      'vec4 viewCenter = transformModelViewMatrix * vec4(splatCenter, 1.0);',
      'vec4 viewCenter = transformModelViewMatrix * vec4(splatCenter, 1.0);\n' +
      '            if (foregroundClipDist > 0.0 && -viewCenter.z < foregroundClipDist) {\n' +
      '                gl_Position = vec4(0.0, 0.0, 2.0, 1.0);\n' +
      '                return;\n' +
      '            }'
    );

  material.uniforms.foregroundClipDist = { value: 0.0 };
  material.userData._fgClipPatched = true;
  material.needsUpdate = true;
}

// ── Colour tint + opacity for splats ────────────────────────
// The splat fragment shader outputs `gl_FragColor = vec4(color.rgb, opacity)`.
// We multiply the colour by a tint (white = no change) and scale the alpha by
// an opacity factor, so splats get the same colour/transparency controls as
// meshes and point clouds. Both known render-mode shader variants are handled.
function _patchSplatAppearanceMaterial(material) {
  if (!material || material.userData._appearancePatched) return;
  const fs = material.fragmentShader;
  if (!fs) return;

  const patched = fs
    .replace('gl_FragColor = vec4(color.rgb, opacity);',
             'gl_FragColor = vec4(color.rgb * splatTint, opacity * splatOpacity);')
    .replace('gl_FragColor = vec4(vColor.rgb, w);',
             'gl_FragColor = vec4(vColor.rgb * splatTint, w * splatOpacity);');
  if (patched === fs) return; // output line not found — leave material untouched

  const decl = 'uniform vec3 splatTint;\nuniform float splatOpacity;\n';
  material.fragmentShader = patched.includes('void main')
    ? patched.replace('void main', decl + 'void main')
    : decl + patched;

  material.uniforms.splatTint = material.uniforms.splatTint || { value: new THREE.Color(1, 1, 1) };
  material.uniforms.splatOpacity = material.uniforms.splatOpacity || { value: 1.0 };
  material.userData._appearancePatched = true;
  material.needsUpdate = true;
}

// Push the current clip distance into every loaded splat's material.
// Called once per drawn frame from the animate loop. The clip is by
// eye-space depth, so it works the same in perspective and orthographic
// views. The distance is derived from the orbit radius so the slider
// feels scale-independent: fraction 0.5 clips up to the orbit target
// (the front half).
export function updateSplatClip() {
  const splats = State.importedSTLs.filter(s => s.isSplat);

  // The clip slider is only relevant while a splat is loaded. Both adding
  // and removing a splat request a render, so toggling visibility here
  // (before the early-out) keeps it in sync for every code path.
  const row = document.getElementById('splatClipRow');
  if (row) row.style.display = splats.length > 0 ? 'flex' : 'none';

  if (splats.length === 0) return;

  let dist = 0;
  if (State.splatClipFraction > 0) {
    const radius = State.activeCamera.position.distanceTo(State.orbitControls.target);
    dist = State.splatClipFraction * 2 * radius;
  }

  for (const s of splats) {
    const mesh = s._splatViewer && s._splatViewer.splatMesh;
    const mat = mesh && mesh.material;
    // Before the splat scene finishes loading, SplatMesh carries a
    // placeholder MeshBasicMaterial (no `.uniforms`), so guard both the
    // material and its uniforms before touching them.
    if (!mat || !mat.uniforms) continue;
    _patchSplatClipMaterial(mat);
    _patchSplatAppearanceMaterial(mat);
    _patchSplatBoxClip(mat);
    if (mat.uniforms.uBoxClipEnabled) {
      if (_clipEnabled) {
        // splatCenter (splat-local) → world → box-local in one matrix.
        mesh.updateWorldMatrix(true, false);
        _clipSplatMat.multiplyMatrices(_clipInv, mesh.matrixWorld);
        mat.uniforms.uBoxClipInv.value.copy(_clipSplatMat);
        mat.uniforms.uBoxClipMode.value = _clipMode;
        mat.uniforms.uBoxClipEnabled.value = true;
      } else {
        mat.uniforms.uBoxClipEnabled.value = false;
      }
    }
    if (mat.uniforms.foregroundClipDist) {
      mat.uniforms.foregroundClipDist.value = dist;
    }
    if (mat.uniforms.splatTint) {
      mat.uniforms.splatTint.value.set(s._splatTint ?? 0xffffff);
    }
    if (mat.uniforms.splatOpacity) {
      mat.uniforms.splatOpacity.value = s.opacity ?? 1;
    }
  }
}

function _parseSplatPLYHeader(buffer) {
  if (!buffer || buffer.byteLength < 100) return null;
  const headerBytes = new Uint8Array(buffer, 0, Math.min(8192, buffer.byteLength));
  const header = new TextDecoder().decode(headerBytes);
  const vertexMatch = header.match(/element vertex (\d+)/);
  if (!vertexMatch) return null;
  const vertexCount = parseInt(vertexMatch[1], 10);

  const props = [];
  for (const line of header.split('\n')) {
    const m = line.match(/^property (\w+) (\w+)/);
    if (m) props.push({ type: m[1], name: m[2] });
  }
  const bytesPerVertex = props.reduce((s, p) => s + (p.type === 'double' ? 8 : 4), 0);
  const headerEnd = header.indexOf('end_header');
  if (headerEnd < 0) return null;
  const dataOffset = new TextEncoder().encode(header.substring(0, headerEnd) + 'end_header\n').byteLength;

  return { vertexCount, bytesPerVertex, dataOffset };
}

function _estimateSplatBounds(buffer, ext) {
  if (ext !== 'ply') return null;
  const info = _parseSplatPLYHeader(buffer);
  if (!info) return null;
  const { vertexCount, bytesPerVertex, dataOffset } = info;

  const step = Math.max(1, Math.floor(vertexCount / 500));
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  const view = new DataView(buffer);
  for (let i = 0; i < vertexCount; i += step) {
    const off = dataOffset + i * bytesPerVertex;
    if (off + 12 > buffer.byteLength) break;
    const x = view.getFloat32(off, true);
    const y = view.getFloat32(off + 4, true);
    const z = view.getFloat32(off + 8, true);
    if (!isFinite(x) || !isFinite(y) || !isFinite(z)) continue;
    if (x < min[0]) min[0] = x; if (x > max[0]) max[0] = x;
    if (y < min[1]) min[1] = y; if (y > max[1]) max[1] = y;
    if (z < min[2]) min[2] = z; if (z > max[2]) max[2] = z;
  }
  if (!isFinite(min[0])) return null;
  return {
    center: [(min[0]+max[0])/2, (min[1]+max[1])/2, (min[2]+max[2])/2],
    size: [max[0]-min[0], max[1]-min[1], max[2]-min[2]],
  };
}

function _extractSplatPointCloud(buffer) {
  const info = _parseSplatPLYHeader(buffer);
  if (!info) return null;
  const { vertexCount, bytesPerVertex, dataOffset } = info;

  const positions = new Float32Array(vertexCount * 3);
  const view = new DataView(buffer);
  for (let i = 0; i < vertexCount; i++) {
    const off = dataOffset + i * bytesPerVertex;
    if (off + 12 > buffer.byteLength) break;
    positions[i * 3]     = view.getFloat32(off, true);
    positions[i * 3 + 1] = view.getFloat32(off + 4, true);
    positions[i * 3 + 2] = view.getFloat32(off + 8, true);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.computeBoundingBox();
  const material = new THREE.PointsMaterial({ size: 0.001, visible: false });
  const points = new THREE.Points(geometry, material);
  points.visible = false;
  return points;
}

export function _addSplatToScene(buffer, ext, name, color, stlId, transforms, fileName) {
  const format = _splatFormatMap[ext] ?? SceneFormat.Splat;
  const blob = new Blob([buffer]);
  const blobUrl = URL.createObjectURL(blob);

  const bounds = _estimateSplatBounds(buffer, ext);

  const wrapper = new THREE.Group();
  wrapper.name = 'splat_' + name;

  // The per-frame depth sort dominates splat cost. sharedMemoryForWorkers lets
  // the sort worker run zero-copy via SharedArrayBuffer — but that needs the
  // page to be cross-origin isolated (COOP/COEP headers, set in server.py), so
  // detect it at runtime and fall back gracefully on a plain HTTP load.
  // gpuAcceleratedSort is left off: the library force-disables it in WebXR
  // anyway, and on desktop its transform-feedback path renders nothing on some
  // drivers — so it adds risk with no VR benefit.
  const isolated = (typeof window !== 'undefined' && window.crossOriginIsolated) || false;
  const viewer = new DropInViewer({
    gpuAcceleratedSort: false,
    sharedMemoryForWorkers: isolated,
  });
  wrapper.add(viewer);

  if (transforms) {
    wrapper.position.set(...transforms.position);
    wrapper.rotation.set(...transforms.rotation);
    wrapper.scale.set(...transforms.scale);
    wrapper.visible = transforms.visible;
  }

  State.scene.add(wrapper);

  const div = document.createElement('div');
  div.className = 'mesh-label';
  div.textContent = name;
  const label = new CSS2DObject(div);
  label.visible = State.labelsOn;
  label.position.set(0, 0.05, 0);
  wrapper.add(label);

  const collisionPoints = (ext === 'ply') ? _extractSplatPointCloud(buffer) : null;
  if (collisionPoints) wrapper.add(collisionPoints);

  const entry = {
    mesh: wrapper, label, name, color, opacity: 1, stlId, _buffer: buffer,
    fileType: ext, isSplat: true, isPointCloud: false, parentLink: null,
    importScale: wrapper.scale.clone(), _splatViewer: viewer, _blobUrl: blobUrl,
    _collisionPoints: collisionPoints,
    _fileName: fileName || null,
    // Tint (multiplied into the per-splat colour) and opacity are applied via
    // injected shader uniforms in updateSplatClip(). White = original colours.
    _splatTint: 0xffffff,
  };
  State.importedSTLs.push(entry);
  State.setStlColorIdx(Math.max(State.stlColorIdx, stlColors.indexOf(color) + 1));
  addSTLListItem(entry);

  // A splat viewer re-sorts by camera direction and loads progressively,
  // so keep drawing every frame while any splat is present.
  State.setContinuousRender('splat', true);

  viewer.addSplatScene(blobUrl, {
    format,
    splatAlphaRemovalThreshold: 5,
    showLoadingUI: false,
  }).then(() => {
    if (bounds && !transforms) {
      const sizeLen = Math.sqrt(bounds.size[0]**2 + bounds.size[1]**2 + bounds.size[2]**2);
      if (sizeLen > 100) {
        wrapper.scale.setScalar(0.001);
        entry.importScale.copy(wrapper.scale);
      }
      label.position.set(
        bounds.center[0] * wrapper.scale.x,
        bounds.center[2] * wrapper.scale.y,
        bounds.center[1] * wrapper.scale.z
      );
    }
    State.requestRender();
  }).catch((err) => {
    console.error('[Splat] Failed to load', name, err);
  });

  State.requestRender();
  return entry;
}

export function loadSplatFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const buffer = e.target.result;
    const ext = file.name.split('.').pop().toLowerCase();
    const baseName = file.name.replace(/\.(splat|ksplat|spz)$/i, '');
    const color = nextColor();
    const stlId = Date.now() + '_' + Math.random().toString(36).slice(2, 8);
    _addSplatToScene(buffer, ext, baseName, color, stlId, null, file.name);
  };
  reader.readAsArrayBuffer(file);
}

// ============================================================
// Primitive creation
// ============================================================
function geometryToSTLBuffer(geometry) {
  const pos = geometry.getAttribute('position');
  const idx = geometry.getIndex();
  const triCount = idx ? idx.count / 3 : pos.count / 3;
  const bufLen = 84 + triCount * 50;
  const buf = new ArrayBuffer(bufLen);
  const view = new DataView(buf);
  view.setUint32(80, triCount, true);
  let offset = 84;
  const vA = new THREE.Vector3(), vB = new THREE.Vector3(), vC = new THREE.Vector3();
  const cb = new THREE.Vector3(), ab = new THREE.Vector3();
  for (let i = 0; i < triCount; i++) {
    const a = idx ? idx.getX(i * 3)     : i * 3;
    const b = idx ? idx.getX(i * 3 + 1) : i * 3 + 1;
    const c = idx ? idx.getX(i * 3 + 2) : i * 3 + 2;
    vA.fromBufferAttribute(pos, a);
    vB.fromBufferAttribute(pos, b);
    vC.fromBufferAttribute(pos, c);
    cb.subVectors(vC, vB); ab.subVectors(vA, vB); cb.cross(ab).normalize();
    view.setFloat32(offset, cb.x, true); offset += 4;
    view.setFloat32(offset, cb.y, true); offset += 4;
    view.setFloat32(offset, cb.z, true); offset += 4;
    for (const v of [vA, vB, vC]) {
      view.setFloat32(offset, v.x, true); offset += 4;
      view.setFloat32(offset, v.y, true); offset += 4;
      view.setFloat32(offset, v.z, true); offset += 4;
    }
    view.setUint16(offset, 0, true); offset += 2;
  }
  return buf;
}

export function addPrimitive(type) {
  const size = 0.05;
  let geometry;
  let name;
  if (type === 'cube') {
    geometry = new THREE.BoxGeometry(size, size, size);
    name = 'Cube';
  } else if (type === 'sphere') {
    geometry = new THREE.SphereGeometry(size / 2, 24, 16);
    name = 'Sphere';
  } else {
    geometry = new THREE.CylinderGeometry(size / 2, size / 2, size, 24);
    name = 'Cylinder';
  }
  const nonIndexed = geometry.index ? geometry.toNonIndexed() : geometry;
  const buffer = geometryToSTLBuffer(nonIndexed);
  geometry.dispose();
  nonIndexed.dispose();

  const color = nextColor();
  const stlId = Date.now() + '_' + Math.random().toString(36).slice(2, 8);
  const entry = createSTLFromBuffer(buffer, name, color, stlId, null);
  entry.primType = type;   // 'cube' | 'sphere' | 'cylinder' — usable as an eraser volume
  return entry;
}

// ============================================================
// Duplicate an imported object
// ============================================================
export async function duplicateSTL(srcEntry) {
  const m = srcEntry.mesh;
  const stlId = Date.now() + '_' + Math.random().toString(36).slice(2, 8);
  const name = srcEntry.name + ' copy';
  const color = srcEntry.color;
  const transforms = {
    position: [m.position.x + 0.02, m.position.y, m.position.z],
    rotation: [m.rotation.x, m.rotation.y, m.rotation.z],
    scale: [m.scale.x, m.scale.y, m.scale.z],
    visible: m.visible,
    parentLink: srcEntry.parentLink || null,
  };

  let newEntry;
  const ft = srcEntry.fileType || 'stl';
  if (srcEntry.isSplat) {
    newEntry = _addSplatToScene(srcEntry._buffer, ft, name, color, stlId, transforms);
  } else if (ft === 'stl') {
    newEntry = createSTLFromBuffer(srcEntry._buffer, name, color, stlId, transforms);
  } else if (ft === 'ply' && srcEntry.isPointCloud) {
    const geometry = plyLoader.parse(srcEntry._buffer);
    newEntry = _addPointsToScene(geometry, srcEntry._buffer, name, color, stlId, transforms);
  } else if (ft === 'ply') {
    const geometry = plyLoader.parse(srcEntry._buffer);
    geometry.computeVertexNormals();
    newEntry = _addMeshToScene(geometry, srcEntry._buffer, 'ply', name, color, stlId, transforms);
  } else if (ft === 'obj') {
    const text = new TextDecoder().decode(srcEntry._buffer);
    const group = objLoader.parse(text);
    const geometry = _mergeObject3D(group);
    newEntry = _addMeshToScene(geometry, srcEntry._buffer, 'obj', name, color, stlId, transforms);
  } else if (ft === 'glb') {
    try {
      const gltf = await new Promise((resolve, reject) =>
        gltfImportLoader.parse(srcEntry._buffer, '', resolve, reject));
      const geometry = _mergeObject3D(gltf.scene);
      newEntry = _addMeshToScene(geometry, srcEntry._buffer, 'glb', name, color, stlId, transforms);
    } catch (e) {
      console.warn('Failed to duplicate GLB mesh:', name, e);
      return;
    }
  } else if (_splatFormatMap[ft] && ft !== 'ply') {
    newEntry = _addSplatToScene(srcEntry._buffer, ft, name, color, stlId, transforms);
  }
  // Copy opacity from source
  if (newEntry && !newEntry.isSplat) {
    newEntry.opacity = srcEntry.opacity;
    newEntry.mesh.material.opacity = srcEntry.opacity;
  }
}

// ============================================================
// STL list UI
// ============================================================
export function addSTLListItem(entry) {
  const list = document.getElementById('stl-list');
  const item = document.createElement('div');
  item.className = 'stl-item';

  const colorSwatch = document.createElement('input');
  colorSwatch.type = 'color';
  colorSwatch.className = 'stl-color';
  colorSwatch.value = '#' + new THREE.Color(entry.isSplat ? (entry._splatTint ?? 0xffffff) : entry.color).getHexString();
  colorSwatch.title = entry.isSplat ? 'Tint colour (white = original)' : 'Change color';
  if (entry.isSplat) {
    // Splats keep their own per-point colours; the swatch tints them (multiply).
    colorSwatch.addEventListener('input', () => {
      entry._splatTint = new THREE.Color(colorSwatch.value).getHex();
      State.requestRender();
    });
  } else {
    colorSwatch.addEventListener('input', () => {
      entry.mesh.material.color.set(colorSwatch.value);
      entry.color = entry.mesh.material.color.getHex();
    });
  }

  const alphaSlider = document.createElement('input');
  alphaSlider.type = 'range';
  alphaSlider.className = 'stl-alpha';
  alphaSlider.min = '0';
  alphaSlider.max = '100';
  alphaSlider.value = Math.round((entry.opacity ?? (entry.isSplat ? 1 : entry.mesh.material.opacity)) * 100);
  alphaSlider.title = 'Opacity';
  if (entry.isSplat) {
    // Opacity scales the splat alpha via the injected shader uniform.
    alphaSlider.addEventListener('input', () => {
      entry.opacity = parseInt(alphaSlider.value, 10) / 100;
      State.requestRender();
    });
  } else {
    alphaSlider.addEventListener('input', () => {
      const val = parseInt(alphaSlider.value, 10) / 100;
      entry.mesh.material.opacity = val;
      entry.opacity = val;
    });
  }

  const nameSpan = document.createElement('span');
  nameSpan.className = 'stl-name';
  nameSpan.textContent = entry.name;
  nameSpan.title = 'Click to select, double-click to rename';
  nameSpan.addEventListener('click', (e) => {
    e.stopPropagation();
    selectSTL(entry, item);
  });
  nameSpan.addEventListener('dblclick', () => {
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'val-input';
    input.style.width = '100px';
    input.value = entry.name;
    nameSpan.style.display = 'none';
    nameSpan.parentNode.insertBefore(input, nameSpan.nextSibling);
    input.focus();
    input.select();
    const finish = (apply) => {
      if (apply && input.value.trim()) {
        entry.name = input.value.trim();
        nameSpan.textContent = entry.name;
        entry.label.element.textContent = entry.name;
      }
      input.remove();
      nameSpan.style.display = '';
    };
    input.addEventListener('blur', () => finish(true));
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); finish(true); }
      if (e.key === 'Escape') finish(false);
    });
  });

  const visBtn = document.createElement('button');
  visBtn.className = 'stl-vis';
  visBtn.textContent = '\uD83D\uDC41';
  visBtn.title = 'Toggle visibility';
  visBtn.style.opacity = entry.mesh.visible ? 1 : 0.3;
  visBtn.addEventListener('click', () => {
    entry.mesh.visible = !entry.mesh.visible;
    visBtn.style.opacity = entry.mesh.visible ? 1 : 0.3;
  });

  let splatToggleBtn = null;
  if (entry.isSplat && entry._collisionPoints) {
    splatToggleBtn = document.createElement('button');
    splatToggleBtn.className = 'stl-vis';
    splatToggleBtn.textContent = '\u2b22';
    splatToggleBtn.title = 'Toggle splat / point cloud';
    splatToggleBtn.style.opacity = 1;
    splatToggleBtn.addEventListener('click', () => {
      const splatOn = entry._splatViewer.visible;
      entry._splatViewer.visible = !splatOn;
      entry._collisionPoints.visible = splatOn;
      entry._collisionPoints.material.visible = splatOn;
      splatToggleBtn.style.opacity = entry._splatViewer.visible ? 1 : 0.3;
    });
  }

  const rmBtn = document.createElement('button');
  rmBtn.className = 'stl-rm';
  rmBtn.textContent = '\u2715';
  rmBtn.title = 'Remove';
  rmBtn.addEventListener('click', () => {
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
    const si = State.importedSTLs.indexOf(entry);
    if (si >= 0) State.importedSTLs.splice(si, 1);
    item.remove();
    if (entry.isSplat && !State.importedSTLs.some(s => s.isSplat)) {
      State.setContinuousRender('splat', false);
    }
    State.requestRender();
  });

  const dupBtn = document.createElement('button');
  dupBtn.className = 'stl-vis';
  dupBtn.textContent = '\u2398';
  dupBtn.title = 'Duplicate';
  dupBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    duplicateSTL(entry);
  });

  item.appendChild(colorSwatch);
  item.appendChild(alphaSlider);
  item.appendChild(nameSpan);
  item.appendChild(dupBtn);
  if (splatToggleBtn) item.appendChild(splatToggleBtn);
  item.appendChild(visBtn);
  item.appendChild(rmBtn);
  list.appendChild(item);
}

// ============================================================
// STL Selection & Transform
// ============================================================
const stlModePanel = document.getElementById('stl-mode');
const stlSelName   = document.getElementById('stl-sel-name');

export function selectSTL(entry, listItem) {
  deselectSTL();
  State.setSelectedSTL(entry);
  State.setSelectedListItem(listItem || null);

  State.stlTransformControls.attach(entry.mesh);
  stlModePanel.style.display = 'block';
  stlSelName.textContent = entry.name;
  syncSTLNumericInputs(entry);

  if (listItem) listItem.classList.add('selected');
}

export function syncSTLNumericInputs(entry) {
  if (!entry) return;
  const m = entry.mesh;
  const fmt = v => +v.toFixed(2);
  const fmtScale = v => +v.toFixed(6);
  document.getElementById('stlPosX').value = fmt(m.position.x * 1000);
  document.getElementById('stlPosY').value = fmt(m.position.z * 1000);
  document.getElementById('stlPosZ').value = fmt(m.position.y * 1000);
  document.getElementById('stlRotX').value = fmt(m.rotation.x * (180 / Math.PI));
  document.getElementById('stlRotY').value = fmt(m.rotation.z * (180 / Math.PI));
  document.getElementById('stlRotZ').value = fmt(m.rotation.y * (180 / Math.PI));
  document.getElementById('stlScX').value = fmtScale(m.scale.x);
  document.getElementById('stlScY').value = fmtScale(m.scale.z);
  document.getElementById('stlScZ').value = fmtScale(m.scale.y);
}

export function deselectSTL() {
  if (State.selectedSTL) {
    State.stlTransformControls.detach();
    State.stlTransformControls.setSpace('world');
    if (State.selectedListItem) State.selectedListItem.classList.remove('selected');
    State.setSelectedSTL(null);
    State.setSelectedListItem(null);
    stlModePanel.style.display = 'none';
    stlSelName.textContent = '';
    document.getElementById('stlSpaceBtn').textContent = 'World';
    document.getElementById('stlSpaceBtn').classList.remove('active');
  }
}

export function setSTLTransformMode(mode) {
  State.stlTransformControls.setMode(mode);
  document.getElementById('stlModeT').classList.toggle('active', mode === 'translate');
  document.getElementById('stlModeR').classList.toggle('active', mode === 'rotate');
  document.getElementById('stlModeS').classList.toggle('active', mode === 'scale');
}
