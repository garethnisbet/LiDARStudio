// ============================================================
// js/storage.js — IndexedDB-backed scene persistence
// Replaces localStorage for auto-save so large mesh buffers
// (which exceed the ~5 MB localStorage quota) are stored safely.
// IndexedDB stores ArrayBuffers natively via structured clone.
// ============================================================

const DB_NAME    = 'robotvis_db';
const DB_VERSION = 1;
const STORE      = 'scene';
const KEY        = 'autosave';
// Heavy mesh/point-cloud/splat ArrayBuffers are stored under their own key so
// the lightweight metadata record (camera, transforms, joint angles) can be
// re-written on every auto-save without re-cloning megabytes of geometry that
// never changed. See autoSaveScene in main.js.
export const BUFFERS_KEY = 'autosave_buffers';
const VR_ANCHOR_KEY = 'vr_anchor';

async function _openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = e => e.target.result.createObjectStore(STORE);
    req.onsuccess       = e => resolve(e.target.result);
    req.onerror         = e => reject(e.target.error);
  });
}

export async function dbSave(data, key = KEY) {
  const db = await _openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(data, key);
    tx.oncomplete = resolve;
    tx.onerror    = e => reject(e.target.error);
    tx.onabort    = e => reject(e.target.error);
  });
}

export async function dbLoad(key = KEY) {
  const db = await _openDB();
  return new Promise((resolve, reject) => {
    const tx  = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).get(key);
    req.onsuccess = e => resolve(e.target.result ?? null);
    req.onerror   = e => reject(e.target.error);
  });
}

export async function dbClear() {
  const db = await _openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).delete(KEY);
    tx.objectStore(STORE).delete(BUFFERS_KEY);
    tx.oncomplete = resolve;
    tx.onerror    = e => reject(e.target.error);
  });
}

export async function dbSaveVRAnchor(data) {
  const db = await _openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(data, VR_ANCHOR_KEY);
    tx.oncomplete = resolve;
    tx.onerror    = e => reject(e.target.error);
    tx.onabort    = e => reject(e.target.error);
  });
}

export async function dbLoadVRAnchor() {
  const db = await _openDB();
  return new Promise((resolve, reject) => {
    const tx  = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).get(VR_ANCHOR_KEY);
    req.onsuccess = e => resolve(e.target.result ?? null);
    req.onerror   = e => reject(e.target.error);
  });
}
