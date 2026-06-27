// ============================================================
// js/panel.js — makeSpanEditable, buildControlPanel,
//               rebuildParentDropdown, setActiveDevice,
//               rebuildDeviceList, rebuildAddDeviceDropdown,
//               removeDevice, findDeviceForObject
// ============================================================
import * as THREE from 'three';
import * as State from './state.js';
import {
  updateFK, getEEWorldPosition, getEEWorldQuaternion,
  eulerToKappa, getCompensation, clampJoints,
} from './kinematics.js';
import {
  updateSliders, setIKMode, syncIKSliders,
} from './device.js';
import { updateVirtualAngles } from './kinematics.js';
import { updateHexapodPose, computeLegLengthsFromPose, solveHexapodFK, syncHexapodSliders } from './hexapod.js';
import { resolveParentLink } from './stl.js';

const deg2rad = Math.PI / 180;
const rad2deg = 180 / Math.PI;

// LidarStudio is a point-cloud / splat editor, not a robot viewer: no bundled
// robot configs.  (Robot models can still be loaded ad-hoc via ?config=… if the
// JSON+GLB are added back.)
export const configFiles = [];

// ============================================================
// makeSpanEditable
// ============================================================
export function makeSpanEditable(spanId, onCommit, trigger = 'dblclick') {
  const span = document.getElementById(spanId);
  if (!span) return;
  span.addEventListener(trigger, () => {
    const input = document.createElement('input');
    input.type = 'number';
    input.className = 'val-input';
    input.value = parseFloat(span.textContent);
    span.style.display = 'none';
    span.parentNode.insertBefore(input, span.nextSibling);
    input.focus();
    input.select();
    const finish = (apply) => {
      if (apply) {
        const v = parseFloat(input.value);
        if (!isNaN(v)) onCommit(v);
      }
      input.remove();
      span.style.display = '';
    };
    input.addEventListener('blur', () => finish(true));
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); finish(true); }
      if (e.key === 'Escape') finish(false);
    });
  });
}

// ============================================================
// _buildSerialSliders — joint-angle sliders for serial manipulators
// ============================================================
function _buildSerialSliders(dev, container) {
  for (let si = 0; si < dev.sliderJointMap.length; si++) {
    const ji = dev.sliderJointMap[si];
    const j = dev.config.joints[ji];
    const displayName = dev.kappaSliderNames[ji] || j.name;
    const sign = dev.apiSign[ji];
    const deg = sign * dev.jointAngles[ji] * rad2deg;
    const lo = sign < 0 ? -j.limits[1] : j.limits[0];
    const hi = sign < 0 ? -j.limits[0] : j.limits[1];
    const div = document.createElement('div');
    div.className = 'slider-row';
    div.innerHTML = `<label>${displayName} <span id="v${si+1}">${deg.toFixed(2)}</span></label>` +
      `<input type="range" id="j${si+1}" min="${lo}" max="${hi}" value="${deg.toFixed(2)}" step="0.01">`;
    container.appendChild(div);
  }
}

// ============================================================
// _buildHexapodSliders — platform pose sliders for hexapod
// ============================================================
function _buildHexapodSliders(dev, container) {
  const lim = dev.config.limits;
  const pose = dev.platformPose;
  const axes = [
    { label: 'X (mm)',  id: 'hpx', min: lim.x[0],  max: lim.x[1],  val: pose[0], step: 0.1 },
    { label: 'Y (mm)',  id: 'hpy', min: lim.y[0],  max: lim.y[1],  val: pose[1], step: 0.1 },
    { label: 'Z (mm)',  id: 'hpz', min: lim.z[0],  max: lim.z[1],  val: pose[2], step: 0.1 },
    { label: 'Rx (deg)', id: 'hprx', min: lim.rx[0], max: lim.rx[1], val: pose[3], step: 0.01 },
    { label: 'Ry (deg)', id: 'hpry', min: lim.ry[0], max: lim.ry[1], val: pose[4], step: 0.01 },
    { label: 'Rz (deg)', id: 'hprz', min: lim.rz[0], max: lim.rz[1], val: pose[5], step: 0.01 },
  ];

  const transHeader = document.createElement('h2');
  transHeader.style.cssText = 'color:#5ad; margin:4px 0 2px; font-size:0.85em;';
  transHeader.textContent = 'Translation';
  container.appendChild(transHeader);

  for (let i = 0; i < 3; i++) {
    const a = axes[i];
    const div = document.createElement('div');
    div.className = 'slider-row';
    div.innerHTML = `<label>${a.label} <span id="${a.id}v">${a.val.toFixed(1)}</span></label>` +
      `<input type="range" id="${a.id}" min="${a.min}" max="${a.max}" value="${a.val}" step="${a.step}">`;
    container.appendChild(div);
  }

  const rotHeader = document.createElement('h2');
  rotHeader.style.cssText = 'color:#5ad; margin:8px 0 2px; font-size:0.85em;';
  rotHeader.textContent = 'Rotation';
  container.appendChild(rotHeader);

  for (let i = 3; i < 6; i++) {
    const a = axes[i];
    const div = document.createElement('div');
    div.className = 'slider-row';
    div.innerHTML = `<label>${a.label} <span id="${a.id}v">${a.val.toFixed(2)}</span></label>` +
      `<input type="range" id="${a.id}" min="${a.min}" max="${a.max}" value="${a.val}" step="${a.step}">`;
    container.appendChild(div);
  }

  const ids = ['hpx', 'hpy', 'hpz', 'hprx', 'hpry', 'hprz'];
  for (let i = 0; i < 6; i++) {
    const sid = ids[i];
    const idx = i;
    const isRot = i >= 3;
    document.getElementById(sid).addEventListener('input', (e) => {
      const v = parseFloat(e.target.value);
      document.getElementById(sid + 'v').textContent = isRot ? v.toFixed(2) : v.toFixed(1);
      dev.platformPose[idx] = v;
      updateHexapodPose(dev);
      _updateLegLengthDisplay(dev);
    });
    makeSpanEditable(sid + 'v', (val) => {
      const s = document.getElementById(sid);
      s.value = Math.max(parseFloat(s.min), Math.min(parseFloat(s.max), val));
      s.dispatchEvent(new Event('input'));
    });
  }

  // --- Leg Lengths (FK) ---
  const legHeader = document.createElement('h2');
  legHeader.style.cssText = 'color:#5ad; margin:8px 0 2px; font-size:0.85em;';
  legHeader.textContent = 'Leg Lengths (mm)';
  container.appendChild(legHeader);

  const currentLengths = computeLegLengthsFromPose(dev, dev.platformPose);
  const restLengths = computeLegLengthsFromPose(dev, [0, 0, 0, 0, 0, 0]);
  const legMins = [...restLengths];
  const legMaxs = [...restLengths];
  const limKeys = ['x', 'y', 'z', 'rx', 'ry', 'rz'];
  for (let axis = 0; axis < 6; axis++) {
    const [lo, hi] = lim[limKeys[axis]];
    for (const val of [lo, hi]) {
      const testPose = [0, 0, 0, 0, 0, 0];
      testPose[axis] = val;
      const tl = computeLegLengthsFromPose(dev, testPose);
      for (let k = 0; k < 6; k++) {
        legMins[k] = Math.min(legMins[k], tl[k]);
        legMaxs[k] = Math.max(legMaxs[k], tl[k]);
      }
    }
  }
  for (let k = 0; k < 6; k++) {
    const range = legMaxs[k] - legMins[k];
    legMins[k] -= range * 0.15;
    legMaxs[k] += range * 0.15;
  }

  for (let i = 0; i < 6; i++) {
    const mm = currentLengths[i] * 1000;
    const div = document.createElement('div');
    div.className = 'slider-row';
    div.innerHTML = `<label>L${i + 1} <span id="hleg${i + 1}v">${mm.toFixed(2)}</span></label>` +
      `<input type="range" id="hleg${i + 1}" min="${(legMins[i] * 1000).toFixed(2)}" max="${(legMaxs[i] * 1000).toFixed(2)}" value="${mm.toFixed(2)}" step="0.01">`;
    container.appendChild(div);
  }

  for (let i = 0; i < 6; i++) {
    const sid = `hleg${i + 1}`;
    document.getElementById(sid).addEventListener('input', (e) => {
      document.getElementById(sid + 'v').textContent = parseFloat(e.target.value).toFixed(2);
      const target = [];
      for (let k = 0; k < 6; k++)
        target.push(parseFloat(document.getElementById(`hleg${k + 1}`).value) / 1000);
      const newPose = solveHexapodFK(dev, target);
      for (let k = 0; k < 6; k++) dev.platformPose[k] = newPose[k];
      updateHexapodPose(dev);
      syncHexapodSliders(dev);
    });
    makeSpanEditable(sid + 'v', (val) => {
      const s = document.getElementById(sid);
      s.value = Math.max(parseFloat(s.min), Math.min(parseFloat(s.max), val));
      s.dispatchEvent(new Event('input'));
    });
  }
}

function _updateLegLengthDisplay(dev) {
  const lengths = computeLegLengthsFromPose(dev, dev.platformPose);
  for (let i = 0; i < 6; i++) {
    const mm = lengths[i] * 1000;
    const s = document.getElementById(`hleg${i + 1}`);
    const l = document.getElementById(`hleg${i + 1}v`);
    if (s) s.value = mm;
    if (l) l.textContent = mm.toFixed(2);
  }
}

// ============================================================
// buildControlPanel
// ============================================================
export function buildControlPanel(dev) {
  // Update panel title
  const isHexapod = dev.type === 'hexapod';
  document.getElementById('panel-title').textContent = isHexapod
    ? `${dev.name} - Platform Pose`
    : `${dev.name} - Joint Angles (deg)`;
  document.title = dev.name + ' - FK/IK Viewer';

  // Clear and rebuild sliders
  const sliderContainer = document.getElementById('joint-sliders');
  sliderContainer.innerHTML = '';

  if (isHexapod) {
    _buildHexapodSliders(dev, sliderContainer);
  } else {
    _buildSerialSliders(dev, sliderContainer);
  }

  if (!isHexapod) {
  // Bind slider events (serial manipulators only)
  for (let si = 1; si <= dev.sliderJointMap.length; si++) {
    const slider = document.getElementById(`j${si}`);
    const label  = document.getElementById(`v${si}`);
    const ji = dev.sliderJointMap[si - 1];
    slider.addEventListener('input', () => {
      const deg = parseFloat(slider.value);
      label.textContent = deg.toFixed(2);
      dev.jointAngles[ji] = dev.apiSign[ji] * deg * deg2rad;
      updateFK(dev);
      updateVirtualAngles(dev);
      if (dev.ikMode) {
        State.scene.updateMatrixWorld(true);
        dev.ikTarget.position.copy(getEEWorldPosition(dev));
        dev.ikTargetQuat.copy(getEEWorldQuaternion(dev));
        dev.ikTargetEuler.setFromQuaternion(dev.ikTargetQuat, 'YZX');
        dev.ikTarget.quaternion.copy(dev.ikTargetQuat);
        syncIKSliders(dev);
      }
    });
    makeSpanEditable(`v${si}`, (val) => {
      const s = document.getElementById(`j${si}`);
      s.value = Math.max(s.min, Math.min(s.max, val));
      s.dispatchEvent(new Event('input'));
    });
  }
  } // end serial slider binding

  // Build virtual angles section
  const virtContainer = document.getElementById('virtual-angles-container');
  virtContainer.innerHTML = '';
  if (dev.isKappaGeometry) {
    const kappaLimits = [dev.jointLimits[dev.kappaJointIdx][0] * rad2deg, dev.jointLimits[dev.kappaJointIdx][1] * rad2deg];
    const chiMin = dev._chiLimits ? dev._chiLimits[0] : Math.min(...kappaLimits);
    const chiMax = dev._chiLimits ? dev._chiLimits[1] : Math.max(...kappaLimits);
    const thetaLimits = dev.config.joints[dev.thetaJointIdx].limits;
    const phiLimits   = dev.config.joints[dev.phiJointIdx].limits;

    virtContainer.innerHTML =
      '<div style="margin-top:8px;padding-top:8px;border-top:1px solid #334;">' +
      '<h2 style="color:#da5;">Virtual Angles (deg)</h2>' +
      '<div class="slider-row"><label>\u03C7 (chi) <span id="vchi">0</span></label>' +
      `<input type="range" id="chiSlider" min="${chiMin.toFixed(2)}" max="${chiMax.toFixed(2)}" value="0" step="0.01">` +
      '</div>' +
      '<div class="slider-row"><label>\u03B8 (theta) <span id="vvtheta">0</span></label>' +
      `<input type="range" id="vthetaSlider" min="${thetaLimits[0]}" max="${thetaLimits[1]}" value="0" step="0.01">` +
      '</div>' +
      '<div class="slider-row"><label>\u03C6 (phi) <span id="vvphi">0</span></label>' +
      `<input type="range" id="vphiSlider" min="${phiLimits[0]}" max="${phiLimits[1]}" value="0" step="0.01">` +
      '</div>' +
      '<button id="kappaSignBtn" class="active">\u03BA Sign: +</button>' +
      '</div>';

    // Bind virtual angle events
    function applyVirtualAngles(vThetaDeg, chiDeg, vPhiDeg) {
      const result = eulerToKappa(dev, -chiDeg);
      if (!result) return;
      const comp = getCompensation(dev, result.kappa);
      const sign = dev.kappaSignPositive ? 1 : -1;
      dev.jointAngles[dev.kappaJointIdx] = sign * result.kappa * deg2rad;
      dev.jointAngles[dev.thetaJointIdx] = sign * dev.kappaThetaSign * (vThetaDeg - 90 + comp.theta) * deg2rad;
      dev.jointAngles[dev.phiJointIdx]   = sign * dev.kappaThetaSign * (vPhiDeg - 90 + comp.phi) * deg2rad;
      clampJoints(dev);
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
    }

    function readVirtualAngles() {
      return {
        theta: parseFloat(document.getElementById('vthetaSlider').value),
        chi:   parseFloat(document.getElementById('chiSlider').value),
        phi:   parseFloat(document.getElementById('vphiSlider').value),
      };
    }

    document.getElementById('chiSlider').addEventListener('input', (e) => {
      const v = readVirtualAngles();
      v.chi = parseFloat(e.target.value);
      document.getElementById('vchi').textContent = v.chi.toFixed(2);
      applyVirtualAngles(v.theta, v.chi, v.phi);
    });

    document.getElementById('vthetaSlider').addEventListener('input', (e) => {
      const v = readVirtualAngles();
      v.theta = parseFloat(e.target.value);
      document.getElementById('vvtheta').textContent = v.theta.toFixed(2);
      applyVirtualAngles(v.theta, v.chi, v.phi);
    });

    document.getElementById('vphiSlider').addEventListener('input', (e) => {
      const v = readVirtualAngles();
      v.phi = parseFloat(e.target.value);
      document.getElementById('vvphi').textContent = v.phi.toFixed(2);
      applyVirtualAngles(v.theta, v.chi, v.phi);
    });

    document.getElementById('kappaSignBtn').addEventListener('click', () => {
      const v = readVirtualAngles();
      dev.kappaSignPositive = !dev.kappaSignPositive;
      document.getElementById('kappaSignBtn').textContent = dev.kappaSignPositive ? '\u03BA Sign: +' : '\u03BA Sign: \u2212';
      document.getElementById('kappaSignBtn').classList.toggle('active', dev.kappaSignPositive);
      applyVirtualAngles(v.theta, v.chi, v.phi);
    });

    // Double-click editing for virtual angles
    ['chiSlider:vchi', 'vthetaSlider:vvtheta', 'vphiSlider:vvphi'].forEach(pair => {
      const [sliderId, spanId] = pair.split(':');
      makeSpanEditable(spanId, (val) => {
        const s = document.getElementById(sliderId);
        s.value = Math.max(s.min, Math.min(s.max, val));
        s.dispatchEvent(new Event('input'));
      });
    });
  }

  // Build virtual axes section (fixed joints named virtual_axis_*)
  // Hexapod configs have no `joints` array, so default to empty.
  const virtualAxes = (dev.config.joints || [])
    .map((j, i) => ({ ...j, idx: i }))
    .filter(j => j.name.startsWith('virtual_axis'));
  if (virtualAxes.length > 0) {
    const parentName = (idx) => idx < 0 ? 'root' : (dev.config.joints[idx].name || `joint ${idx}`);
    const rows = virtualAxes.map(j =>
      `<div class="slider-row" style="font-size:0.85em;color:#9ab;">` +
      `<label>${j.name}</label>` +
      `<span style="text-align:right;flex:1;padding-right:4px;color:#556;">` +
      `follows: ${parentName(j.parent)}</span>` +
      `</div>`
    ).join('');
    virtContainer.insertAdjacentHTML('beforeend',
      '<div style="margin-top:8px;padding-top:8px;border-top:1px solid #334;">' +
      '<h2 style="color:#aaa;">Virtual Axes</h2>' +
      rows +
      '</div>');
  }

  // Show/hide IK / Drag Platform button
  const ikBtn = document.getElementById('ikBtn');
  if (dev.isBranching && !isHexapod) {
    ikBtn.style.display = 'none';
  } else {
    ikBtn.style.display = '';
  }
  document.getElementById('chainBtn').style.display = dev.isBranching ? 'none' : '';

  if (isHexapod) {
    ikBtn.textContent = `Drag Platform: ${dev.ikMode ? 'ON' : 'OFF'}`;
  } else {
    ikBtn.textContent = `IK Mode: ${dev.ikMode ? 'ON' : 'OFF'}`;
  }
  ikBtn.classList.toggle('active', dev.ikMode);

  // Show/hide demo button
  const demoBtn = document.getElementById('demoBtn');
  if (dev.isKappaGeometry) {
    demoBtn.style.display = '';
  } else if (dev.config.demoPose) {
    demoBtn.style.display = '';
  } else {
    demoBtn.style.display = 'none';
  }

  // Update chain button state
  document.getElementById('chainBtn').textContent = `Chain: ${dev.chainVisible ? 'ON' : 'OFF'}`;
  document.getElementById('chainBtn').classList.toggle('active', dev.chainVisible);

  // Update IK panel visibility
  if (isHexapod) {
    document.getElementById('ik-panel').style.display = 'none';
    if (dev.ikMode) {
      State.transformControls.attach(dev.platformGroup);
    } else {
      State.transformControls.detach();
    }
  } else {
    document.getElementById('ik-panel').style.display = dev.ikMode ? 'block' : 'none';
    if (dev.ikMode) {
      State.transformControls.attach(dev.ikTarget);
      syncIKSliders(dev);
    } else {
      State.transformControls.detach();
    }
  }

  // Rebuild parent link dropdowns
  rebuildParentDropdown();
  rebuildDeviceParentDropdown();

  // IK editing: the value text is the only visible control for the X/Y/Z
  // position fields (their number inputs are hidden) — click a value to edit.
  ['ikx','iky','ikz','ika','ikb','ikc'].forEach(id => {
    const span = document.getElementById(id.replace('ik', 'ikv'));
    if (span) span.classList.add('editable-val');
    makeSpanEditable(id.replace('ik', 'ikv'), (val) => {
      const s = document.getElementById(id);
      // Clamp only to bounds that exist (the X/Y/Z position fields are
      // unbounded number inputs; the orientation fields keep their ranges).
      const lo = s.min === '' ? -Infinity : parseFloat(s.min);
      const hi = s.max === '' ?  Infinity : parseFloat(s.max);
      s.value = Math.max(lo, Math.min(hi, val));
      s.dispatchEvent(new Event('input'));
    }, 'click');
  });
}

// ============================================================
// rebuildParentDropdown
// ============================================================
export function rebuildParentDropdown() {
  const parentSelect = document.getElementById('stlParentSelect');
  // Keep only the first "World" option, remove all optgroups too
  while (parentSelect.children.length > 1) parentSelect.removeChild(parentSelect.lastChild);
  for (const dev of State.devices) {
    const group = document.createElement('optgroup');
    group.label = dev.name;
    for (const link of dev.config.links || []) {
      const opt = document.createElement('option');
      opt.value = dev.id + ':' + link.name;
      opt.textContent = `${link.name} \u2013 ${link.label}`;
      group.appendChild(opt);
    }
    parentSelect.appendChild(group);
  }
}

// ============================================================
// Device switching
// ============================================================
export function setActiveDevice(dev) {
  if (State.activeDevice === dev) return;

  // Detach any transform controls from old device
  if (State.activeDevice) {
    if (State.activeDevice.ikMode) {
      State.transformControls.detach();
    }
  }

  // Detach device move controls if active
  if (State.moveDeviceActive) {
    State.deviceTransformControls.detach();
    State.setMoveDeviceActive(false);
    document.getElementById('moveDeviceBtn').textContent = 'Move Device Origin';
    document.getElementById('moveDeviceBtn').classList.remove('active');
    document.getElementById('device-mode').style.display = 'none';
    State.deviceTransformControls.setMode('translate');
    State.deviceTransformControls.setSpace('world');
    document.getElementById('devModeT').classList.add('active');
    document.getElementById('devModeR').classList.remove('active');
    document.getElementById('devSpaceBtn').textContent = 'World';
    document.getElementById('devSpaceBtn').classList.remove('active');
  }

  State.setActiveDevice(dev);
  buildControlPanel(dev);
  rebuildDeviceList();

  if (dev.type === 'hexapod') updateHexapodPose(dev);
  else updateFK(dev);
}

// ============================================================
// rebuildDeviceList
// ============================================================
export function rebuildDeviceList() {
  const list = document.getElementById('device-list');
  list.innerHTML = '';
  for (const dev of State.devices) {
    const item = document.createElement('div');
    item.className = 'device-item' + (dev === State.activeDevice ? ' active' : '');

    const nameSpan = document.createElement('span');
    nameSpan.className = 'dev-name';
    nameSpan.textContent = dev.name;
    nameSpan.title = 'Click to select, double-click to rename';
    nameSpan.addEventListener('click', () => setActiveDevice(dev));
    nameSpan.addEventListener('dblclick', (e) => {
      e.stopPropagation();
      const input = document.createElement('input');
      input.type = 'text';
      input.className = 'val-input';
      input.style.width = '120px';
      input.value = dev.name;
      nameSpan.style.display = 'none';
      nameSpan.parentNode.insertBefore(input, nameSpan.nextSibling);
      input.focus();
      input.select();
      const finish = (apply) => {
        if (apply && input.value.trim()) {
          dev.name = input.value.trim();
          nameSpan.textContent = dev.name;
          dev.rootGroup.name = dev.name + '_root';
          // Update panel title if this is the active device
          if (dev === State.activeDevice) {
            document.getElementById('panel-title').textContent = `${dev.name} - Joint Angles (deg)`;
          }
          rebuildParentDropdown();
          rebuildDeviceParentDropdown();
        }
        input.remove();
        nameSpan.style.display = '';
      };
      input.addEventListener('blur', () => finish(true));
      input.addEventListener('keydown', (ev) => {
        if (ev.key === 'Enter') { ev.preventDefault(); finish(true); }
        if (ev.key === 'Escape') finish(false);
      });
    });

    item.appendChild(nameSpan);

    // Only show remove for non-primary devices (keep at least one)
    if (State.devices.length > 1) {
      const rmBtn = document.createElement('button');
      rmBtn.className = 'dev-rm';
      rmBtn.textContent = '\u2715';
      rmBtn.title = 'Remove device';
      rmBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        removeDevice(dev);
      });
      item.appendChild(rmBtn);
    }

    list.appendChild(item);
  }

  // Update add-device dropdown (exclude already loaded configs)
  rebuildAddDeviceDropdown();
}

// ============================================================
// rebuildAddDeviceDropdown
// ============================================================
export function rebuildAddDeviceDropdown() {
  const select = document.getElementById('addDeviceSelect');
  select.innerHTML = '';
  for (const cf of configFiles) {
    const opt = document.createElement('option');
    opt.value = cf;
    opt.textContent = cf.replace('_config.json', '').replace(/_/g, ' ');
    select.appendChild(opt);
  }
}

// ============================================================
// rebuildPrimaryModelDropdown
// ============================================================
export function rebuildPrimaryModelDropdown(currentConfigFile) {
  const select = document.getElementById('primaryModelSelect');
  if (!select) return;
  select.innerHTML = '';
  for (const cf of configFiles) {
    const opt = document.createElement('option');
    opt.value = cf;
    opt.textContent = cf.replace('_config.json', '').replace(/_/g, ' ');
    if (cf === currentConfigFile) opt.selected = true;
    select.appendChild(opt);
  }
}

// ============================================================
// removeDevice
// ============================================================
export function removeDevice(dev) {
  if (State.devices.length <= 1) return; // keep at least one device

  // Switch to another device if this is the active one
  if (dev === State.activeDevice) {
    const remaining = State.devices.filter(d => d !== dev);
    setActiveDevice(remaining[0]);
  }

  // Clean up scene objects
  State.scene.remove(dev.rootGroup);
  dev.rootGroup.traverse(child => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
      else child.material.dispose();
    }
  });

  // Remove labels
  for (const label of dev.meshLabels) {
    label.removeFromParent();
  }

  // Remove chain visualization
  if (dev.chainLine) { State.scene.remove(dev.chainLine); }
  for (const s of dev.chainSpheres) State.scene.remove(s);

  // Remove IK objects
  if (dev.ikTarget) State.scene.remove(dev.ikTarget);
  if (dev.ikLine)   State.scene.remove(dev.ikLine);

  // Remove from registry
  const idx = State.devices.indexOf(dev);
  if (idx >= 0) State.devices.splice(idx, 1);

  rebuildDeviceList();
  rebuildParentDropdown();
  rebuildDeviceParentDropdown();
}

// ============================================================
// rebuildDeviceParentDropdown
// ============================================================
export function rebuildDeviceParentDropdown() {
  const select = document.getElementById('deviceParentSelect');
  if (!select) return;
  while (select.children.length > 1) select.removeChild(select.lastChild);
  for (const dev of State.devices) {
    if (dev === State.activeDevice) continue; // can't parent to self
    const group = document.createElement('optgroup');
    group.label = dev.name;
    for (const link of dev.config.links || []) {
      const opt = document.createElement('option');
      opt.value = dev.id + ':' + link.name;
      opt.textContent = `${link.name} \u2013 ${link.label}`;
      group.appendChild(opt);
    }
    select.appendChild(group);
  }
  // Set current value
  if (State.activeDevice) {
    select.value = State.activeDevice.parentLink || '';
  }
}

// ============================================================
// setDeviceParent
// ============================================================
const _devReparentMat = new THREE.Matrix4();

export function setDeviceParent(dev, parentValue, preserveLocal = false) {
  const rootGroup = dev.rootGroup;
  const { dev: parentDev, linkName } = resolveParentLink(parentValue);

  if (!preserveLocal) {
    // Save world transform for coordinate conversion
    rootGroup.updateWorldMatrix(true, false);
    _devReparentMat.copy(rootGroup.matrixWorld);
  }

  rootGroup.removeFromParent();

  let targetGroup = null;
  if (parentDev && linkName) {
    if (parentDev.linkToJoint[linkName] !== undefined) {
      targetGroup = parentDev.jointRotGroups[parentDev.linkToJoint[linkName]];
    } else if (parentDev.parentGroups && parentDev.parentGroups[linkName]) {
      targetGroup = parentDev.parentGroups[linkName];
    }
  }

  if (targetGroup) {
    if (!preserveLocal) {
      targetGroup.updateWorldMatrix(true, false);
      const localMat = targetGroup.matrixWorld.clone().invert().multiply(_devReparentMat);
      localMat.decompose(rootGroup.position, rootGroup.quaternion, rootGroup.scale);
    }
    targetGroup.add(rootGroup);
    dev.parentLink = parentDev.id + ':' + linkName;
  } else {
    if (!preserveLocal) {
      _devReparentMat.decompose(rootGroup.position, rootGroup.quaternion, rootGroup.scale);
    }
    State.scene.add(rootGroup);
    dev.parentLink = null;
  }
}

// ============================================================
// findDeviceForObject
// ============================================================
export function findDeviceForObject(obj) {
  let current = obj;
  while (current) {
    if (current.userData && current.userData.deviceId) {
      return State.devices.find(d => d.id === current.userData.deviceId) || null;
    }
    current = current.parent;
  }
  return null;
}
