/** Model list — scan, display, load/switch. */

import { fetchAvailableModels, switchModel, fetchHealth } from './api.js';
import { showToast } from './ui.js';

const modelListEl = document.getElementById('model-list');
const modelBadge  = document.getElementById('model-badge');

let _switching = false;

export async function refreshModelList() {
  try {
    const d = await fetchAvailableModels();
    modelListEl.innerHTML = '';
    if (!d.models || d.models.length === 0) {
      modelListEl.innerHTML =
        '<div class="model-item"><span class="model-name" style="color:var(--text-dim)">no .gguf files in models/</span></div>';
      return;
    }
    for (const m of d.models) {
      const item = document.createElement('div');
      item.className =
        'model-item' +
        (m.loaded ? ' active' : '') +
        (_switching ? ' loading-model' : '');

      const info = document.createElement('div');
      info.className = 'model-info';
      const name = document.createElement('div');
      name.className = 'model-name';
      name.textContent = m.filename;
      name.title = m.path;
      const size = document.createElement('div');
      size.className = 'model-size';
      size.textContent = `${m.size_gb} GB`;
      info.appendChild(name);
      info.appendChild(size);

      const status = document.createElement('span');
      status.className = 'model-status';
      if (_switching && m.loaded) {
        status.classList.add('loading-state');
        status.textContent = 'loading';
      } else if (m.loaded) {
        status.classList.add('loaded');
        status.textContent = 'loaded';
      } else {
        status.classList.add('not-loaded');
        status.textContent = '—';
      }

      item.appendChild(info);
      item.appendChild(status);

      if (!m.loaded) {
        const btn = document.createElement('button');
        btn.className = 'model-load-btn';
        btn.textContent = 'Load';
        btn.disabled = _switching;
        btn.addEventListener('click', () => loadModel(m.filename));
        item.appendChild(btn);
      }

      modelListEl.appendChild(item);
    }
  } catch (_) {}
}

export async function loadModel(filename) {
  if (_switching) return;
  _switching = true;
  modelBadge.textContent = 'loading…';
  showToast(`Loading ${filename}…`);
  refreshModelList();

  try {
    const d = await switchModel(filename);
    modelBadge.textContent = d.model_name || filename;
    showToast(`Model loaded: ${d.model_name} (${d.size_gb} GB)`);
  } catch (e) {
    showToast(`Load failed: ${e.message}`);
    try {
      const h = await fetchHealth();
      modelBadge.textContent = h.model_name || 'unknown';
    } catch (_) {}
  } finally {
    _switching = false;
    refreshModelList();
  }
}
