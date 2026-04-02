/** Settings panel — system prompt, params, mode selector, debug toggle. */

import { state, saveSettings } from './state.js';
import { switchMode, fetchHealth } from './api.js';
import { showToast } from './ui.js';

const overlay       = document.getElementById('settings-overlay');
const closeBtn      = document.getElementById('settings-close');
const saveBtn       = document.getElementById('save-settings-btn');
const systemPrompt  = document.getElementById('system-prompt-input');
const maxTokensEl   = document.getElementById('max-tokens-input');
const temperatureEl = document.getElementById('temperature-input');
const topPEl        = document.getElementById('top-p-input');
const modeSelect    = document.getElementById('mode-select');
const debugToggle   = document.getElementById('debug-toggle');
const modeLoading   = document.getElementById('mode-loading');
const thinkingBtn   = document.getElementById('thinking-btn');
const headerModeSelect = document.getElementById('header-mode-select');

export function initSettings() {
  // Apply saved values to form
  systemPrompt.value  = state.settings.systemPrompt;
  maxTokensEl.value   = state.settings.maxTokens;
  temperatureEl.value = state.settings.temperature;
  topPEl.value        = state.settings.topP;
  modeSelect.value    = state.settings.inferenceMode;
  debugToggle.textContent = state.settings.debug ? 'ON' : 'OFF';
  if (state.settings.debug) debugToggle.classList.add('on');

  // Close panel
  closeBtn.addEventListener('click', () => overlay.classList.remove('open'));
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) overlay.classList.remove('open');
  });

  // Debug toggle
  debugToggle.addEventListener('click', () => {
    state.settings.debug = !state.settings.debug;
    debugToggle.textContent = state.settings.debug ? 'ON' : 'OFF';
    debugToggle.classList.toggle('on', state.settings.debug);
    saveSettings();
  });

  // Thinking toggle (only if model supports it — managed by app.js)
  thinkingBtn.addEventListener('click', () => {
    if (thinkingBtn.classList.contains('disabled')) return;
    state.thinking = !state.thinking;
    thinkingBtn.textContent = state.thinking ? '◈ Think' : '◈ Think';
    thinkingBtn.classList.toggle('off', !state.thinking);
  });

  // Save
  saveBtn.addEventListener('click', async () => {
    state.settings.systemPrompt = systemPrompt.value.trim();
    state.settings.maxTokens    = Math.max(1, Math.min(16384, parseInt(maxTokensEl.value, 10) || 2048));
    state.settings.temperature  = Math.max(0, Math.min(2, parseFloat(temperatureEl.value) || 0.7));
    state.settings.topP         = Math.max(0, Math.min(1, parseFloat(topPEl.value) || 0.95));

    const newMode     = modeSelect.value;
    const modeChanged = newMode !== state.settings.inferenceMode;
    state.settings.inferenceMode = newMode;
    saveSettings();
    overlay.classList.remove('open');

    if (modeChanged) {
      modeLoading.classList.add('active');
      showToast(`Switching to ${newMode}… (model reloading)`);
      try {
        await switchMode(newMode);
        showToast(`Mode switched to ${newMode}`);
      } catch (e) {
        showToast(`Mode switch failed: ${e.message}`);
        try {
          const d = await fetchHealth();
          modeSelect.value       = d.inference_mode;
          state.settings.inferenceMode = d.inference_mode;
        } catch (_) {}
      } finally {
        modeLoading.classList.remove('active');
      }
    } else {
      showToast('Configuration updated.');
    }
  });
}

export function openSettings() {
  overlay.classList.add('open');
}

/** Sync mode selector from a health poll response. */
export function syncModeFromHealth(inferenceMode) {
  if (inferenceMode && !modeLoading.classList.contains('active')) {
    modeSelect.value = inferenceMode;
    if (headerModeSelect) headerModeSelect.value = inferenceMode;
    state.settings.inferenceMode = inferenceMode;
  }
}
