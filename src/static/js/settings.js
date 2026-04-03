/** Settings panel — enhanced with local/cloud toggle, accessibility, and user guidance. */

import { state, saveSettings } from './state.js';
import { switchMode, fetchHealth, switchProvider, fetchCloudProviders } from './api.js';
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

// New enhanced controls
const localToggle   = document.getElementById('local-toggle');
const cloudToggle   = document.getElementById('cloud-toggle');
const localCard     = document.getElementById('local-card');
const cloudCard     = document.getElementById('cloud-card');
const localSettings = document.getElementById('local-settings');
const cloudSettings = document.getElementById('cloud-settings');
const providerSelect = document.getElementById('provider-select');
const apiKeyStatus  = document.getElementById('api-key-status');
const configureApiBtn = document.getElementById('configure-api-btn');

let currentInferenceType = 'local';

export function initSettings() {
  // Apply saved values to form
  systemPrompt.value  = state.settings.systemPrompt;
  maxTokensEl.value   = state.settings.maxTokens;
  temperatureEl.value = state.settings.temperature;
  topPEl.value        = state.settings.topP;
  modeSelect.value    = state.settings.inferenceMode;
  
  // Initialize inference type from state or default to local
  currentInferenceType = state.settings.inferenceType || 'local';
  
  // Initialize cloud provider dropdown from saved state
  if (providerSelect && state.settings.cloudProvider) {
    providerSelect.value = state.settings.cloudProvider;
  }
  
  updateInferenceDisplay();
  
  // Initialize debug toggle with proper ARIA
  updateDebugToggle();

  // Close panel
  closeBtn.addEventListener('click', () => overlay.classList.remove('open'));
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) overlay.classList.remove('open');
  });

  // Enhanced local/cloud toggle
  setupInferenceToggle();

  // Debug toggle with ARIA support
  debugToggle.addEventListener('click', () => {
    state.settings.debug = !state.settings.debug;
    updateDebugToggle();
    saveSettings();
  });

  // Thinking toggle (only if model supports it — managed by app.js)
  thinkingBtn.addEventListener('click', () => {
    if (thinkingBtn.classList.contains('disabled')) return;
    state.thinking = !state.thinking;
    thinkingBtn.textContent = state.thinking ? '◈ Think' : '◈ Think';
    thinkingBtn.classList.toggle('off', !state.thinking);
  });

  // Enhanced save with inference type support
  saveBtn.addEventListener('click', async () => {
    await saveEnhancedSettings();
  });

  // Accessibility: Help buttons
  setupHelpButtons();

  // API configuration
  if (configureApiBtn) {
    configureApiBtn.addEventListener('click', () => {
      showApiConfigDialog();
    });
  }

  // Re-check API key status when provider dropdown changes
  if (providerSelect) {
    providerSelect.addEventListener('change', () => {
      checkCloudProviderStatus();
    });
  }
}

/** Enhanced inference toggle setup */
function setupInferenceToggle() {
  // Check if elements exist before adding listeners
  if (localToggle && cloudToggle) {
    localToggle.addEventListener('click', () => {
      setInferenceType('local');
    });
    
    cloudToggle.addEventListener('click', () => {
      setInferenceType('cloud');
    });
  }

  // Inference cards
  if (localCard && cloudCard) {
    localCard.addEventListener('click', () => {
      setInferenceType('local');
    });
    
    cloudCard.addEventListener('click', () => {
      setInferenceType('cloud');
    });

    // Keyboard support for cards
    localCard.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        setInferenceType('local');
      }
    });
    
    cloudCard.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        setInferenceType('cloud');
      }
    });
  }
}

/** Set inference type (local/cloud) — also triggers server-side switch */
async function setInferenceType(type) {
  const prev = currentInferenceType;
  currentInferenceType = type;
  state.settings.inferenceType = type;
  updateInferenceDisplay();
  saveSettings();

  try {
    if (type === 'cloud') {
      let provider = state.settings.cloudProvider || (providerSelect ? providerSelect.value : '');
      // Auto-detect configured provider if none saved
      if (!provider) {
        try {
          const data = await fetchCloudProviders();
          const configured = data.providers?.find(p => p.configured);
          if (configured) provider = configured.name;
        } catch (_) {}
      }
      if (!provider) {
        showToast('No cloud provider configured. Set an API key first.', 'error');
        currentInferenceType = prev;
        state.settings.inferenceType = prev;
        updateInferenceDisplay();
        saveSettings();
        return;
      }
      showToast(`Switching to cloud (${provider})…`, 'info');
      await switchProvider(provider);
      state.settings.cloudProvider = provider;
      if (providerSelect) providerSelect.value = provider;
      showToast(`Cloud provider: ${provider}`, 'info');
    } else {
      const mode = state.settings.inferenceMode || (modeSelect ? modeSelect.value : '') || 'turboquant';
      showToast(`Switching to local (${mode})…`, 'info');
      await switchMode(mode);
      state.settings.inferenceMode = mode;
      showToast(`Local mode: ${mode}`, 'info');
    }
    window.dispatchEvent(new CustomEvent('settings-changed'));
  } catch (e) {
    showToast(`Switch failed: ${e.message}`, 'error');
    // Revert on failure
    currentInferenceType = prev;
    state.settings.inferenceType = prev;
    updateInferenceDisplay();
    saveSettings();
  }
}

/** Update inference display based on current type */
function updateInferenceDisplay() {
  // Update header toggle
  if (localToggle && cloudToggle) {
    localToggle.classList.toggle('active', currentInferenceType === 'local');
    cloudToggle.classList.toggle('active', currentInferenceType === 'cloud');
    
    // Update ARIA states
    localToggle.setAttribute('aria-checked', currentInferenceType === 'local');
    cloudToggle.setAttribute('aria-checked', currentInferenceType === 'cloud');
  }
  
  // Update inference cards
  if (localCard && cloudCard) {
    localCard.classList.toggle('active', currentInferenceType === 'local');
    cloudCard.classList.toggle('active', currentInferenceType === 'cloud');
  }
  
  // Show/hide settings sections
  if (localSettings) {
    if (currentInferenceType === 'local') {
      localSettings.classList.remove('hidden');
    } else {
      localSettings.classList.add('hidden');
    }
  }
  
  if (cloudSettings) {
    if (currentInferenceType === 'cloud') {
      cloudSettings.classList.remove('hidden');
      // Auto-check provider status when switched to cloud
      checkCloudProviderStatus();
    } else {
      cloudSettings.classList.add('hidden');
    }
  }
  
  // Update model display
  updateModelDisplay();
}

/** Update debug toggle with ARIA support */
function updateDebugToggle() {
  debugToggle.textContent = state.settings.debug ? 'ON' : 'OFF';
  debugToggle.setAttribute('aria-checked', state.settings.debug);
  debugToggle.classList.toggle('on', state.settings.debug);
}

/** Enhanced save function */
async function saveEnhancedSettings() {
  state.settings.systemPrompt = systemPrompt.value.trim();
  state.settings.maxTokens    = Math.max(1, Math.min(16384, parseInt(maxTokensEl.value, 10) || 2048));
  state.settings.temperature  = Math.max(0, Math.min(2, parseFloat(temperatureEl.value) || 0.7));
  state.settings.topP         = Math.max(0, Math.min(1, parseFloat(topPEl.value) || 0.95));
  state.settings.inferenceType = currentInferenceType;

  const newMode = modeSelect.value;
  const newProvider = providerSelect ? providerSelect.value : null;
  
  const modeChanged = newMode !== state.settings.inferenceMode;
  const providerChanged = newProvider && newProvider !== state.settings.cloudProvider;
  
  if (currentInferenceType === 'local') {
    state.settings.inferenceMode = newMode;
  } else {
    state.settings.cloudProvider = newProvider;
  }
  
  saveSettings();
  overlay.classList.remove('open');

  if (modeChanged || providerChanged) {
    modeLoading.classList.add('active');
    const changeType = currentInferenceType === 'local' ? `mode: ${newMode}` : `provider: ${newProvider}`;
    showToast(`Switching ${changeType}… (model reloading)`);
    
    try {
      if (currentInferenceType === 'local') {
        await switchMode(newMode);
      } else {
        // TODO: Add cloud provider switching API
        await switchCloudProvider(newProvider);
      }
      showToast(`Successfully switched to ${changeType}`);
    } catch (e) {
      showToast(`Switch failed: ${e.message}`, 'error');
      // Revert settings on failure
      try {
        const d = await fetchHealth();
        if (currentInferenceType === 'local') {
          modeSelect.value = d.inference_mode;
          state.settings.inferenceMode = d.inference_mode;
        }
      } catch (_) {}
    } finally {
      modeLoading.classList.remove('active');
      // Trigger health poll to update status bar model name
      window.dispatchEvent(new CustomEvent('settings-changed'));
    }
  } else {
    showToast('Configuration updated.');
  }
}

/** Setup help button interactions */
function setupHelpButtons() {
  const helpButtons = document.querySelectorAll('.help-btn');
  helpButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
      const buttonId = e.target.closest('.help-btn').id;
      showHelpTooltip(buttonId, e.target);
    });
  });
}

/** Show contextual help */
function showHelpTooltip(helpId, target) {
  const helpMessages = {
    'inference-help': 'Local inference runs models on your device for privacy and offline use. Cloud inference uses remote AI providers for faster responses and latest models.',
    'quant-help': 'Quantization reduces model memory usage. TurboQuant (recommended) offers best balance. Zero-Quant saves more memory. Ultra-Quant enables large models on limited hardware.',
  };
  
  const message = helpMessages[helpId] || 'Help information not available.';
  showToast(message, 'info');
}

/** Show API configuration dialog */
function showApiConfigDialog() {
  // Remove existing dialog if any
  const existing = document.getElementById('api-config-dialog');
  if (existing) existing.remove();

  const provider = providerSelect ? providerSelect.value : 'openai';
  const envVar = `TURBOQUANT_CLOUD_${provider.toUpperCase()}_API_KEY`;

  const dialog = document.createElement('div');
  dialog.id = 'api-config-dialog';
  dialog.className = 'api-config-overlay';
  dialog.setAttribute('role', 'dialog');
  dialog.setAttribute('aria-modal', 'true');
  dialog.setAttribute('aria-label', 'Configure API Key');
  dialog.innerHTML = `
    <div class="api-config-panel">
      <div class="panel-header">
        <h3 class="panel-title">◈ Configure ${provider}</h3>
        <button class="close-btn api-config-close" aria-label="Close">✕</button>
      </div>
      <p class="api-config-hint">
        Enter your API key for <strong>${provider}</strong>.
        Alternatively, set the <code>${envVar}</code> environment variable.
      </p>
      <div class="api-config-field">
        <label for="api-key-input" class="field-label">API Key</label>
        <input type="password" id="api-key-input" class="api-key-input"
               placeholder="sk-..." autocomplete="off"
               aria-describedby="api-key-hint" />
        <div id="api-key-hint" class="field-description">Your key is sent to the server and not stored in the browser.</div>
      </div>
      <div class="api-config-actions">
        <button class="text-btn api-config-cancel">Cancel</button>
        <button class="save-btn api-config-save">Connect</button>
      </div>
    </div>
  `;

  document.body.appendChild(dialog);

  const keyInput = dialog.querySelector('#api-key-input');
  const closeBtn = dialog.querySelector('.api-config-close');
  const cancelBtn = dialog.querySelector('.api-config-cancel');
  const saveApiBtn = dialog.querySelector('.api-config-save');

  function closeDialog() { dialog.remove(); }

  closeBtn.addEventListener('click', closeDialog);
  cancelBtn.addEventListener('click', closeDialog);
  dialog.addEventListener('click', (e) => { if (e.target === dialog) closeDialog(); });

  saveApiBtn.addEventListener('click', async () => {
    const apiKey = keyInput.value.trim();
    if (!apiKey) {
      showToast('Please enter an API key', 'error');
      return;
    }
    saveApiBtn.disabled = true;
    saveApiBtn.textContent = 'Connecting…';
    try {
      const result = await switchProvider(provider, apiKey);
      showToast(`Connected to ${provider} (${result.model})`, 'info');
      state.settings.cloudProvider = provider;
      state.settings.inferenceType = 'cloud';
      saveSettings();
      updateApiKeyStatus(true, provider, true);
      closeDialog();
    } catch (e) {
      showToast(`Connection failed: ${e.message}`, 'error');
    } finally {
      saveApiBtn.disabled = false;
      saveApiBtn.textContent = 'Connect';
    }
  });

  keyInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); saveApiBtn.click(); }
  });

  setTimeout(() => keyInput.focus(), 100);
}

/** Update API key status indicator */
function updateApiKeyStatus(configured, providerName, active) {
  if (!apiKeyStatus) return;
  const indicator = apiKeyStatus.querySelector('.status-indicator');
  const text = apiKeyStatus.querySelector('.status-text');
  const configBtn = apiKeyStatus.querySelector('.text-btn');
  if (configured && active) {
    indicator.className = 'status-indicator active';
    text.textContent = 'Active';
    if (configBtn) configBtn.textContent = 'Reconfigure';
  } else if (configured) {
    indicator.className = 'status-indicator configured';
    text.textContent = 'Key set (from .env)';
    if (configBtn) configBtn.textContent = 'Connect';
  } else {
    indicator.className = 'status-indicator not-configured';
    text.textContent = 'Not configured';
    if (configBtn) configBtn.textContent = 'Configure';
  }
}

/** Check cloud provider status on settings open */
async function checkCloudProviderStatus() {
  try {
    const data = await fetchCloudProviders();
    const provider = providerSelect ? providerSelect.value : '';
    const p = (data.providers || []).find(prov => prov.name === provider);
    updateApiKeyStatus(p?.configured || false, provider, p?.active || false);
  } catch (_) {
    updateApiKeyStatus(false);
  }
}

/** Update model/provider display based on inference type */
function updateModelDisplay() {
  const modelSelect = document.getElementById('header-model-select');
  if (!modelSelect) return;
  
  if (currentInferenceType === 'local') {
    // Show local models (existing behavior)
    modelSelect.setAttribute('aria-label', 'Select local model');
  } else {
    // Show cloud providers
    modelSelect.setAttribute('aria-label', 'Select cloud provider');
    // TODO: Populate with cloud providers
  }
}

/** Switch cloud provider via server API */
async function switchCloudProvider(provider) {
  const result = await switchProvider(provider);
  state.settings.cloudProvider = provider;
  saveSettings();
  updateApiKeyStatus(true, provider, true);
  return result;
}

export function openSettings() {
  overlay.classList.add('open');
  checkCloudProviderStatus();
  // Focus first interactive element for accessibility
  setTimeout(() => {
    const firstInput = overlay.querySelector('input, select, button, textarea');
    if (firstInput) firstInput.focus();
  }, 100);
}

/** Sync mode selector from a health poll response. */
export function syncModeFromHealth(inferenceMode) {
  if (!inferenceMode || modeLoading.classList.contains('active')) return;

  // "cloud" is an inference type, not a local mode — don't set the mode dropdown
  if (inferenceMode === 'cloud') {
    currentInferenceType = 'cloud';
    state.settings.inferenceType = 'cloud';
    updateInferenceDisplay();
    return;
  }

  // Local modes: standard, turboquant, zero-quant, ultra-quant
  modeSelect.value = inferenceMode;
  const headerMode = document.getElementById('header-mode-select');
  if (headerMode) headerMode.value = inferenceMode;
  state.settings.inferenceMode = inferenceMode;

  if (currentInferenceType === 'cloud') {
    currentInferenceType = 'local';
    state.settings.inferenceType = 'local';
    updateInferenceDisplay();
  }
}
