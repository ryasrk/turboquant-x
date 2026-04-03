/** Main entry — wires all modules together. */

import { state } from './state.js';
import { fetchHealth, switchMode, fetchAvailableModels, switchModel, switchProvider } from './api.js';
import { showToast, appendMessage, renderAssistantContent, getChatEl, updateContextMeter } from './ui.js';
import { sendMessage, clearSession } from './chat.js';
import { initSettings, openSettings, syncModeFromHealth } from './settings.js';
import { refreshModelList } from './models.js';
import { initUpload, setVisionEnabled } from './upload.js';
import { toggleAgent } from './agent.js';
import { initMcpPanel } from './mcp.js';
import {
  isLoggedIn, getUser, logout as authLogout,
  apiLogin, apiRegister, apiMe,
} from './auth.js';
import {
  fetchSessions, createSession, loadSession, saveMessage,
  getCurrentSessionId, setCurrentSessionId, renderSessionList,
  initSidebar, autoTitleSession,
} from './sessions.js';

// ── DOM refs ──────────────────────────────────────────────────────────
const inputEl      = document.getElementById('user-input');
const sendBtn      = document.getElementById('send-btn');
const clearBtn     = document.getElementById('clear-btn');
const settingsBtn  = document.getElementById('settings-btn');
const thinkingBtn  = document.getElementById('thinking-btn');
const agentBtn     = document.getElementById('agent-btn');
const uptimeDisp   = document.getElementById('uptime-disp');
const modelNameDisp = document.getElementById('model-name-disp');
const headerModeSelect  = document.getElementById('header-mode-select');
const headerModelSelect = document.getElementById('header-model-select');
const localToggleHeader = document.getElementById('local-toggle');
const cloudToggleHeader = document.getElementById('cloud-toggle');

// ── Init modules ──────────────────────────────────────────────────────
initSettings();
initUpload();
initInferenceToggle();
initMcpPanel();

// ── Inference type toggle (header) ───────────────────────────────────
function initInferenceToggle() {
  if (!localToggleHeader || !cloudToggleHeader) return;
  
  localToggleHeader.addEventListener('click', () => {
    setHeaderInferenceType('local');
  });
  
  cloudToggleHeader.addEventListener('click', () => {
    setHeaderInferenceType('cloud');
  });
  
  // Initialize display and model select for current type
  const currentType = state.settings.inferenceType || 'local';
  updateHeaderInferenceDisplay(currentType);
  updateHeaderModelSelect(currentType);
}

function setHeaderInferenceType(type) {
  state.settings.inferenceType = type;
  updateHeaderInferenceDisplay(type);
  
  // Update model select based on inference type
  updateHeaderModelSelect(type);
  
  showToast(`Switched to ${type} inference`, 'info');
}

function updateHeaderInferenceDisplay(type) {
  if (!localToggleHeader || !cloudToggleHeader) return;
  
  localToggleHeader.classList.toggle('active', type === 'local');
  cloudToggleHeader.classList.toggle('active', type === 'cloud');
  
  localToggleHeader.setAttribute('aria-checked', type === 'local');
  cloudToggleHeader.setAttribute('aria-checked', type === 'cloud');
}

function updateHeaderModelSelect(type) {
  if (!headerModelSelect) return;
  if (type === 'local') {
    headerModelSelect.title = 'Switch local model';
    headerModelSelect.setAttribute('aria-label', 'Select local model');
    populateHeaderModels(); // Existing local models
  } else {
    headerModelSelect.title = 'Switch cloud provider';
    headerModelSelect.setAttribute('aria-label', 'Select cloud provider');
    populateHeaderProviders(); // New cloud providers
  }
}

async function populateHeaderProviders() {
  if (!headerModelSelect) return;
  const providers = [
    { name: 'OpenAI (GPT-4)', value: 'openai' },
    { name: 'Anthropic (Claude)', value: 'anthropic' },
    { name: 'Zhipu (GLM)', value: 'zhipu' },
    { name: 'Moonshot (Kimi)', value: 'moonshot' },
    { name: 'DeepSeek', value: 'deepseek' }
  ];
  
  headerModelSelect.innerHTML = '';
  for (const provider of providers) {
    const opt = document.createElement('option');
    opt.value = provider.value;
    opt.textContent = provider.name;
    opt.selected = provider.value === (state.settings.cloudProvider || 'openai');
    headerModelSelect.appendChild(opt);
  }
}

// ── Thinking support ──────────────────────────────────────────────────
let _supportsThinking = false;

function updateThinkingUI(supports) {
  _supportsThinking = supports;
  if (!supports) {
    state.thinking = false;
    thinkingBtn.textContent = '◈ Think N/A';
    thinkingBtn.classList.add('off', 'disabled');
    thinkingBtn.title = 'This model does not support thinking mode';
  } else {
    thinkingBtn.classList.remove('disabled');
    thinkingBtn.title = 'Toggle chain-of-thought thinking';
    thinkingBtn.textContent = state.thinking ? '◈ Think ON' : '◈ Think';
    thinkingBtn.classList.toggle('off', !state.thinking);
  }
}

// ── Header mode switcher ──────────────────────────────────────────────
let _modeSwitching = false;
if (headerModeSelect) {
  headerModeSelect.addEventListener('change', async () => {
    const newMode = headerModeSelect.value;
    if (_modeSwitching || newMode === state.settings.inferenceMode) return;
    _modeSwitching = true;
    headerModeSelect.disabled = true;
    showToast(`Switching to ${newMode}… (model reloading)`);
    try {
      await switchMode(newMode);
      state.settings.inferenceMode = newMode;
      syncModeFromHealth(newMode);
      showToast(`Mode: ${newMode}`);
    } catch (e) {
      showToast(`Mode switch failed: ${e.message}`);
      headerModeSelect.value = state.settings.inferenceMode;
    } finally {
      _modeSwitching = false;
      headerModeSelect.disabled = false;
      pollHealth();
    }
  });
}

// ── Header model switcher ─────────────────────────────────────────────
let _modelSwitching = false;
if (headerModelSelect) {
  headerModelSelect.addEventListener('change', async () => {
    const value = headerModelSelect.value;
    if (_modelSwitching || !value) return;

    // Cloud mode: switch provider on server
    if (state.settings.inferenceType === 'cloud') {
      _modelSwitching = true;
      headerModelSelect.disabled = true;
      const displayName = headerModelSelect.selectedOptions[0]?.textContent || value;
      showToast(`Switching to ${displayName}…`);
      try {
        const result = await switchProvider(value);
        state.settings.cloudProvider = value;
        showToast(`Cloud provider: ${displayName} (${result.model})`, 'info');
      } catch (e) {
        showToast(`Provider switch failed: ${e.message}`, 'error');
      } finally {
        _modelSwitching = false;
        headerModelSelect.disabled = false;
      }
      return;
    }

    // Local mode: load the model file
    _modelSwitching = true;
    headerModelSelect.disabled = true;
    showToast(`Loading ${value}…`);
    try {
      const d = await switchModel(value);
      showToast(`Model loaded: ${d.model_name}`);
    } catch (e) {
      showToast(`Load failed: ${e.message}`);
    } finally {
      _modelSwitching = false;
      headerModelSelect.disabled = false;
      pollHealth();
      populateHeaderModels();
    }
  });
}

async function populateHeaderModels() {
  if (!headerModelSelect) return;
  try {
    const d = await fetchAvailableModels();
    headerModelSelect.innerHTML = '';
    for (const m of (d.models || [])) {
      const opt = document.createElement('option');
      opt.value = m.filename;
      opt.textContent = m.loaded ? `● ${m.filename}` : m.filename;
      opt.selected = m.loaded;
      headerModelSelect.appendChild(opt);
    }
  } catch (_) {}
}

// ── Health poll ───────────────────────────────────────────────────────
async function pollHealth() {
  try {
    const d = await fetchHealth();
    updateThinkingUI(!!d.supports_thinking);
    setVisionEnabled(!!d.supports_vision);
    if (headerModeSelect) {
      headerModeSelect.value = d.inference_mode || 'standard';
    }
    syncModeFromHealth(d.inference_mode);

    // Sync inference type (local/cloud) from server state
    const serverIsCloud = d.inference_mode === 'cloud';
    const clientIsCloud = state.settings.inferenceType === 'cloud';
    if (serverIsCloud !== clientIsCloud) {
      state.settings.inferenceType = serverIsCloud ? 'cloud' : 'local';
      updateHeaderInferenceDisplay(state.settings.inferenceType);
      updateHeaderModelSelect(state.settings.inferenceType);
    }

    if (d.model_name && modelNameDisp) {
      modelNameDisp.textContent = `model: ${d.model_name}`;
      modelNameDisp.title = d.model_name;
    }
    if (d.uptime_s != null && uptimeDisp) {
      uptimeDisp.textContent = `uptime: ${Math.floor(d.uptime_s)}s`;
    }
    if (d.context_max) {
      state.contextMax = d.context_max;
      updateContextMeter();
    }
    if (d.loading) {
      if (headerModelSelect) headerModelSelect.disabled = true;
      if (headerModeSelect) headerModeSelect.disabled = true;
    }
  } catch (_) {}
}
pollHealth();
setInterval(pollHealth, 15000);

// ── Context warning dismiss ───────────────────────────────────────────
const ctxDismiss = document.getElementById('ctx-dismiss');
if (ctxDismiss) {
  ctxDismiss.addEventListener('click', () => {
    document.getElementById('context-warning')?.classList.remove('visible');
  });
}

// ── Scroll-to-bottom button ──────────────────────────────────────────
const scrollBottomBtn = document.getElementById('scroll-bottom-btn');
const chatEl = getChatEl();
function checkScrollBottom() {
  const atBottom = chatEl.scrollHeight - chatEl.scrollTop - chatEl.clientHeight < 100;
  scrollBottomBtn.classList.toggle('visible', !atBottom);
}
chatEl.addEventListener('scroll', checkScrollBottom);
scrollBottomBtn.addEventListener('click', () => {
  chatEl.scrollTo({ top: chatEl.scrollHeight, behavior: 'smooth' });
});

// ── Model list ────────────────────────────────────────────────────────
refreshModelList();
// Only populate local models if in local mode (cloud handled by initInferenceToggle)
if (state.settings.inferenceType !== 'cloud') {
  populateHeaderModels();
}

// ── Input handling ────────────────────────────────────────────────────
inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 216) + 'px';
});

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!state.streaming) sendMessage();
  }
});

sendBtn.addEventListener('click', () => { if (!state.streaming) sendMessage(); });
clearBtn.addEventListener('click', clearSession);
settingsBtn.addEventListener('click', () => {
  openSettings();
  refreshModelList();
});
agentBtn.addEventListener('click', () => {
  const on = toggleAgent();
  agentBtn.classList.toggle('active', on);
  showToast(on ? 'Agent mode ON' : 'Agent mode OFF');
});

// ── Auth overlay ──────────────────────────────────────────────────────
const authOverlay  = document.getElementById('auth-overlay');
const authTabs     = document.querySelectorAll('.auth-tab');
const authForm     = document.getElementById('auth-form');
const authUsername  = document.getElementById('auth-username');
const authPassword  = document.getElementById('auth-password');
const authSubmit   = document.getElementById('auth-submit');
const authError    = document.getElementById('auth-error');
const authSkipBtn  = document.getElementById('auth-skip-btn');
const authUserDisp = document.getElementById('auth-user-display');
const logoutBtn    = document.getElementById('logout-btn');

let _authMode = 'login'; // 'login' | 'register'

function showAuthOverlay() {
  authOverlay.classList.remove('hidden');
}
function hideAuthOverlay() {
  authOverlay.classList.add('hidden');
}
function updateAuthUI() {
  if (isLoggedIn()) {
    hideAuthOverlay();
    const u = getUser();
    if (authUserDisp) authUserDisp.textContent = u?.username || '';
    if (logoutBtn) logoutBtn.style.display = '';
    // Load sessions
    _loadSessions();
  } else {
    if (authUserDisp) authUserDisp.textContent = 'guest';
    if (logoutBtn) logoutBtn.style.display = 'none';
  }
}

authTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    _authMode = tab.dataset.tab;
    authTabs.forEach(t => {
      t.classList.toggle('active', t === tab);
      t.setAttribute('aria-selected', t === tab);
    });
    authSubmit.textContent = _authMode === 'login' ? 'Login' : 'Register';
    authPassword.setAttribute('autocomplete', _authMode === 'login' ? 'current-password' : 'new-password');
    authPassword.placeholder = _authMode === 'login' ? 'Password' : 'Choose a password';
    authError.textContent = '';
  });
});

authForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  authError.textContent = '';
  const username = authUsername.value.trim();
  const password = authPassword.value;
  if (!username) { authError.textContent = 'Username is required'; authUsername.focus(); return; }
  if (username.length < 3) { authError.textContent = 'Username must be at least 3 characters'; authUsername.focus(); return; }
  if (!password) { authError.textContent = 'Password is required'; authPassword.focus(); return; }
  if (password.length < 4) { authError.textContent = 'Password must be at least 4 characters'; authPassword.focus(); return; }
  authSubmit.disabled = true;
  authSubmit.textContent = _authMode === 'login' ? 'Logging in…' : 'Registering…';
  try {
    if (_authMode === 'login') {
      await apiLogin(username, password);
    } else {
      await apiRegister(username, password);
    }
    updateAuthUI();
    showToast(`Welcome, ${username}`);
  } catch (err) {
    authError.textContent = err.message;
    authPassword.focus();
  } finally {
    authSubmit.disabled = false;
    authSubmit.textContent = _authMode === 'login' ? 'Login' : 'Register';
  }
});

authSkipBtn.addEventListener('click', () => {
  hideAuthOverlay();
  showToast('Continuing as guest — chat history won\'t be saved');
});

if (logoutBtn) {
  logoutBtn.addEventListener('click', () => {
    authLogout();
    state.history = [];
    clearSession();
    showAuthOverlay();
    updateAuthUI();
    showToast('Logged out');
  });
}

// ── Session sidebar ───────────────────────────────────────────────────
async function _loadSessions() {
  const sessions = await fetchSessions();
  renderSessionList();
}

function _renderLoadedMessages(messages) {
  const chatEl = getChatEl();
  // Clear chat area
  chatEl.innerHTML = '';
  for (const m of messages) {
    const { bubble } = appendMessage(m.role, '');
    if (m.role === 'assistant') {
      renderAssistantContent(bubble, m.content);
    } else {
      bubble.textContent = m.content;
    }
  }
}

initSidebar(
  // onLoadSession
  async (sessionId) => {
    await loadSession(sessionId, _renderLoadedMessages);
    const sidebar = document.getElementById('session-sidebar');
    if (sidebar) {
      sidebar.classList.remove('open');
      document.body.classList.remove('sidebar-open');
    }
  },
  // onNewChat
  async () => {
    const session = await createSession('New Chat');
    state.history = [];
    clearSession();
    await fetchSessions();
    renderSessionList();
    showToast('New chat created');
    const sidebar = document.getElementById('session-sidebar');
    if (sidebar) {
      sidebar.classList.remove('open');
      document.body.classList.remove('sidebar-open');
    }
  },
);

// ── Export saveMessage for chat.js hook ────────────────────────────────
// chat.js needs to save messages after send/receive
window._tqSaveMessage = async (role, content) => {
  if (!isLoggedIn()) return;
  // Auto-create session if none exists
  if (!getCurrentSessionId()) {
    await createSession('New Chat');
    await fetchSessions();
    renderSessionList();
  }
  await saveMessage(role, content);
  // Auto-title on first user message
  if (role === 'user' && state.history.length <= 1) {
    await autoTitleSession(content);
    await fetchSessions();
  }
};

// ── Startup ───────────────────────────────────────────────────────────
if (isLoggedIn()) {
  // Validate token
  apiMe().then(user => {
    if (!user) showAuthOverlay();
    else updateAuthUI();
  });
} else {
  showAuthOverlay();
}
