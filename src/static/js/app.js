/** Main entry — wires all modules together. */

import { state } from './state.js';
import { fetchHealth, switchMode, fetchAvailableModels, switchModel } from './api.js';
import { showToast, appendMessage, renderAssistantContent, getChatEl, updateContextMeter } from './ui.js';
import { sendMessage, clearSession } from './chat.js';
import { initSettings, openSettings, syncModeFromHealth } from './settings.js';
import { refreshModelList } from './models.js';
import { initUpload, setVisionEnabled } from './upload.js';
import { toggleAgent } from './agent.js';
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
const headerModeSelect  = document.getElementById('header-mode-select');
const headerModelSelect = document.getElementById('header-model-select');

// ── Init modules ──────────────────────────────────────────────────────
initSettings();
initUpload();

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

// ── Header model switcher ─────────────────────────────────────────────
let _modelSwitching = false;
headerModelSelect.addEventListener('change', async () => {
  const filename = headerModelSelect.value;
  if (_modelSwitching || !filename) return;
  _modelSwitching = true;
  headerModelSelect.disabled = true;
  showToast(`Loading ${filename}…`);
  try {
    const d = await switchModel(filename);
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

async function populateHeaderModels() {
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
    headerModeSelect.value = d.inference_mode || 'standard';
    syncModeFromHealth(d.inference_mode);
    if (d.uptime_s != null) {
      uptimeDisp.textContent = `uptime: ${Math.floor(d.uptime_s)}s`;
    }
    if (d.context_max) {
      state.contextMax = d.context_max;
      updateContextMeter();
    }
    if (d.loading) {
      headerModelSelect.disabled = true;
      headerModeSelect.disabled  = true;
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
populateHeaderModels();

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
    authTabs.forEach(t => t.classList.toggle('active', t === tab));
    authSubmit.textContent = _authMode === 'login' ? 'Login' : 'Register';
    authError.textContent = '';
  });
});

authForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  authError.textContent = '';
  const username = authUsername.value.trim();
  const password = authPassword.value;
  if (!username || !password) { authError.textContent = 'Fill in all fields'; return; }
  authSubmit.disabled = true;
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
  } finally {
    authSubmit.disabled = false;
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
