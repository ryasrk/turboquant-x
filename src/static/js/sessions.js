/** Session management — sidebar with chat history. */

import { authHeaders, isLoggedIn } from './auth.js';
import { state } from './state.js';
import { showConfirm } from './ui.js';

let _currentSessionId = null;
let _sessions = [];

export function getCurrentSessionId() { return _currentSessionId; }
export function setCurrentSessionId(id) { _currentSessionId = id; }

// ── API calls ───────────────────────────────────────────────────────

export async function fetchSessions() {
  if (!isLoggedIn()) return [];
  const r = await fetch('/v1/sessions', { headers: authHeaders() });
  if (!r.ok) return [];
  _sessions = await r.json();
  return _sessions;
}

export async function createSession(title) {
  const r = await fetch('/v1/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ title }),
  });
  if (!r.ok) throw new Error('Failed to create session');
  const session = await r.json();
  _currentSessionId = session.id;
  return session;
}

export async function renameSession(sessionId, title) {
  const r = await fetch(`/v1/sessions/${sessionId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ title }),
  });
  return r.ok;
}

export async function deleteSession(sessionId) {
  const r = await fetch(`/v1/sessions/${sessionId}`, {
    method: 'DELETE',
    headers: authHeaders(),
  });
  if (r.ok && _currentSessionId === sessionId) {
    _currentSessionId = null;
  }
  return r.ok;
}

export async function fetchMessages(sessionId) {
  const r = await fetch(`/v1/sessions/${sessionId}/messages`, {
    headers: authHeaders(),
  });
  if (!r.ok) return [];
  return r.json();
}

export async function saveMessage(role, content) {
  if (!isLoggedIn() || !_currentSessionId) return null;
  const r = await fetch(`/v1/sessions/${_currentSessionId}/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ role, content }),
  });
  if (!r.ok) return null;
  return r.json();
}

// ── Sidebar rendering ───────────────────────────────────────────────

const sidebar       = document.getElementById('session-sidebar');
const sessionList   = document.getElementById('session-list');
const newChatBtn    = document.getElementById('new-chat-btn');
const sidebarToggle = document.getElementById('sidebar-toggle');

/** Load a specific session — restore messages in the chat UI. */
export async function loadSession(sessionId, renderFn) {
  _currentSessionId = sessionId;
  const messages = await fetchMessages(sessionId);
  // Re-populate state.history and render bubbles
  state.history = messages.map(m => ({ role: m.role, content: m.content }));
  if (renderFn) renderFn(messages);
  renderSessionList();
}

/** Render the session list in the sidebar. */
export function renderSessionList() {
  if (!sessionList) return;
  sessionList.innerHTML = '';
  for (const s of _sessions) {
    const el = document.createElement('div');
    el.className = 'session-item' + (s.id === _currentSessionId ? ' active' : '');
    el.dataset.id = s.id;

    const titleSpan = document.createElement('span');
    titleSpan.className = 'session-title';
    titleSpan.textContent = s.title;
    titleSpan.title = s.title;

    const delBtn = document.createElement('button');
    delBtn.className = 'session-delete';
    delBtn.textContent = '×';
    delBtn.title = 'Delete session';

    el.appendChild(titleSpan);
    el.appendChild(delBtn);
    sessionList.appendChild(el);
  }
}

/** Initialize sidebar event listeners. */
export function initSidebar(onLoadSession, onNewChat) {
  if (!sidebar) return;

  // Toggle sidebar
  if (sidebarToggle) {
    sidebarToggle.addEventListener('click', () => {
      const isOpen = sidebar.classList.toggle('open');
      document.body.classList.toggle('sidebar-open', isOpen);
    });
  }

  // New chat
  if (newChatBtn) {
    newChatBtn.addEventListener('click', async () => {
      if (onNewChat) await onNewChat();
    });
  }

  // Session click / delete delegation
  if (sessionList) {
    sessionList.addEventListener('click', async (e) => {
      const delBtn = e.target.closest('.session-delete');
      if (delBtn) {
        const item = delBtn.closest('.session-item');
        if (item && await showConfirm('Delete this chat?')) {
          await deleteSession(item.dataset.id);
          await fetchSessions();
          renderSessionList();
        }
        return;
      }
      const item = e.target.closest('.session-item');
      if (item && onLoadSession) {
        await onLoadSession(item.dataset.id);
      }
    });
  }
}

/** Auto-title a session from the first user message. */
export async function autoTitleSession(firstMessage) {
  if (!_currentSessionId) return;
  const title = firstMessage.slice(0, 60) + (firstMessage.length > 60 ? '…' : '');
  await renameSession(_currentSessionId, title);
  const s = _sessions.find(x => x.id === _currentSessionId);
  if (s) s.title = title;
  renderSessionList();
}
