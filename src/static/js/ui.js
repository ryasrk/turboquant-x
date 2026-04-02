/** Shared UI utilities — toast, message bubbles, streaming state. */

import { state } from './state.js';
import { renderMarkdown } from './markdown.js';

const chatEl     = document.getElementById('chat');
const welcomeEl  = document.getElementById('welcome');
const sendBtn    = document.getElementById('send-btn');
const inputEl    = document.getElementById('user-input');
const tokenCount = document.getElementById('token-count');
const notifCard  = document.getElementById('notif-card');
const notifText  = notifCard?.querySelector('.notif-text');
const notifClose = notifCard?.querySelector('.notif-close');

let toastTimer;

if (notifClose) {
  notifClose.addEventListener('click', () => {
    notifCard.classList.remove('show');
    clearTimeout(toastTimer);
  });
}

/**
 * Show a notification card.
 * @param {string} msg — notification text
 * @param {'info'|'error'|'success'} type — visual style
 * @param {number} duration — auto-dismiss in ms (0 = manual only)
 */
export function showToast(msg, type = 'info', duration = 3500) {
  if (!notifCard || !notifText) return;
  notifText.textContent = msg;
  notifCard.classList.remove('show', 'error', 'success');
  if (type === 'error') notifCard.classList.add('error');
  if (type === 'success') notifCard.classList.add('success');
  // Force reflow for re-animation
  void notifCard.offsetWidth;
  notifCard.classList.add('show');
  clearTimeout(toastTimer);
  if (duration > 0) {
    toastTimer = setTimeout(() => notifCard.classList.remove('show'), duration);
  }
}

/* ── Confirm modal helper ──────────────────────────────────────────── */
const confirmOverlay = document.getElementById('confirm-overlay');
const confirmMessage = document.getElementById('confirm-message');
const confirmOk      = document.getElementById('confirm-ok');
const confirmCancel  = document.getElementById('confirm-cancel');

/**
 * Show a styled confirmation modal.
 * @param {string} message — prompt text
 * @param {string} [okLabel='Delete'] — text for the confirm button
 * @returns {Promise<boolean>} resolves true if confirmed, false if cancelled
 */
export function showConfirm(message, okLabel = 'Delete') {
  return new Promise((resolve) => {
    if (!confirmOverlay) { resolve(confirm(message)); return; }
    confirmMessage.textContent = message;
    confirmOk.textContent = okLabel;
    confirmOverlay.classList.remove('hidden');

    function cleanup(result) {
      confirmOverlay.classList.add('hidden');
      confirmOk.removeEventListener('click', onOk);
      confirmCancel.removeEventListener('click', onCancel);
      resolve(result);
    }
    function onOk()     { cleanup(true);  }
    function onCancel() { cleanup(false); }
    confirmOk.addEventListener('click', onOk);
    confirmCancel.addEventListener('click', onCancel);
  });
}

/**
 * Append a chat message bubble.
 * @returns {{ wrapper: HTMLDivElement, bubble: HTMLDivElement }}
 */
export function appendMessage(role, content, streaming = false) {
  if (welcomeEl && welcomeEl.parentNode === chatEl) {
    chatEl.removeChild(welcomeEl);
  }

  const wrapper = document.createElement('div');
  wrapper.className = `message ${role}`;

  const roleEl = document.createElement('div');
  roleEl.className = 'msg-role';
  roleEl.textContent = role === 'user' ? '▶ operator' : '◈ tq-neural';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.textContent = content;

  if (streaming) {
    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    bubble.appendChild(cursor);
  }

  wrapper.appendChild(roleEl);
  wrapper.appendChild(bubble);
  chatEl.appendChild(wrapper);
  chatEl.scrollTop = chatEl.scrollHeight;
  return { wrapper, bubble };
}

/** Render final assistant content.
 *
 * Thoughts are stripped server-side and logged to file.  If any
 * ``<think>`` blocks leak through, they are silently removed.
 */
export function renderAssistantContent(bubble, text) {
  bubble.innerHTML = '';
  // Strip any residual think blocks (server normally handles this)
  const clean = text
    .replace(/<think>[\s\S]*?<\/think>/g, '')     // complete blocks
    .replace(/<think>[\s\S]*$/g, '')               // orphaned open tag
    .replace(/^[\s\S]*?<\/think>/g, '')            // orphaned close tag
    .trim();
  const answerEl = document.createElement('div');
  answerEl.className = 'msg-answer md-content';
  answerEl.innerHTML = renderMarkdown(clean);
  bubble.appendChild(answerEl);
}

export function setStreaming(on) {
  state.streaming = on;
  sendBtn.disabled = on;
  inputEl.disabled = on;
  // Keep SVG icon; just toggle a loading class
  sendBtn.classList.toggle('streaming', on);
}

/**
 * Update bubble during streaming.
 *
 * Chain-of-thought ``<think>`` blocks are stripped server-side and
 * written to ``logs/agent-thoughts.log`` instead of being streamed.
 * This saves client bandwidth and avoids exposing scaffolding tokens
 * in the UI.  If any ``<think>`` tags leak through (e.g. older server),
 * they are silently removed here as a safety net.
 */
export function updateStreamBubble(bubble, text) {
  // Safety-net strip in case server filter missed anything
  const clean = text
    .replace(/<think>[\s\S]*?<\/think>/g, '')     // complete blocks
    .replace(/<think>[\s\S]*$/g, '')               // orphaned open tag
    .replace(/^[\s\S]*?<\/think>/g, '')            // orphaned close tag
    .trim();
  const cursor = bubble.querySelector('.cursor');
  bubble.textContent = clean;
  if (cursor) bubble.appendChild(cursor);
  else {
    const c = document.createElement('span');
    c.className = 'cursor';
    bubble.appendChild(c);
  }
}

export function updateTokenCount() {
  tokenCount.textContent = `tokens: ~${state.totalTokens}`;
}

/**
 * Update the context meter bar and text from state.contextMax / totalTokens.
 * Shows a warning banner when usage exceeds 85%.
 */
export function updateContextMeter() {
  const fill = document.getElementById('context-bar-fill');
  const text = document.getElementById('context-text');
  const warning = document.getElementById('context-warning');
  const warningText = document.getElementById('context-warning-text');
  if (!fill || !text) return;

  const max = state.contextMax;
  if (!max) {
    text.textContent = '—';
    fill.style.width = '0%';
    return;
  }

  // Estimate current usage: rough char/3 heuristic on history
  const histChars = state.history.reduce((s, m) => s + (m.content?.length || 0), 0);
  const estimated = Math.round(histChars / 3) + state.history.length * 4;
  const pct = Math.min(100, Math.round((estimated / max) * 100));

  fill.style.width = pct + '%';
  fill.classList.remove('warn', 'danger');
  if (pct >= 90) fill.classList.add('danger');
  else if (pct >= 70) fill.classList.add('warn');

  // Format: "12.4k / 32.7k"
  const fmt = n => n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n);
  text.textContent = `${fmt(estimated)} / ${fmt(max)}`;

  // Warning banner
  if (warning) {
    if (pct >= 85) {
      const remaining = max - estimated;
      warningText.textContent =
        `⚠ Context ${pct}% full (~${fmt(estimated)} / ${fmt(max)} tokens). ` +
        `~${fmt(Math.max(0, remaining))} remaining. Clear conversation to free space.`;
      warning.classList.add('visible');
    } else {
      warning.classList.remove('visible');
    }
  }
}

export function getChatEl()    { return chatEl; }
export function getWelcomeEl() { return welcomeEl; }
