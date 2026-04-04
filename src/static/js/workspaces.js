/**
 * TurboQuant-X — Workspace Portal
 * ES module: workspace CRUD, design lifecycle (SSE), and UI rendering.
 */

import { isLoggedIn, authHeaders, getUser, logout } from './auth.js';

// ── State ──────────────────────────────────────────────────────────────
let workspaces = [];
let activeWsId = null;
let designAbort = null; // AbortController for in-flight SSE

// ── DOM refs (cached on init) ──────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ── API helpers ────────────────────────────────────────────────────────

async function apiFetch(path, opts = {}) {
  const res = await fetch(path, {
    ...opts,
    headers: {
      'Content-Type': 'application/json',
      ...authHeaders(),
      ...(opts.headers || {}),
    },
  });
  if (res.status === 401) {
    logout();
    window.location.href = '/static/login.html';
    return;
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Request failed (${res.status})`);
  }
  if (res.status === 204) return null;
  return res.json();
}

// ── Workspace CRUD ─────────────────────────────────────────────────────

async function fetchWorkspaces() {
  const data = await apiFetch('/v1/workspaces');
  workspaces = data?.workspaces ?? [];
  return workspaces;
}

async function createWorkspace(title) {
  const ws = await apiFetch('/v1/workspaces', {
    method: 'POST',
    body: JSON.stringify({ title }),
  });
  workspaces.unshift(ws);
  return ws;
}

async function deleteWorkspace(id) {
  await apiFetch(`/v1/workspaces/${encodeURIComponent(id)}`, { method: 'DELETE' });
  workspaces = workspaces.filter((w) => w.id !== id);
  if (activeWsId === id) {
    activeWsId = null;
    showEmpty();
  }
}

async function renameWorkspace(id, title) {
  const ws = await apiFetch(`/v1/workspaces/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    body: JSON.stringify({ title }),
  });
  const idx = workspaces.findIndex((w) => w.id === id);
  if (idx !== -1) workspaces[idx] = { ...workspaces[idx], ...ws };
  return ws;
}

// ── Model selector ─────────────────────────────────────────────────────

async function populateModelSelector() {
  const select = $('#ws-model-select');
  if (!select) return;

  try {
    const data = await apiFetch('/v1/cloud-providers');
    const providers = data?.providers ?? [];
    const active = providers.find((p) => p.active);

    for (const p of providers) {
      if (!p.configured) continue;
      const opt = document.createElement('option');
      opt.value = '';  // Will be populated with specific models
      opt.textContent = `${p.display_name}${p.active ? ' (active)' : ''}`;
      opt.disabled = true;
      select.appendChild(opt);

      // Fetch models for this provider
      try {
        const modelData = await apiFetch(`/v1/cloud-providers/${encodeURIComponent(p.name)}/models`);
        const models = modelData?.models ?? [];
        for (const m of models) {
          const mOpt = document.createElement('option');
          const modelId = typeof m === 'string' ? m : m.id || m.name || String(m);
          mOpt.value = modelId;
          mOpt.textContent = `  ${modelId}`;
          select.appendChild(mOpt);
        }
      } catch {
        // Provider may not support model listing
      }
    }

    // If no models were added, add a note
    if (select.options.length <= 1) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No cloud providers configured';
      opt.disabled = true;
      select.appendChild(opt);
    }
  } catch (err) {
    console.warn('Could not load cloud providers:', err);
  }
}

// ── Design lifecycle ───────────────────────────────────────────────────

async function startDesign(workspaceId, prompt, model = null) {
  if (designAbort) designAbort.abort();
  designAbort = new AbortController();

  showDesignProgress();

  try {
    const body = { prompt };
    if (model) body.model = model;

    const res = await fetch(`/v1/workspaces/${encodeURIComponent(workspaceId)}/design`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: JSON.stringify(body),
      signal: designAbort.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Design request failed');
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line in buffer

      let currentEventType = 'message';
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEventType = line.slice(7).trim();
          continue;
        }
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (!raw || raw === '[DONE]') continue;
        try {
          const evt = JSON.parse(raw);
          evt._sseType = currentEventType;
          handleDesignEvent(workspaceId, evt);
        } catch { /* skip malformed events */ }
        currentEventType = 'message'; // reset after consuming
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') return;
    appendProgressLog('error', `Error: ${err.message}`);
    hideDesignProgress();
  } finally {
    designAbort = null;
  }
}

function handleDesignEvent(workspaceId, evt) {
  const type = evt._sseType || evt.type || evt.event || 'info';
  const msg = evt.message || evt.data || '';

  appendProgressLog(type, msg);

  if (type === 'workflow_json' && evt.workflow) {
    showWorkflowPreview(evt.workflow);
  } else if (type === 'complete') {
    hideDesignProgress();
    showActions();
    if (evt.workflow_id) {
      loadN8nIframe(evt.workflow_id);
    }
    // Refresh workspace to get updated status
    refreshActiveWorkspace(workspaceId);
  } else if (type === 'error') {
    hideDesignProgress();
    refreshActiveWorkspace(workspaceId);
  }
}

async function approveDesign(workspaceId) {
  try {
    await apiFetch(`/v1/workspaces/${encodeURIComponent(workspaceId)}/approve`, { method: 'POST' });
  } catch (err) {
    if (err.message?.includes('409') || err.message?.includes('Cannot approve')) {
      console.warn('Workspace already approved');
    } else {
      throw err;
    }
  }
  await refreshActiveWorkspace(workspaceId);
}

async function modifyDesign(workspaceId, prompt) {
  const model = $('#ws-model-select')?.value || null;
  await startDesign(workspaceId, prompt, model);
}

async function rejectDesign(workspaceId) {
  await apiFetch(`/v1/workspaces/${encodeURIComponent(workspaceId)}/reject`, { method: 'POST' });
  hideActions();
  hideN8nIframe();
  await refreshActiveWorkspace(workspaceId);
}

async function fetchDesignHistory(workspaceId) {
  const data = await apiFetch(`/v1/workspaces/${encodeURIComponent(workspaceId)}/designs`);
  return data?.designs ?? [];
}

// ── Refresh helper ─────────────────────────────────────────────────────

async function refreshActiveWorkspace(wsId) {
  await fetchWorkspaces();
  renderWorkspaceList(workspaces);
  const ws = workspaces.find((w) => w.id === wsId);
  if (ws) {
    selectWorkspace(ws);
  }
}

// ── UI: Workspace list (sidebar) ───────────────────────────────────────

function renderWorkspaceList(list) {
  const container = $('#ws-list');
  if (!container) return;

  if (!list.length) {
    container.innerHTML = '<div class="ws-list-empty">No workspaces yet</div>';
    return;
  }

  container.innerHTML = list
    .map(
      (ws) => `
    <div class="ws-card${ws.id === activeWsId ? ' active' : ''}"
         role="listitem"
         data-id="${ws.id}"
         tabindex="0"
         aria-label="${escapeHtml(ws.title)} — ${ws.status || 'draft'}">
      <div class="ws-card-top">
        <span class="ws-card-title">${escapeHtml(ws.title)}</span>
        <span class="ws-status-badge ws-status-${ws.status || 'draft'}">${ws.status || 'draft'}</span>
      </div>
      <div class="ws-card-bottom">
        <span class="ws-card-date">${formatDate(ws.updated_at || ws.created_at)}</span>
        <div class="ws-card-actions">
          <button class="ws-card-action" data-action="rename" data-id="${ws.id}" title="Rename" aria-label="Rename workspace">✎</button>
          <button class="ws-card-action ws-card-action-danger" data-action="delete" data-id="${ws.id}" title="Delete" aria-label="Delete workspace">✕</button>
        </div>
      </div>
    </div>`
    )
    .join('');

  // Click handlers
  container.querySelectorAll('.ws-card').forEach((card) => {
    card.addEventListener('click', (e) => {
      if (e.target.closest('.ws-card-action')) return;
      const id = card.dataset.id;
      const ws = workspaces.find((w) => w.id === id);
      if (ws) selectWorkspace(ws);
    });
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        card.click();
      }
    });
  });

  container.querySelectorAll('.ws-card-action').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const action = btn.dataset.action;
      const id = btn.dataset.id;
      if (action === 'rename') openRenameDialog(id);
      else if (action === 'delete') openDeleteDialog(id);
    });
  });
}

// ── UI: Workspace detail ───────────────────────────────────────────────

function selectWorkspace(ws) {
  activeWsId = ws.id;
  sessionStorage.setItem('tq-active-ws', ws.id);

  // Update sidebar active state
  $$('.ws-card').forEach((c) => c.classList.toggle('active', c.dataset.id === ws.id));

  // Show detail, hide empty
  $('#ws-empty')?.classList.add('hidden');
  $('#ws-detail')?.classList.remove('hidden');
  $('#ws-history-section')?.classList.remove('hidden');

  // Populate header
  $('#ws-detail-title').textContent = ws.title;
  const statusEl = $('#ws-detail-status');
  statusEl.textContent = ws.status || 'draft';
  statusEl.className = `ws-status-badge ws-status-${ws.status || 'draft'}`;
  $('#ws-detail-updated').textContent = ws.updated_at ? `Updated ${formatDate(ws.updated_at)}` : '';

  // Render panels based on status
  renderDesignPanel(ws);

  // Load design history
  fetchDesignHistory(ws.id).then(renderDesignHistory).catch(() => {});

  // Close mobile sidebar
  $('#ws-sidebar')?.classList.remove('open');
  $('#ws-sidebar-toggle')?.setAttribute('aria-expanded', 'false');
}

function renderDesignPanel(ws) {
  const status = ws.status || 'draft';

  // Reset all sub-panels
  hideDesignProgress();
  hideActions();
  hideN8nIframe();

  if (status === 'building') {
    showDesignProgress();
    appendProgressLog('info', 'Design is currently in progress…');
  } else if (status === 'designed') {
    showActions('designed');
    if (ws.workflow_id) loadN8nIframe(ws.workflow_id);
  } else if (status === 'ready' || status === 'approved') {
    showActions('approved');
    // Show prompt input so user can re-design
    $('#ws-prompt-area')?.classList.remove('hidden');
    if (ws.workflow_id) loadN8nIframe(ws.workflow_id);
  } else if (status === 'active') {
    showActions('active');
    if (ws.workflow_id) loadN8nIframe(ws.workflow_id);
  }
  // draft / failed: show prompt input (default state)
}

// ── UI: Design progress (SSE) ──────────────────────────────────────────

function showDesignProgress() {
  $('#ws-design-progress')?.classList.remove('hidden');
  $('#ws-prompt-area')?.classList.add('hidden');
  $('#ws-progress-log').innerHTML = '';
}

function hideDesignProgress() {
  $('#ws-design-progress')?.classList.add('hidden');
  $('#ws-prompt-area')?.classList.remove('hidden');
}

function appendProgressLog(type, message) {
  const log = $('#ws-progress-log');
  if (!log) return;

  const label = $('#ws-progress-label');
  if (label) label.textContent = message;

  const entry = document.createElement('div');
  entry.className = `ws-log-entry ws-log-${type}`;

  const icon = {
    thinking: '◈', building: '⚙', workflow_json: '📋',
    importing: '↑', imported: '✓', import_skipped: '⚠',
    complete: '✓', error: '✗', info: '→',
  }[type] || '→';
  entry.textContent = `${icon} ${message}`;
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
}

function showWorkflowPreview(workflow) {
  let container = $('#ws-workflow-preview');
  if (!container) {
    container = document.createElement('div');
    container.id = 'ws-workflow-preview';
    container.className = 'ws-workflow-preview';
    const detail = $('#ws-detail');
    if (detail) detail.appendChild(container);
  }

  const nodes = workflow.nodes || [];

  // Determine node category for visual styling
  function nodeCategory(type) {
    const t = (type || '').toLowerCase();
    if (t.includes('trigger') || t.includes('webhook') || t.includes('cron') || t.includes('schedule')) return 'trigger';
    if (t.includes('respond') || t.includes('send') || t.includes('output') || t.includes('email')) return 'output';
    return 'action';
  }

  function nodeIcon(category) {
    return { trigger: '⚡', output: '📤', action: '⚙' }[category] || '⚙';
  }

  // Build visual pipeline
  const pipelineHtml = nodes
    .map((n, i) => {
      const cat = nodeCategory(n.type);
      const connector = i < nodes.length - 1
        ? `<div class="ws-node-connector" aria-hidden="true"><svg viewBox="0 0 28 12"><path d="M0 6h20M16 2l6 4-6 4" fill="none" stroke="currentColor" stroke-width="1.5"/></svg></div>`
        : '';
      return `
        <div class="ws-node-card ws-node-${cat}" title="${escapeHtml(n.type)}">
          <div class="ws-node-icon" aria-hidden="true">${nodeIcon(cat)}</div>
          <div class="ws-node-info">
            <div class="ws-node-name">${escapeHtml(n.name)}</div>
            <div class="ws-node-type-label">${escapeHtml(n.type)}</div>
          </div>
        </div>${connector}`;
    })
    .join('');

  container.innerHTML = `
    <h4>Generated Workflow: ${escapeHtml(workflow.name || 'Untitled')}</h4>
    <div class="ws-node-pipeline" role="list" aria-label="Workflow nodes">${pipelineHtml || '<span class="ws-node-empty">No nodes</span>'}</div>
    <details>
      <summary>Raw JSON</summary>
      <pre class="ws-json-preview">${escapeHtml(JSON.stringify(workflow, null, 2))}</pre>
    </details>
  `;
  container.classList.remove('hidden');
}

// ── UI: Action buttons ─────────────────────────────────────────────────

function showActions(state = 'designed') {
  const container = $('#ws-actions');
  container?.classList.remove('hidden');

  const approveBtn = $('#ws-approve-btn');
  const modifyBtn = $('#ws-modify-btn');
  const rejectBtn = $('#ws-reject-btn');

  if (state === 'approved' || state === 'active') {
    // Already approved/active: only modify is available
    approveBtn?.setAttribute('disabled', '');
    rejectBtn?.setAttribute('disabled', '');
    modifyBtn?.removeAttribute('disabled');
  } else {
    // Designed: all actions available
    approveBtn?.removeAttribute('disabled');
    rejectBtn?.removeAttribute('disabled');
    modifyBtn?.removeAttribute('disabled');
  }
}

function hideActions() {
  $('#ws-actions')?.classList.add('hidden');
}

// ── UI: n8n Iframe ─────────────────────────────────────────────────────

function loadN8nIframe(workflowId) {
  const section = $('#ws-n8n-section');
  const iframe = $('#ws-n8n-iframe');
  const loading = $('#ws-iframe-loading');
  if (!section || !iframe) return;

  section.classList.remove('hidden');
  loading?.classList.remove('hidden');

  const src = `/workspace/n8n/workflow/${encodeURIComponent(workflowId)}`;
  iframe.src = src;
  iframe.onload = () => {
    loading?.classList.add('hidden');
  };
}

function hideN8nIframe() {
  const section = $('#ws-n8n-section');
  const iframe = $('#ws-n8n-iframe');
  if (section) section.classList.add('hidden');
  if (iframe) iframe.src = 'about:blank';
}

// ── UI: Design history ─────────────────────────────────────────────────

function renderDesignHistory(designs) {
  const list = $('#ws-history-list');
  if (!list) return;

  if (!designs.length) {
    list.innerHTML = '<div class="ws-history-empty">No designs yet</div>';
    return;
  }

  list.innerHTML = designs
    .map(
      (d) => `
    <div class="ws-history-item" role="listitem">
      <div class="ws-history-item-header">
        <span class="ws-status-badge ws-status-${d.status}">${d.status}</span>
        <span class="ws-history-date">${formatDate(d.created_at)}</span>
      </div>
      <div class="ws-history-prompt">${escapeHtml(d.prompt)}</div>
    </div>`
    )
    .join('');
}

function showEmpty() {
  $('#ws-empty')?.classList.remove('hidden');
  $('#ws-detail')?.classList.add('hidden');
  $('#ws-history-section')?.classList.add('hidden');
  sessionStorage.removeItem('tq-active-ws');
}

// ── Dialogs ────────────────────────────────────────────────────────────

let pendingDialogId = null;

function openRenameDialog(wsId) {
  pendingDialogId = wsId;
  const ws = workspaces.find((w) => w.id === wsId);
  const input = $('#ws-rename-input');
  input.value = ws?.title || '';
  $('#ws-rename-overlay')?.classList.remove('hidden');
  input.focus();
  input.select();
}

function closeRenameDialog() {
  $('#ws-rename-overlay')?.classList.add('hidden');
  pendingDialogId = null;
}

function openDeleteDialog(wsId) {
  pendingDialogId = wsId;
  $('#ws-delete-overlay')?.classList.remove('hidden');
}

function closeDeleteDialog() {
  $('#ws-delete-overlay')?.classList.add('hidden');
  pendingDialogId = null;
}

function openModifyDialog() {
  const input = $('#ws-modify-input');
  if (input) input.value = '';
  $('#ws-modify-overlay')?.classList.remove('hidden');
  input?.focus();
}

function closeModifyDialog() {
  $('#ws-modify-overlay')?.classList.add('hidden');
}

// ── Theme toggle ───────────────────────────────────────────────────────

function initTheme() {
  const saved = localStorage.getItem('tq-theme');
  if (saved) document.documentElement.setAttribute('data-theme', saved);

  $('#theme-toggle')?.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('tq-theme', next);
  });
}

// ── Utilities ──────────────────────────────────────────────────────────

function escapeHtml(str) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(str || ''));
  return div.innerHTML;
}

function formatDate(iso) {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    const now = new Date();
    const diffMs = now - d;
    const diffMin = Math.floor(diffMs / 60000);
    const diffHr = Math.floor(diffMs / 3600000);

    // Relative time for recent dates
    if (diffMin < 1) return 'just now';
    if (diffMin < 60) return `${diffMin}m ago`;
    if (diffHr < 24) return `${diffHr}h ago`;

    // Include year if not current year
    const opts = { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' };
    if (d.getFullYear() !== now.getFullYear()) {
      opts.year = 'numeric';
    }
    return d.toLocaleDateString(undefined, opts);
  } catch {
    return iso;
  }
}

// ── Init ───────────────────────────────────────────────────────────────

export function initWorkspaces() {
  // Auth gate
  if (!isLoggedIn()) {
    window.location.href = '/static/login.html';
    return;
  }

  // Display user
  const user = getUser();
  const userDisp = $('#ws-user-display');
  if (userDisp && user) userDisp.textContent = user.username || '';

  initTheme();

  // Load workspaces
  fetchWorkspaces()
    .then((list) => {
      renderWorkspaceList(list);
      // Restore previously active workspace from session
      const savedId = sessionStorage.getItem('tq-active-ws');
      if (savedId) {
        const ws = list.find((w) => w.id === savedId);
        if (ws) selectWorkspace(ws);
      }
    })
    .catch((err) => console.error('Failed to load workspaces:', err));

  // Populate model selector
  populateModelSelector();

  // ── Event bindings ──────────────────────────────────────────────────

  // New workspace
  $('#ws-new-btn')?.addEventListener('click', async () => {
    try {
      const ws = await createWorkspace('Untitled Workspace');
      renderWorkspaceList(workspaces);
      selectWorkspace(ws);
    } catch (err) {
      console.error('Create workspace failed:', err);
    }
  });

  // Design button
  $('#ws-design-btn')?.addEventListener('click', () => {
    const prompt = $('#ws-prompt-input')?.value?.trim();
    if (!prompt || !activeWsId) return;
    const model = $('#ws-model-select')?.value || null;
    startDesign(activeWsId, prompt, model);
  });

  // Approve / Modify / Reject
  $('#ws-approve-btn')?.addEventListener('click', (e) => {
    if (e.target.disabled || e.target.hasAttribute('disabled')) return;
    if (activeWsId) approveDesign(activeWsId);
  });

  $('#ws-modify-btn')?.addEventListener('click', () => {
    if (!activeWsId) return;
    openModifyDialog();
  });

  $('#ws-reject-btn')?.addEventListener('click', () => {
    if (activeWsId) rejectDesign(activeWsId);
  });

  // Design history toggle
  $('#ws-history-toggle')?.addEventListener('click', () => {
    const list = $('#ws-history-list');
    const btn = $('#ws-history-toggle');
    const expanded = btn?.getAttribute('aria-expanded') === 'true';
    btn?.setAttribute('aria-expanded', String(!expanded));
    list?.classList.toggle('collapsed');
  });

  // Rename dialog
  $('#ws-rename-cancel')?.addEventListener('click', closeRenameDialog);
  $('#ws-rename-confirm')?.addEventListener('click', async () => {
    const title = $('#ws-rename-input')?.value?.trim();
    if (!title || !pendingDialogId) return;
    try {
      await renameWorkspace(pendingDialogId, title);
      renderWorkspaceList(workspaces);
      if (activeWsId === pendingDialogId) {
        $('#ws-detail-title').textContent = title;
      }
    } catch (err) {
      console.error('Rename failed:', err);
    }
    closeRenameDialog();
  });
  $('#ws-rename-input')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      $('#ws-rename-confirm')?.click();
    } else if (e.key === 'Escape') {
      closeRenameDialog();
    }
  });

  // Delete dialog
  $('#ws-delete-cancel')?.addEventListener('click', closeDeleteDialog);
  $('#ws-delete-confirm')?.addEventListener('click', async () => {
    if (!pendingDialogId) return;
    try {
      await deleteWorkspace(pendingDialogId);
      renderWorkspaceList(workspaces);
    } catch (err) {
      console.error('Delete failed:', err);
    }
    closeDeleteDialog();
  });

  // Close overlays on backdrop click
  ['ws-rename-overlay', 'ws-delete-overlay', 'ws-modify-overlay'].forEach((id) => {
    $(`#${id}`)?.addEventListener('click', (e) => {
      if (e.target.id === id) {
        $(`#${id}`)?.classList.add('hidden');
        pendingDialogId = null;
      }
    });
  });

  // Modify dialog
  $('#ws-modify-cancel')?.addEventListener('click', closeModifyDialog);
  $('#ws-modify-confirm')?.addEventListener('click', async () => {
    const instruction = $('#ws-modify-input')?.value?.trim();
    if (!instruction || !activeWsId) return;
    closeModifyDialog();
    modifyDesign(activeWsId, instruction);
  });
  $('#ws-modify-input')?.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      $('#ws-modify-confirm')?.click();
    } else if (e.key === 'Escape') {
      closeModifyDialog();
    }
  });

  // Close overlays on Escape
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeRenameDialog();
      closeDeleteDialog();
      closeModifyDialog();
    }
  });

  // Mobile sidebar toggle
  $('#ws-sidebar-toggle')?.addEventListener('click', () => {
    const sidebar = $('#ws-sidebar');
    const btn = $('#ws-sidebar-toggle');
    const open = sidebar?.classList.toggle('open');
    btn?.setAttribute('aria-expanded', String(!!open));
  });

  $('#sidebar-collapse-btn')?.addEventListener('click', () => {
    $('#ws-sidebar')?.classList.remove('open');
    $('#ws-sidebar-toggle')?.setAttribute('aria-expanded', 'false');
  });

  // Logout
  $('#ws-logout-btn')?.addEventListener('click', () => {
    logout();
    window.location.href = '/static/login.html';
  });

  // Ctrl+Enter in prompt textarea
  $('#ws-prompt-input')?.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      $('#ws-design-btn')?.click();
    }
  });
}

// ── Auto-init on DOM ready ─────────────────────────────────────────────
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initWorkspaces);
} else {
  initWorkspaces();
}
