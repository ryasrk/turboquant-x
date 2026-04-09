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
  try {
    await apiFetch(`/v1/workspaces/${encodeURIComponent(id)}`, { method: 'DELETE' });
  } catch (err) {
    // 404 = already gone — treat as success
    if (!err.message?.includes('not found') && !err.message?.includes('404')) throw err;
  }
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

  // Clear existing options except the default
  while (select.options.length > 1) select.remove(1);

  try {
    const data = await apiFetch('/v1/workspaces/models');
    const groups = data?.groups ?? [];

    for (const g of groups) {
      // Group header (disabled)
      const header = document.createElement('option');
      header.disabled = true;
      header.textContent = `── ${g.display_name} ──`;
      select.appendChild(header);

      const models = g.models ?? [];
      if (models.length === 0) {
        // Provider with no specific model listed — add a generic option
        const opt = document.createElement('option');
        opt.value = JSON.stringify({ provider: g.provider });
        opt.textContent = `  ${g.display_name} (default)`;
        select.appendChild(opt);
      } else {
        for (const m of models) {
          const opt = document.createElement('option');
          opt.value = JSON.stringify({ provider: g.provider, model: m.id });
          opt.textContent = `  ${m.label || m.name || m.id}`;
          select.appendChild(opt);
        }
      }
    }

    if (groups.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No models available';
      opt.disabled = true;
      select.appendChild(opt);
    }
  } catch (err) {
    console.warn('Could not load workspace models:', err);
  }
}

/** Parse the model selector value into {provider, model} */
function getSelectedModel() {
  const val = $('#ws-model-select')?.value;
  if (!val) return { provider: null, model: null };
  try {
    return JSON.parse(val);
  } catch {
    // Legacy: plain model string
    return { provider: null, model: val };
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
    // Parse model selector for provider info
    const sel = getSelectedModel();
    if (sel.provider) body.provider = sel.provider;
    if (sel.model) body.model = sel.model;

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
    const result = await apiFetch(`/v1/workspaces/${encodeURIComponent(workspaceId)}/approve`, { method: 'POST' });
    // Check for missing credentials response
    if (result?.missing_credentials?.length > 0) {
      const types = [...new Set(result.missing_credentials.map(m => m.cred_type))];
      alert(`Approved, but cannot activate yet.\n\nMissing credentials: ${types.join(', ')}\n\nUse the 🔑 Credentials button to set them up, then re-deploy.`);
      // Auto-open credential panel
      checkCredentials(workspaceId);
    }
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
  await startDesign(workspaceId, prompt);
}

async function rejectDesign(workspaceId) {
  await apiFetch(`/v1/workspaces/${encodeURIComponent(workspaceId)}/reject`, { method: 'POST' });
  hideActions();
  hideN8nIframe();
  await refreshActiveWorkspace(workspaceId);
}

async function removeWorkflow(workspaceId) {
  const ws = workspaces.find((w) => w.id === workspaceId);
  const hasWf = ws?.n8n_workflow_id;
  const msg = hasWf
    ? 'Remove the linked workflow and reset this workspace to draft? The workflow will also be deleted from n8n if reachable.'
    : 'Reset this workspace to draft? The current design will be cleared.';
  if (!confirm(msg)) return;
  try {
    await apiFetch(`/v1/workspaces/${encodeURIComponent(workspaceId)}/workflow`, { method: 'DELETE' });
    hideActions();
    hideN8nIframe();
    // Clear workflow preview
    const preview = $('#ws-workflow-preview');
    if (preview) preview.classList.add('hidden');
    await refreshActiveWorkspace(workspaceId);
  } catch (err) {
    alert(`Failed to remove workflow: ${err.message}`);
  }
}

async function redeployWorkflow(workspaceId) {
  if (!confirm('Re-deploy the design to n8n? This will update the n8n workflow with the latest design data.')) return;
  const btn = $('#ws-redeploy-btn');
  if (btn) { btn.disabled = true; btn.textContent = '⟳ Deploying…'; }
  try {
    const result = await apiFetch(`/v1/workspaces/${encodeURIComponent(workspaceId)}/redeploy`, { method: 'POST' });
    let msg = `${result.message || 'Redeployed successfully'} (${result.node_count} nodes)`;
    if (result.missing_credentials?.length > 0) {
      const types = [...new Set(result.missing_credentials.map(m => m.cred_type))];
      msg += `\n\n⚠ Missing credentials: ${types.join(', ')}\nUse the 🔑 Credentials button to set them up.`;
      // Auto-open credential panel
      setTimeout(() => checkCredentials(workspaceId), 300);
    }
    alert(msg);
    await refreshActiveWorkspace(workspaceId);
  } catch (err) {
    alert(`Redeploy failed: ${err.message}`);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = '⟳ Redeploy'; }
  }
}

// ── Credential setup ───────────────────────────────────────────────────

/** Known credential field schemas */
const CREDENTIAL_FIELDS = {
  telegramApi:     [{ key: 'accessToken', label: 'Bot Token',       type: 'password', placeholder: '123456:ABC-DEF1234ghIkl-zyx57W2v' }],
  openAiApi:       [{ key: 'apiKey',      label: 'API Key',         type: 'password', placeholder: 'sk-...' }],
  slackApi:        [{ key: 'accessToken', label: 'Bot Token',       type: 'password', placeholder: 'xoxb-...' }],
  httpHeaderAuth:  [{ key: 'name',  label: 'Header Name',  type: 'text',     placeholder: 'Authorization' },
                    { key: 'value', label: 'Header Value', type: 'password', placeholder: 'Bearer ...' }],
  httpBasicAuth:   [{ key: 'user',     label: 'Username', type: 'text' },
                    { key: 'password', label: 'Password', type: 'password' }],
  githubApi:       [{ key: 'accessToken', label: 'Personal Access Token', type: 'password', placeholder: 'ghp_...' }],
  notionApi:       [{ key: 'apiKey',      label: 'Integration Token',     type: 'password', placeholder: 'secret_...' }],
  discordApi:      [{ key: 'botToken',    label: 'Bot Token',             type: 'password', placeholder: '' }],
  postgresApi:     [{ key: 'host', label: 'Host', type: 'text', placeholder: 'localhost' },
                    { key: 'database', label: 'Database', type: 'text' },
                    { key: 'user', label: 'User', type: 'text' },
                    { key: 'password', label: 'Password', type: 'password' },
                    { key: 'port', label: 'Port', type: 'number', placeholder: '5432' }],
  gmailOAuth2:     [{ key: 'clientId', label: 'Client ID', type: 'text' },
                    { key: 'clientSecret', label: 'Client Secret', type: 'password' },
                    { key: 'refreshToken', label: 'Refresh Token', type: 'password' }],
  smtp:            [{ key: 'host', label: 'Host', type: 'text', placeholder: 'smtp.gmail.com' },
                    { key: 'port', label: 'Port', type: 'number', placeholder: '587' },
                    { key: 'user', label: 'User', type: 'text' },
                    { key: 'password', label: 'Password', type: 'password' }],
};

async function checkCredentials(workspaceId) {
  const panel = $('#ws-cred-panel');
  const status = $('#ws-cred-status');
  const list = $('#ws-cred-list');
  if (!panel || !status || !list) return;

  panel.classList.remove('hidden');
  status.textContent = 'Checking credentials…';
  list.innerHTML = '';
  $('#ws-cred-form')?.classList.add('hidden');

  try {
    const result = await apiFetch(`/v1/workspaces/${encodeURIComponent(workspaceId)}/credentials/check`);

    if (result.ok && result.missing?.length === 0) {
      status.innerHTML = '<span class="ws-cred-ok">✓ All credentials configured</span>';
      renderCredentialList(result.required, result.available, []);
      return;
    }

    if (result.missing?.length > 0) {
      status.innerHTML = `<span class="ws-cred-warn">⚠ ${result.missing.length} missing credential(s) — workflow cannot activate until these are created</span>`;
      renderCredentialList(result.required, result.available, result.missing);
    } else {
      status.innerHTML = '<span class="ws-cred-ok">✓ No credentials required</span>';
    }
  } catch (err) {
    status.innerHTML = `<span class="ws-cred-err">Error: ${err.message}</span>`;
  }
}

function renderCredentialList(required, available, missing) {
  const list = $('#ws-cred-list');
  if (!list) return;
  list.innerHTML = '';

  if (!required?.length) {
    list.innerHTML = '<p class="ws-cred-empty">No credentials required by this workflow.</p>';
    return;
  }

  // Deduplicate missing by cred_type
  const missingTypes = new Set(missing.map(m => m.cred_type));

  const table = document.createElement('table');
  table.className = 'ws-cred-table';
  table.innerHTML = `
    <thead><tr><th>Node</th><th>Credential Type</th><th>Status</th><th>Action</th></tr></thead>
    <tbody></tbody>
  `;
  const tbody = table.querySelector('tbody');

  for (const req of required) {
    const isMissing = missingTypes.has(req.cred_type);
    const tr = document.createElement('tr');
    tr.className = isMissing ? 'ws-cred-missing-row' : 'ws-cred-ok-row';
    tr.innerHTML = `
      <td>${escapeHtml(req.node_name)}</td>
      <td><code>${escapeHtml(req.cred_type)}</code></td>
      <td>${isMissing ? '<span class="ws-cred-warn">⚠ Missing</span>' : '<span class="ws-cred-ok">✓ OK</span>'}</td>
      <td>${isMissing ? `<button class="ws-cred-add-btn" data-type="${escapeHtml(req.cred_type)}" data-node="${escapeHtml(req.node_name)}">+ Add</button>` : '—'}</td>
    `;
    tbody.appendChild(tr);
  }

  list.appendChild(table);

  // Wire add buttons
  list.querySelectorAll('.ws-cred-add-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      showCredentialForm(btn.dataset.type, btn.dataset.node);
    });
  });
}

function showCredentialForm(credType, nodeName) {
  const form = $('#ws-cred-form');
  const title = $('#ws-cred-form-title');
  const typeInput = $('#ws-cred-type');
  const nameInput = $('#ws-cred-name');
  const fieldsDiv = $('#ws-cred-fields');
  const formStatus = $('#ws-cred-form-status');
  if (!form || !fieldsDiv) return;

  form.classList.remove('hidden');
  if (title) title.textContent = `Add ${credType} for ${nodeName}`;
  if (typeInput) typeInput.value = credType;
  if (nameInput) nameInput.value = `${nodeName} - ${credType}`;
  if (formStatus) formStatus.textContent = '';
  fieldsDiv.innerHTML = '';

  const fields = CREDENTIAL_FIELDS[credType] || [{ key: 'apiKey', label: 'API Key / Token', type: 'password' }];

  for (const field of fields) {
    const label = document.createElement('label');
    label.className = 'ws-cred-label';
    label.innerHTML = `${escapeHtml(field.label)}
      <input type="${field.type || 'text'}" class="ws-cred-input ws-cred-field" data-key="${field.key}"
             placeholder="${field.placeholder || ''}" autocomplete="off" />`;
    fieldsDiv.appendChild(label);
  }

  // Scroll form into view
  form.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function saveCredential() {
  const credType = $('#ws-cred-type')?.value;
  const credName = $('#ws-cred-name')?.value?.trim();
  const formStatus = $('#ws-cred-form-status');
  if (!credType || !credName || !activeWsId) return;

  // Collect field values
  const data = {};
  document.querySelectorAll('.ws-cred-field').forEach(input => {
    const key = input.dataset.key;
    const val = input.value.trim();
    if (key && val) data[key] = val;
  });

  if (Object.keys(data).length === 0) {
    if (formStatus) formStatus.innerHTML = '<span class="ws-cred-err">Please fill in at least one field</span>';
    return;
  }

  const saveBtn = $('#ws-cred-save-btn');
  if (saveBtn) { saveBtn.disabled = true; saveBtn.textContent = 'Creating…'; }

  try {
    const result = await apiFetch(`/v1/workspaces/${encodeURIComponent(activeWsId)}/credentials/create`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cred_type: credType, name: credName, data }),
    });

    if (formStatus) {
      const linked = result.linked_nodes?.length ? ` (auto-linked to: ${result.linked_nodes.join(', ')})` : '';
      formStatus.innerHTML = `<span class="ws-cred-ok">✓ Created credential ${escapeHtml(result.credential_id)}${linked}</span>`;
    }

    // Re-check credentials to update the table
    setTimeout(() => checkCredentials(activeWsId), 500);
  } catch (err) {
    if (formStatus) formStatus.innerHTML = `<span class="ws-cred-err">Failed: ${escapeHtml(err.message)}</span>`;
  } finally {
    if (saveBtn) { saveBtn.disabled = false; saveBtn.textContent = '💾 Create Credential'; }
  }
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

let chatHistory = [];
let chatStreaming = false;

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

  // Load execution history if workflow linked
  if (ws.n8n_workflow_id) {
    loadExecutions(ws.id);
  } else {
    $('#ws-exec-section')?.classList.add('hidden');
  }

  // Show workspace agent chat
  $('#ws-chat-section')?.classList.remove('hidden');
  chatHistory = [];
  const msgContainer = $('#ws-chat-messages');
  if (msgContainer) msgContainer.innerHTML = '';

  // Close mobile sidebar
  $('#ws-sidebar')?.classList.remove('open');
  $('#ws-sidebar-toggle')?.setAttribute('aria-expanded', 'false');
}

function renderDesignPanel(ws) {
  const status = ws.status || 'draft';
  const wfId = ws.n8n_workflow_id || ws.workflow_id;

  // Reset all sub-panels
  hideDesignProgress();
  hideActions();
  hideN8nIframe();

  // Clear stale workflow preview from previous workspace
  const preview = $('#ws-workflow-preview');
  if (preview) { preview.innerHTML = ''; preview.classList.add('hidden'); }

  // Update "Open in n8n" link
  const openLink = $('#ws-n8n-open');
  if (openLink && wfId) {
    openLink.href = `/workspace/n8n-login?next=/workspace/n8n/workflow/${encodeURIComponent(wfId)}`;
    openLink.classList.remove('hidden');
  } else if (openLink) {
    openLink.classList.add('hidden');
  }

  if (status === 'building') {
    showDesignProgress();
    appendProgressLog('info', 'Design is currently in progress…');
  } else if (status === 'designed') {
    showActions('designed');
    if (wfId) loadN8nIframe(wfId);
    loadLatestDesignPreview(ws.id);
  } else if (status === 'ready' || status === 'approved') {
    showActions('approved');
    // Show prompt input so user can re-design
    $('#ws-prompt-area')?.classList.remove('hidden');
    if (wfId) loadN8nIframe(wfId);
    loadLatestDesignPreview(ws.id);
  } else if (status === 'active') {
    showActions('active');
    if (wfId) loadN8nIframe(wfId);
    loadLatestDesignPreview(ws.id);
  }
  // draft / failed: clear prompt and show prompt input (default state)
  if (status === 'draft' || status === 'failed') {
    const promptInput = $('#ws-prompt-input');
    if (promptInput) promptInput.value = '';
  }
}

// ── UI: Load latest design preview (on page load / workspace select) ───

async function loadLatestDesignPreview(workspaceId) {
  try {
    const data = await apiFetch(`/v1/workspaces/${encodeURIComponent(workspaceId)}/designs`);
    const designs = data?.designs ?? [];
    if (designs.length > 0) {
      // Populate the prompt textarea with the latest design prompt
      const promptInput = $('#ws-prompt-input');
      if (promptInput && designs[0].prompt) {
        promptInput.value = designs[0].prompt;
      }
      // Show workflow preview if design has result data
      if (designs[0].result_data) {
        const workflow = JSON.parse(designs[0].result_data);
        showWorkflowPreview(workflow);
      }
    }
  } catch (err) {
    console.warn('Failed to load design preview:', err);
  }
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
    <details id="ws-json-editor-details">
      <summary>Raw JSON Editor</summary>
      <textarea id="ws-json-editor" class="ws-json-editor" spellcheck="false">${escapeHtml(JSON.stringify(workflow, null, 2))}</textarea>
      <div class="ws-json-actions">
        <button id="ws-json-save" class="ws-json-save-btn">💾 Save JSON</button>
        <button id="ws-json-format" class="ws-json-format-btn">{ } Format</button>
        <span id="ws-json-status" class="ws-json-status"></span>
      </div>
    </details>
  `;
  container.classList.remove('hidden');

  // Wire up JSON editor buttons
  $('#ws-json-format')?.addEventListener('click', () => {
    const editor = $('#ws-json-editor');
    const status = $('#ws-json-status');
    if (!editor) return;
    try {
      const parsed = JSON.parse(editor.value);
      editor.value = JSON.stringify(parsed, null, 2);
      if (status) { status.textContent = 'Formatted ✓'; status.className = 'ws-json-status ws-json-ok'; }
    } catch (e) {
      if (status) { status.textContent = `Invalid JSON: ${e.message}`; status.className = 'ws-json-status ws-json-err'; }
    }
  });

  $('#ws-json-save')?.addEventListener('click', async () => {
    const editor = $('#ws-json-editor');
    const status = $('#ws-json-status');
    if (!editor || !activeWsId) return;

    let parsed;
    try {
      parsed = JSON.parse(editor.value);
    } catch (e) {
      if (status) { status.textContent = `Invalid JSON: ${e.message}`; status.className = 'ws-json-status ws-json-err'; }
      return;
    }

    const btn = $('#ws-json-save');
    if (btn) { btn.disabled = true; btn.textContent = '💾 Saving…'; }
    try {
      await apiFetch(`/v1/workspaces/${encodeURIComponent(activeWsId)}/design/json`, {
        method: 'PUT',
        body: JSON.stringify({ workflow_json: parsed }),
      });
      if (status) { status.textContent = 'Saved ✓'; status.className = 'ws-json-status ws-json-ok'; }
      // Refresh preview with new data
      showWorkflowPreview(parsed);
    } catch (err) {
      if (status) { status.textContent = `Save failed: ${err.message}`; status.className = 'ws-json-status ws-json-err'; }
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = '💾 Save JSON'; }
    }
  });
}

// ── UI: Action buttons ─────────────────────────────────────────────────

function showActions(state = 'designed') {
  const container = $('#ws-actions');
  container?.classList.remove('hidden');

  const approveBtn = $('#ws-approve-btn');
  const modifyBtn = $('#ws-modify-btn');
  const rejectBtn = $('#ws-reject-btn');

  // Show/hide remove workflow button — available when not in draft
  const removeBtn = $('#ws-remove-workflow-btn');
  const redeployBtn = $('#ws-redeploy-btn');
  const ws = workspaces.find((w) => w.id === activeWsId);
  if (removeBtn) {
    if (ws && ws.status !== 'draft') {
      removeBtn.classList.remove('hidden');
    } else {
      removeBtn.classList.add('hidden');
    }
  }
  // Show redeploy button when workspace has a design (designed/approved/active)
  if (redeployBtn) {
    if (ws && ['designed', 'approved', 'active'].includes(ws.status)) {
      redeployBtn.classList.remove('hidden');
    } else {
      redeployBtn.classList.add('hidden');
    }
  }

  // Show credential button when workspace has a design
  const credBtn = $('#ws-cred-check-btn');
  if (credBtn) {
    if (ws && ['designed', 'approved', 'active'].includes(ws.status)) {
      credBtn.classList.remove('hidden');
    } else {
      credBtn.classList.add('hidden');
    }
  }

  if (state === 'approved' || state === 'active') {
    // Already approved/active: only modify and remove are available
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

// ── UI: Execution history ──────────────────────────────────────────────

async function loadExecutions(workspaceId) {
  const section = $('#ws-exec-section');
  const list = $('#ws-exec-list');
  const empty = $('#ws-exec-empty');
  if (!section) return;

  section.classList.remove('hidden');

  try {
    const data = await apiFetch(`/v1/workspaces/${encodeURIComponent(workspaceId)}/executions`);
    const execs = data?.executions ?? [];

    if (!execs.length) {
      list.innerHTML = '';
      empty?.classList.remove('hidden');
      return;
    }

    empty?.classList.add('hidden');
    list.innerHTML = execs.map((ex) => {
      const finished = ex.finished;
      const statusClass = finished === true ? 'success' : finished === false ? 'error' : 'running';
      const statusLabel = finished === true ? 'success' : finished === false ? 'error' : 'running';
      const icon = finished === true ? '✓' : finished === false ? '✗' : '⋯';
      const started = ex.startedAt ? formatDate(ex.startedAt) : '—';

      return `
        <div class="ws-exec-item" data-exec-id="${escapeHtml(String(ex.id))}" role="listitem" tabindex="0">
          <div class="ws-exec-icon ${statusClass}">${icon}</div>
          <div class="ws-exec-info">
            <div class="ws-exec-id">#${escapeHtml(String(ex.id))}</div>
            <div class="ws-exec-time">${started}</div>
          </div>
          <span class="ws-exec-status ${statusClass}">${statusLabel}</span>
        </div>`;
    }).join('');

    // Click handlers for execution details
    list.querySelectorAll('.ws-exec-item').forEach((item) => {
      item.addEventListener('click', () => {
        const execId = item.dataset.execId;
        if (execId && workspaceId) showExecutionDetail(workspaceId, execId);
      });
    });
  } catch (err) {
    console.error('Failed to load executions:', err);
    list.innerHTML = `<div class="ws-exec-empty">Failed to load executions</div>`;
  }
}

async function showExecutionDetail(workspaceId, executionId) {
  // Remove any existing overlay
  document.querySelector('.ws-exec-detail-overlay')?.remove();

  const overlay = document.createElement('div');
  overlay.className = 'ws-exec-detail-overlay';
  overlay.innerHTML = `
    <div class="ws-exec-detail-dialog">
      <div class="ws-exec-detail-header">
        <span>Execution #${escapeHtml(executionId)}</span>
        <button class="ws-exec-detail-close" aria-label="Close">✕</button>
      </div>
      <div class="ws-exec-detail-body">
        <pre>Loading execution details...</pre>
      </div>
    </div>`;

  document.body.appendChild(overlay);

  // Close handlers
  overlay.querySelector('.ws-exec-detail-close').addEventListener('click', () => overlay.remove());
  overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });

  try {
    const data = await apiFetch(`/v1/workspaces/${encodeURIComponent(workspaceId)}/executions/${encodeURIComponent(executionId)}`);
    overlay.querySelector('pre').textContent = data?.summary || 'No details available';
  } catch (err) {
    overlay.querySelector('pre').textContent = `Error: ${err.message}`;
  }
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
    // Backend stores Unix epoch seconds — convert to ms if needed
    let d;
    if (typeof iso === 'number' && iso < 1e12) {
      d = new Date(iso * 1000);
    } else {
      d = new Date(iso);
    }
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

// ── Template Browser ───────────────────────────────────────────────────

let templateCache = [];
let templateCategories = [];
let templateSearchTimer = null;

async function openTemplateBrowser() {
  const overlay = $('#ws-template-overlay');
  if (!overlay) return;
  overlay.classList.remove('hidden');

  const listEl = $('#ws-template-list');
  const countEl = $('#ws-template-count');
  const catSelect = $('#ws-template-category');
  const searchInput = $('#ws-template-search');

  if (listEl) listEl.innerHTML = '<div class="ws-template-loading"><span class="ws-spinner"></span> Loading templates…</div>';
  if (searchInput) searchInput.value = '';

  try {
    // Load categories + templates in parallel
    const [catData, tplData] = await Promise.all([
      apiFetch('/v1/workspaces/templates/categories'),
      apiFetch('/v1/workspaces/templates?limit=2000'),
    ]);

    templateCategories = catData?.categories ?? [];
    templateCache = tplData?.templates ?? [];

    // Populate category selector
    if (catSelect) {
      while (catSelect.options.length > 1) catSelect.remove(1);
      for (const cat of templateCategories) {
        const opt = document.createElement('option');
        opt.value = cat;
        opt.textContent = cat;
        catSelect.appendChild(opt);
      }
    }

    renderTemplateList(templateCache);
  } catch (err) {
    if (listEl) listEl.innerHTML = `<div class="ws-template-empty">Failed to load templates: ${escapeHtml(err.message)}</div>`;
  }
}

function closeTemplateBrowser() {
  $('#ws-template-overlay')?.classList.add('hidden');
}

async function searchTemplates() {
  const q = $('#ws-template-search')?.value?.trim() || '';
  const cat = $('#ws-template-category')?.value || '';
  const listEl = $('#ws-template-list');

  // Client-side filter on cached data for speed
  let results = templateCache;

  if (cat) {
    results = results.filter((t) => t.category === cat);
  }

  if (q) {
    const terms = q.toLowerCase().split(/\s+/);
    results = results.filter((t) => {
      const searchable = `${t.name} ${t.category} ${(t.node_types || []).join(' ')} ${(t.node_names || []).join(' ')}`.toLowerCase();
      return terms.every((term) => searchable.includes(term));
    });
  }

  renderTemplateList(results);
}

function renderTemplateList(templates) {
  const listEl = $('#ws-template-list');
  const countEl = $('#ws-template-count');
  if (!listEl) return;

  if (countEl) countEl.textContent = `${templates.length} template${templates.length !== 1 ? 's' : ''}`;

  if (!templates.length) {
    listEl.innerHTML = '<div class="ws-template-empty">No templates match your search</div>';
    return;
  }

  listEl.innerHTML = templates.map((t) => {
    const nodeTypes = (t.node_types || []).slice(0, 5);
    const tagsHtml = nodeTypes
      .map((nt) => {
        const short = nt.split('.').pop() || nt;
        return `<span class="ws-tpl-tag">${escapeHtml(short)}</span>`;
      })
      .join('');
    const moreCount = (t.node_types || []).length - 5;
    const moreHtml = moreCount > 0 ? `<span class="ws-tpl-tag ws-tpl-tag-more">+${moreCount}</span>` : '';

    return `
      <div class="ws-tpl-card" role="listitem" data-tpl-id="${t.id}">
        <div class="ws-tpl-card-top">
          <span class="ws-tpl-name">${escapeHtml(t.name)}</span>
          <span class="ws-tpl-category">${escapeHtml(t.category)}</span>
        </div>
        <div class="ws-tpl-tags">${tagsHtml}${moreHtml}</div>
        <div class="ws-tpl-card-bottom">
          <span class="ws-tpl-nodes">${t.node_count || 0} nodes</span>
          <button class="ws-tpl-use-btn" data-tpl-id="${t.id}" data-tpl-name="${escapeHtml(t.name)}">Use Template</button>
        </div>
      </div>`;
  }).join('');

  // Wire up "Use Template" buttons
  listEl.querySelectorAll('.ws-tpl-use-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const tplId = parseInt(btn.dataset.tplId, 10);
      const tplName = btn.dataset.tplName;
      loadTemplateIntoWorkspace(tplId, tplName);
    });
  });
}

async function loadTemplateIntoWorkspace(templateId, templateName) {
  if (!activeWsId) {
    showTplStatus('Please select or create a workspace first.', 'error');
    return;
  }

  const ws = workspaces.find((w) => w.id === activeWsId);
  if (ws && !['draft', 'designed', 'rejected', 'approved', 'failed'].includes(ws.status)) {
    showTplStatus(`Cannot load template: workspace is in "${ws.status}" state.`, 'error');
    return;
  }

  // Show inline confirmation
  showTplConfirm(
    `Load "${templateName}" into this workspace? This will replace the current design.`,
    async () => {
      hideTplConfirm();
      // Find and disable the button
      const btn = $(`.ws-tpl-use-btn[data-tpl-id="${templateId}"]`);
      if (btn) { btn.disabled = true; btn.textContent = 'Loading…'; }

      showTplStatus('Loading template…', 'loading');

      try {
        const result = await apiFetch(`/v1/workspaces/${encodeURIComponent(activeWsId)}/load-template`, {
          method: 'POST',
          body: JSON.stringify({ template_id: templateId }),
        });

        let msg = `✓ "${result.template_name}" loaded (${result.node_count} nodes)`;
        if (result.missing_credentials?.length > 0) {
          const types = [...new Set(result.missing_credentials.map((m) => m.cred_type))];
          msg += ` — ⚠ Missing credentials: ${types.join(', ')}`;
        }
        showTplStatus(msg, 'success');

        // Auto-close browser after short delay and navigate to workspace editor
        setTimeout(async () => {
          closeTemplateBrowser();
          await refreshActiveWorkspace(activeWsId);
        }, 1200);
      } catch (err) {
        showTplStatus(`Failed to load template: ${err.message}`, 'error');
      } finally {
        if (btn) { btn.disabled = false; btn.textContent = 'Use Template'; }
      }
    },
  );
}

function showTplConfirm(text, onConfirm) {
  const el = $('#ws-tpl-confirm');
  const textEl = $('#ws-tpl-confirm-text');
  if (!el || !textEl) return;
  textEl.textContent = text;
  el.classList.remove('hidden');

  const yesBtn = $('#ws-tpl-confirm-yes');
  const noBtn = $('#ws-tpl-confirm-no');

  // Replace buttons to clear old listeners
  const newYes = yesBtn.cloneNode(true);
  const newNo = noBtn.cloneNode(true);
  yesBtn.replaceWith(newYes);
  noBtn.replaceWith(newNo);

  newYes.addEventListener('click', onConfirm);
  newNo.addEventListener('click', hideTplConfirm);
}

function hideTplConfirm() {
  $('#ws-tpl-confirm')?.classList.add('hidden');
}

function showTplStatus(msg, type) {
  const el = $('#ws-tpl-status');
  if (!el) return;
  el.className = `ws-tpl-status ws-tpl-status-${type}`;
  el.textContent = type === 'loading' ? `⏳ ${msg}` : msg;
  el.classList.remove('hidden');
  if (type === 'success' || type === 'error') {
    setTimeout(() => el.classList.add('hidden'), 8000);
  }
}

function hideTplStatus() {
  $('#ws-tpl-status')?.classList.add('hidden');
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

  // ── Template browser bindings ─────────────────────────────────────
  $('#ws-templates-btn')?.addEventListener('click', () => openTemplateBrowser());
  $('#ws-template-close')?.addEventListener('click', () => closeTemplateBrowser());
  $('#ws-template-overlay')?.addEventListener('click', (e) => {
    if (e.target.id === 'ws-template-overlay') closeTemplateBrowser();
  });
  $('#ws-template-search')?.addEventListener('input', () => {
    clearTimeout(templateSearchTimer);
    templateSearchTimer = setTimeout(searchTemplates, 200);
  });
  $('#ws-template-category')?.addEventListener('change', () => searchTemplates());

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
    startDesign(activeWsId, prompt);
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

  $('#ws-remove-workflow-btn')?.addEventListener('click', () => {
    if (activeWsId) removeWorkflow(activeWsId);
  });

  $('#ws-redeploy-btn')?.addEventListener('click', () => {
    if (activeWsId) redeployWorkflow(activeWsId);
  });

  // Credential buttons
  $('#ws-cred-check-btn')?.addEventListener('click', () => {
    if (activeWsId) checkCredentials(activeWsId);
  });
  $('#ws-cred-save-btn')?.addEventListener('click', () => saveCredential());
  $('#ws-cred-cancel-btn')?.addEventListener('click', () => {
    $('#ws-cred-form')?.classList.add('hidden');
  });

  // Design history toggle
  $('#ws-history-toggle')?.addEventListener('click', () => {
    const list = $('#ws-history-list');
    const btn = $('#ws-history-toggle');
    const expanded = btn?.getAttribute('aria-expanded') === 'true';
    btn?.setAttribute('aria-expanded', String(!expanded));
    list?.classList.toggle('collapsed');
  });

  // Execution refresh
  $('#ws-exec-refresh')?.addEventListener('click', () => {
    if (activeWsId) loadExecutions(activeWsId);
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
  ['ws-rename-overlay', 'ws-delete-overlay', 'ws-modify-overlay', 'ws-template-overlay'].forEach((id) => {
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
      closeTemplateBrowser();
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

  // ── Workspace Agent Chat ──────────────────────────────────────
  $('#ws-chat-send')?.addEventListener('click', () => sendChatMessage());
  $('#ws-chat-input')?.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      sendChatMessage();
    }
  });
  $('#ws-chat-clear')?.addEventListener('click', () => {
    chatHistory = [];
    const el = $('#ws-chat-messages');
    if (el) el.innerHTML = '';
  });
}

// ── Workspace Agent Chat ───────────────────────────────────────────────

function appendChatBubble(cls, html) {
  const container = $('#ws-chat-messages');
  if (!container) return null;
  const div = document.createElement('div');
  div.className = `ws-chat-msg ${cls}`;
  div.innerHTML = html;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

function escapeChat(s) {
  const el = document.createElement('span');
  el.textContent = s;
  return el.innerHTML;
}

async function sendChatMessage() {
  if (chatStreaming) return;
  const input = $('#ws-chat-input');
  const msg = input?.value?.trim();
  if (!msg || !activeWsId) return;

  input.value = '';
  chatHistory.push({ role: 'user', content: msg });
  appendChatBubble('user', escapeChat(msg));

  const modelSelect = $('#ws-model-select');
  const sel = getSelectedModel();

  chatStreaming = true;
  const sendBtn = $('#ws-chat-send');
  sendBtn?.classList.add('ws-chat-streaming');
  sendBtn && (sendBtn.disabled = true);

  let assistantText = '';
  let assistantBubble = null;

  try {
    const chatBody = { message: msg, history: chatHistory.slice(-20) };
    if (sel.provider) chatBody.provider = sel.provider;
    if (sel.model) chatBody.model = sel.model;

    const resp = await fetch(`/v1/workspaces/${activeWsId}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...authHeaders(),
      },
      body: JSON.stringify(chatBody),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      appendChatBubble('assistant', `<em>Error: ${escapeChat(err.detail || resp.statusText)}</em>`);
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        const raw = line.slice(5).trim();
        if (raw === '[DONE]') continue;

        let evt;
        try { evt = JSON.parse(raw); } catch { continue; }

        const evtType = evt.type || 'info';

        if (evtType === 'content') {
          const delta = evt.delta || '';
          assistantText += delta;
          if (!assistantBubble) {
            assistantBubble = appendChatBubble('assistant', '');
          }
          assistantBubble.innerHTML = formatMarkdown(assistantText);
        } else if (evtType === 'tool_call') {
          appendChatBubble('tool-call', `🔧 ${escapeChat(evt.name || 'tool')}(${escapeChat(JSON.stringify(evt.arguments || {}).slice(0, 100))})`);
        } else if (evtType === 'tool_result') {
          const content = evt.content || '';
          const short = content.length > 300 ? content.slice(0, 300) + '…' : content;
          appendChatBubble('tool-result', escapeChat(short));
        } else if (evtType === 'error') {
          appendChatBubble('assistant', `<em>Error: ${escapeChat(evt.message || 'Unknown error')}</em>`);
        }
      }
    }

    if (assistantText) {
      chatHistory.push({ role: 'assistant', content: assistantText });
    }
  } catch (err) {
    appendChatBubble('assistant', `<em>Connection error: ${escapeChat(err.message)}</em>`);
  } finally {
    chatStreaming = false;
    sendBtn?.classList.remove('ws-chat-streaming');
    sendBtn && (sendBtn.disabled = false);
    input?.focus();
  }
}

function formatMarkdown(text) {
  // Minimal markdown: code blocks, inline code, bold, italic, links
  return text
    .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br>');
}

// ── Auto-init on DOM ready ─────────────────────────────────────────────
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initWorkspaces);
} else {
  initWorkspaces();
}
