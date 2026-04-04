/** Agent mode — tool call rendering and state management. */

import { state } from './state.js';

/** Tool emoji map */
const TOOL_ICONS = {
  // File System
  read_file: '📁',
  write_file: '✏️',
  list_dir: '📂',
  find_files: '🔎',
  replace_in_file: '🔄',
  // System
  exec_shell: '💻',
  get_env: '⚙️',
  current_time: '🕐',
  // Terminal (requires approval)
  terminal_exec: '🖥️',
  // Web
  web_search: '🔍',
  fetch_webpage: '🌐',
  http_request: '📡',
  // Code Analysis
  grep_code: '🔬',
  python_eval: '🐍',
  count_lines: '📊',
  // Data & Math
  json_query: '📋',
  calculate: '🧮',
  csv_read: '📈',
  // RAG / Document
  read_pdf: '📄',
  index_document: '🌲',
  search_document: '🔎',
};

/** Create a tool call card element */
export function createToolCallCard(name, args) {
  const card = document.createElement('div');
  card.className = 'tool-card tool-call';
  
  const isMcp = name.startsWith('mcp_');
  const icon = isMcp ? '🔌' : (TOOL_ICONS[name] || '🔧');
  const argSummary = Object.entries(args)
    .map(([k, v]) => {
      let display;
      if (typeof v === 'string') {
        display = v.length > 80 ? v.slice(0, 80) + '...' : v;
      } else if (typeof v === 'object' && v !== null) {
        const json = JSON.stringify(v);
        display = json.length > 80 ? json.slice(0, 80) + '...' : json;
      } else {
        display = String(v);
      }
      return `<span class="tool-arg-key">${escapeHtml(k)}:</span> ${escapeHtml(display)}`;
    })
    .join(', ');

  card.innerHTML = `
    <div class="tool-header">
      <span class="tool-icon">${icon}</span>
      <span class="tool-name">${escapeHtml(name)}</span>
      <span class="tool-status spinning">⏳</span>
    </div>
    ${argSummary ? `<div class="tool-args">${argSummary}</div>` : ''}
  `;
  return card;
}

/** Create a tool result card (appended below the call card) */
export function createToolResultCard(name, content) {
  const card = document.createElement('div');
  card.className = 'tool-card tool-result';
  
  // Collapsible content
  const preview = content.length > 200 ? content.slice(0, 200) + '...' : content;
  card.innerHTML = `
    <div class="tool-result-header">
      <span class="tool-result-label">Result</span>
      <span class="tool-expand">▶</span>
    </div>
    <pre class="tool-result-preview">${linkifyText(preview)}</pre>
    <pre class="tool-result-full">${linkifyText(content)}</pre>
  `;
  card.querySelector('.tool-result-header').addEventListener('click', () => {
    card.classList.toggle('expanded');
  });
  return card;
}

/** Mark a tool call card as completed */
export function markToolCompleted(card) {
  const status = card.querySelector('.tool-status');
  if (status) {
    status.textContent = '✅';
    status.classList.remove('spinning');
  }
}

/** Create iteration header */
export function createIterationHeader(iteration) {
  const el = document.createElement('div');
  el.className = 'agent-iteration';
  el.textContent = `Step ${iteration}`;
  return el;
}

/** Create agent summary */
export function createAgentSummary(data) {
  const el = document.createElement('div');
  el.className = 'agent-summary';
  const tools = [...new Set(data.tools_used || [])];
  el.innerHTML = `
    <span class="agent-summary-label">🤖 Agent</span>
    <span>${data.iterations} steps</span>
    <span>${tools.map(t => (TOOL_ICONS[t] || '🔧') + ' ' + escapeHtml(t)).join(', ')}</span>
  `;
  return el;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

/** Escape HTML then convert bare URLs to clickable links. */
function linkifyText(str) {
  let out = escapeHtml(str);
  // Markdown links: [text](url) — supports both absolute and relative URLs
  out = out.replace(
    /\[([^\]]+)\]\(((?:https?:\/\/|\/)[^)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
  );
  // Bare URLs not already inside an <a> tag
  out = out.replace(
    /(?<!href="|">)(https?:\/\/[^\s<"']+)/g,
    '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
  );
  return out;
}

export function isAgentEnabled() {
  return state.agent === true;
}

export function toggleAgent() {
  state.agent = !state.agent;
  return state.agent;
}

/** Risk level configuration */
const RISK_CONFIG = {
  low:      { emoji: '🟢', label: 'Low Risk',      color: 'var(--success, #4caf50)' },
  medium:   { emoji: '🟡', label: 'Medium Risk',   color: '#ff9800' },
  high:     { emoji: '🟠', label: 'High Risk',     color: '#f57c00' },
  critical: { emoji: '🔴', label: 'Critical Risk', color: 'var(--alert, #f44336)' },
};

const INTENT_LABELS = {
  read_only:           '📖 Read Only',
  write:               '✏️ File Write',
  destructive:         '💥 Destructive',
  network:             '🌐 Network',
  package_management:  '📦 Package Install',
  process_management:  '⚙️ Process Management',
  system_admin:        '🔒 System Admin',
  unknown:             '❓ Unknown',
};

/** Create a tool approval request card with Allow/Deny buttons.
 *  Returns { card, promise } where promise resolves to true (allowed) or false (denied). */
export function createToolApprovalCard(name, args, approvalId, riskLevel, intent, warnings) {
  const card = document.createElement('div');
  const risk = riskLevel || 'medium';
  card.className = `tool-card tool-approval risk-${risk}`;
  card.dataset.approvalId = approvalId;

  const icon = TOOL_ICONS[name] || '🔧';
  const command = args.command || JSON.stringify(args);
  const reason = args.reason || '';
  const riskCfg = RISK_CONFIG[risk] || RISK_CONFIG.medium;
  const intentLabel = INTENT_LABELS[intent] || INTENT_LABELS.unknown;
  const warningsList = (warnings || []);

  card.innerHTML = `
    <div class="tool-header">
      <span class="tool-icon">🛡️</span>
      <span class="tool-name">Approval Required</span>
      <span class="tool-risk-badge" style="color:${riskCfg.color}">${riskCfg.emoji} ${riskCfg.label}</span>
      <span class="tool-status">⏳</span>
    </div>
    <div class="tool-approval-meta">
      <span class="tool-intent-badge">${intentLabel}</span>
    </div>
    <div class="tool-approval-command">
      <span class="tool-approval-label">${icon} ${escapeHtml(name)}</span>
      <pre class="tool-approval-cmd">${escapeHtml(command)}</pre>
      ${reason ? `<div class="tool-approval-reason">${escapeHtml(reason)}</div>` : ''}
    </div>
    ${warningsList.length > 0 ? `
      <div class="tool-approval-warnings">
        ${warningsList.map(w => `<div class="tool-approval-warning">⚠️ ${escapeHtml(w)}</div>`).join('')}
      </div>
    ` : ''}
    <div class="tool-approval-actions">
      <button class="tool-approve-btn" data-action="allow" title="Allow this command (Enter)">✓ Allow</button>
      <button class="tool-deny-btn" data-action="deny" title="Deny this command (Esc)">✕ Deny</button>
    </div>
  `;

  const approveBtn = card.querySelector('.tool-approve-btn');
  const denyBtn = card.querySelector('.tool-deny-btn');

  const promise = new Promise((resolve) => {
    approveBtn.addEventListener('click', async () => {
      setApprovalState(card, true);
      await sendApprovalDecision(approvalId, true);
      resolve(true);
    });
    denyBtn.addEventListener('click', async () => {
      setApprovalState(card, false);
      await sendApprovalDecision(approvalId, false);
      resolve(false);
    });

    // Keyboard shortcuts: Enter=Allow, Escape=Deny
    function handleKeyboard(e) {
      if (card.classList.contains('approved') || card.classList.contains('denied')) return;
      if (e.key === 'Enter' && document.activeElement?.closest('.tool-approval') === card) {
        e.preventDefault();
        approveBtn.click();
      } else if (e.key === 'Escape' && document.activeElement?.closest('.tool-approval') === card) {
        e.preventDefault();
        denyBtn.click();
      }
    }
    card.addEventListener('keydown', handleKeyboard);
  });

  // Auto-focus the Allow button after it appears in the DOM
  requestAnimationFrame(() => approveBtn.focus());

  return { card, promise };
}

/** Update approval card to show the decision */
function setApprovalState(card, approved) {
  const actions = card.querySelector('.tool-approval-actions');
  const warnings = card.querySelector('.tool-approval-warnings');
  const status = card.querySelector('.tool-status');
  if (actions) actions.remove();
  if (warnings) warnings.remove();
  if (status) {
    status.textContent = approved ? '✅' : '❌';
    status.classList.remove('spinning');
  }
  card.classList.add(approved ? 'approved' : 'denied');
  const label = card.querySelector('.tool-name');
  if (label) label.textContent = approved ? 'Approved' : 'Denied';
}

/** Send the allow/deny decision to the server */
async function sendApprovalDecision(approvalId, approved) {
  try {
    await fetch('/v1/agent/approve-tool', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ approval_id: approvalId, approved }),
    });
  } catch (err) {
    console.error('Failed to send approval decision:', err);
  }
}
