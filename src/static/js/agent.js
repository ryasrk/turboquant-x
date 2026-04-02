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
  
  const icon = TOOL_ICONS[name] || '🔧';
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
  return escapeHtml(str).replace(
    /(https?:\/\/[^\s<"']+)/g,
    '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
  );
}

export function isAgentEnabled() {
  return state.agent === true;
}

export function toggleAgent() {
  state.agent = !state.agent;
  return state.agent;
}
