/** MCP server management — UI for viewing and reloading MCP tool servers. */

import { showToast } from './ui.js';

const listEl = document.getElementById('mcp-server-list');
const summaryEl = document.getElementById('mcp-tool-summary');
const reloadBtn = document.getElementById('mcp-reload-btn');

let cachedTools = [];

/** Fetch all registered agent tools from the server */
async function fetchAgentTools() {
  try {
    const r = await fetch('/v1/agent/tools');
    if (!r.ok) return [];
    const data = await r.json();
    return data.tools || [];
  } catch {
    return [];
  }
}

/** Reload MCP servers via API */
async function reloadMcpServers() {
  try {
    reloadBtn.disabled = true;
    reloadBtn.textContent = '↻ Reloading…';
    const r = await fetch('/v1/agent/mcp/reload', { method: 'POST' });
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(err.detail || r.statusText);
    }
    const result = await r.json();
    showToast(`MCP reloaded: ${result.tools_registered} tools registered`);
    await renderMcpPanel();
  } catch (err) {
    showToast(`MCP reload failed: ${err.message}`, 'error');
  } finally {
    reloadBtn.disabled = false;
    reloadBtn.textContent = '↻ Reload';
  }
}

/** Render the MCP panel with tool list */
export async function renderMcpPanel() {
  const tools = await fetchAgentTools();
  cachedTools = tools;

  const mcpTools = tools.filter(t => t.is_mcp);
  const builtinTools = tools.filter(t => !t.is_mcp);

  // Render MCP tool list
  if (mcpTools.length === 0) {
    listEl.innerHTML = `
      <div class="mcp-empty">
        <div class="mcp-empty-icon">🔌</div>
        <div>No MCP servers connected</div>
        <div class="mcp-empty-hint">Edit <code>config/mcp.yaml</code> to add servers</div>
      </div>
    `;
  } else {
    // Group by server name
    const servers = {};
    for (const tool of mcpTools) {
      // name format: mcp_{server}_{tool}
      const parts = tool.name.split('_');
      const serverName = parts.length >= 3 ? parts[1] : 'unknown';
      if (!servers[serverName]) servers[serverName] = [];
      servers[serverName].push(tool);
    }

    let html = '';
    for (const [server, serverTools] of Object.entries(servers)) {
      html += `
        <div class="mcp-server-card" role="listitem">
          <div class="mcp-server-header">
            <span class="mcp-server-icon">🔌</span>
            <span class="mcp-server-name">${escapeHtml(server)}</span>
            <span class="mcp-server-count">${serverTools.length} tools</span>
            ${serverTools.some(t => t.requires_approval) ? '<span class="mcp-approval-badge">🛡️ approval</span>' : ''}
          </div>
          <div class="mcp-tool-list">
            ${serverTools.map(t => `
              <div class="mcp-tool-item" title="${escapeHtml(t.description)}">
                <span class="mcp-tool-name">${escapeHtml(t.name.replace(`mcp_${server}_`, ''))}</span>
                ${t.requires_approval ? '<span class="mcp-tool-approval">🛡️</span>' : ''}
              </div>
            `).join('')}
          </div>
        </div>
      `;
    }
    listEl.innerHTML = html;
  }

  // Summary line
  const approvalCount = tools.filter(t => t.requires_approval).length;
  summaryEl.innerHTML = `
    <span>${tools.length} tools total</span>
    <span class="mcp-sep">·</span>
    <span>${builtinTools.length} built-in</span>
    <span class="mcp-sep">·</span>
    <span>${mcpTools.length} MCP</span>
    ${approvalCount > 0 ? `<span class="mcp-sep">·</span><span>🛡️ ${approvalCount} need approval</span>` : ''}
  `;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

/** Initialize MCP panel */
export function initMcpPanel() {
  if (reloadBtn) {
    reloadBtn.addEventListener('click', reloadMcpServers);
  }
  // Load on first settings open
  const settingsBtn = document.getElementById('settings-btn');
  if (settingsBtn) {
    settingsBtn.addEventListener('click', () => {
      // Slight delay to allow panel animation
      setTimeout(renderMcpPanel, 100);
    });
  }
}
