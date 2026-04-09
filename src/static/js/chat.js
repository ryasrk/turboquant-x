/** Chat message handling — send, stream, clear. */

import { state } from './state.js';
import { chatStream } from './api.js';
import {
  showToast,
  appendMessage,
  renderAssistantContent,
  setStreaming,
  updateTokenCount,
  updateContextMeter,
  getChatEl,
  getWelcomeEl,
  updateStreamBubble,
  cancelStreamRender,
} from './ui.js';
import {
  isAgentEnabled,
  createToolCallCard,
  createToolResultCard,
  markToolCompleted,
  createAgentSummary,
  createToolApprovalCard,
} from './agent.js';
import { waitForUploads } from './upload.js';

const inputEl = document.getElementById('user-input');

/** Build the messages array for the API, including attachments if present. */
const TEXT_MIMES = ['text/plain', 'text/markdown', 'text/csv', 'application/json', 'text/x-yaml', 'text/x-python'];

function buildUserContent(text, attachments) {
  if (!attachments || attachments.length === 0) return text;

  const parts = [{ type: 'text', text }];
  for (const att of attachments) {
    if (att.type === 'image') {
      parts.push({
        type: 'image_url',
        image_url: { url: att.dataUrl },
      });
    } else if (att.id) {
      // Document with server ID: resolved server-side via attachment refs
      // (content injected by _resolve_attachments on server)
    } else if (att.dataUrl && TEXT_MIMES.includes(att.mimeType)) {
      // Fallback: decode text content from dataUrl for text-based files
      try {
        const base64 = att.dataUrl.split(',')[1];
        const decoded = atob(base64);
        parts.push({
          type: 'text',
          text: `\n[Document: ${att.name}]\n${decoded}`,
        });
      } catch {
        parts.push({ type: 'text', text: `\n[Attached: ${att.name}]` });
      }
    } else {
      parts.push({
        type: 'text',
        text: `\n[Attached: ${att.name} (${att.mimeType}) — login required for document reading]`,
      });
    }
  }
  return parts;
}

/** Build attachment refs for server-side resolution. */
function buildAttachmentRefs(attachments) {
  return attachments
    .filter(a => a.id)
    .map(a => ({ id: a.id, type: a.type }));
}

export async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text && state.pendingAttachments.length === 0) return;

  // Wait for any in-flight uploads to complete before sending
  if (state.pendingAttachments.some(a => a._uploadPromise && !a.id && !a._uploadFailed)) {
    showToast('Uploading files…');
    await waitForUploads();
  }

  inputEl.value = '';
  inputEl.style.height = 'auto';

  // Capture and clear pending attachments
  const attachments = [...state.pendingAttachments];
  state.pendingAttachments.length = 0;
  // Dispatch event so upload module clears previews
  document.dispatchEvent(new CustomEvent('tq:attachments-consumed'));

  const userContent = buildUserContent(text, attachments);
  const attachmentRefs = buildAttachmentRefs(attachments);
  const historyEntry = { role: 'user', content: userContent };
  if (attachmentRefs.length > 0) historyEntry.attachments = attachmentRefs;
  state.history.push(historyEntry);

  // Persist user message
  const textContent = typeof userContent === 'string' ? userContent : text;
  if (window._tqSaveMessage) window._tqSaveMessage('user', textContent);

  // Render user bubble (with optional attachment thumbnails)
  const { wrapper: userWrapper } = appendMessage('user', text || '[attachment]');
  if (attachments.length > 0) {
    const bubble = userWrapper.querySelector('.msg-bubble');

    // Attachment summary badge
    const badge = document.createElement('div');
    badge.className = 'msg-attach-badge';
    const fileCount = attachments.length;
    const docCount = attachments.filter(a => a.type === 'document').length;
    const imgCount = attachments.filter(a => a.type === 'image').length;
    let badgeText = `📎 ${fileCount} file${fileCount !== 1 ? 's' : ''} attached`;
    if (imgCount > 0 && docCount > 0) {
      badgeText = `📎 ${imgCount} image${imgCount !== 1 ? 's' : ''}, ${docCount} document${docCount !== 1 ? 's' : ''}`;
    }
    badge.innerHTML = `<span class="badge-text">${badgeText}</span><button class="badge-toggle" title="Toggle attachments">▼</button>`;
    bubble.prepend(badge);

    // Collapsible attachment detail container
    const detailContainer = document.createElement('div');
    detailContainer.className = 'msg-attach-detail';

    // Render images
    const images = attachments.filter(a => a.type === 'image');
    if (images.length > 0) {
      const imgBar = document.createElement('div');
      imgBar.className = 'msg-images';
      for (const img of images) {
        const el = document.createElement('img');
        el.src = img.dataUrl;
        imgBar.appendChild(el);
      }
      detailContainer.appendChild(imgBar);
    }
    
    // Render documents
    if (attachments.some(a => a.type === 'document')) {
      const attBar = document.createElement('div');
      attBar.className = 'msg-attachments';
      for (const att of attachments.filter(a => a.type === 'document')) {
        const chip = document.createElement('div');
        chip.className = 'attachment-chip';
        const sizeStr = att.size ? ` · ${formatSize(att.size)}` : '';
        chip.innerHTML = `<span class="chip-icon">${getDocIcon(att.mimeType)}</span><span class="chip-name">${escapeHtml(att.name)}</span><span class="chip-size">${sizeStr}</span>`;
        attBar.appendChild(chip);
      }
      detailContainer.appendChild(attBar);
    }

    bubble.insertBefore(detailContainer, badge.nextSibling);

    // Toggle expand/collapse
    badge.querySelector('.badge-toggle').addEventListener('click', () => {
      const isOpen = detailContainer.classList.toggle('open');
      badge.querySelector('.badge-toggle').textContent = isOpen ? '▲' : '▼';
    });
  }

  setStreaming(true);
  const { wrapper: asstWrapper, bubble } = appendMessage('assistant', '', true);
  const streamStart = performance.now();
  let firstTokenTime = null;
  let genTokens = 0;

  const messages = [];
  if (state.settings.systemPrompt) {
    messages.push({ role: 'system', content: state.settings.systemPrompt });
  }
  messages.push(...state.history);

  let assistantText = '';

  try {
    let lastFinishReason = null;
    let hasToolCalls = false;

    const reader = await chatStream(messages, {
      maxTokens:   state.settings.maxTokens,
      temperature: state.settings.temperature,
      topP:        state.settings.topP,
      thinking:    state.thinking,
      agent:       isAgentEnabled(),
    });
    const activeToolCards = {};
    const dec = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = dec.decode(value, { stream: true });
      for (const line of chunk.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) continue;
        const data = trimmed.slice(5).trim();
        if (data === '[DONE]') break;
        try {
          const parsed = JSON.parse(data);
          if (parsed.type === 'tool_call') {
            if (!hasToolCalls) {
              bubble.style.display = 'none';
              hasToolCalls = true;
            }
            const card = createToolCallCard(parsed.name, parsed.arguments || {});
            card.dataset.toolId = parsed.id;
            activeToolCards[parsed.id] = card;
            // Hide document-generation tool call cards unless debug is on
            const _DOC_TOOLS_CALL = ['generate_word', 'generate_pdf', 'generate_csv'];
            if (_DOC_TOOLS_CALL.includes(parsed.name) && !state.settings.debug) {
              card.style.display = 'none';
            }
            asstWrapper.appendChild(card);
            getChatEl().scrollTop = getChatEl().scrollHeight;
          } else if (parsed.type === 'tool_approval_request') {
            // Show approval card and wait for user decision
            const { card, promise } = createToolApprovalCard(
              parsed.name,
              parsed.arguments || {},
              parsed.approval_id,
              parsed.risk_level,
              parsed.intent,
              parsed.warnings,
            );
            card.dataset.toolId = parsed.id;
            asstWrapper.appendChild(card);
            getChatEl().scrollTop = getChatEl().scrollHeight;
          } else if (parsed.type === 'tool_approval_result') {
            // Approval resolved — no UI action needed (card already updated)
          } else if (parsed.type === 'tool_result') {
            const callCard = activeToolCards[parsed.id];
            if (callCard) markToolCompleted(callCard);
            const resultCard = createToolResultCard(parsed.name, parsed.content || '');
            // Hide document-generation result cards unless debug is on
            const _DOC_TOOLS = ['generate_word', 'generate_pdf', 'generate_csv'];
            if (_DOC_TOOLS.includes(parsed.name) && !state.settings.debug) {
              resultCard.style.display = 'none';
            }
            asstWrapper.appendChild(resultCard);
            getChatEl().scrollTop = getChatEl().scrollHeight;
          } else if (parsed.type === 'content') {
            if (!firstTokenTime) firstTokenTime = performance.now();
            genTokens++;
            assistantText += parsed.delta || '';
            if (hasToolCalls) {
              bubble.style.display = '';
              asstWrapper.appendChild(bubble);
            }
            updateStreamBubble(bubble, assistantText);
            getChatEl().scrollTop = getChatEl().scrollHeight;
          } else if (parsed.type === 'done') {
            lastFinishReason = parsed.finish_reason === 'length' ? 'length' : 'stop';
            asstWrapper.appendChild(createAgentSummary(parsed));
            getChatEl().scrollTop = getChatEl().scrollHeight;
          } else if (parsed.type === 'error') {
            bubble.textContent = `[ERROR] ${parsed.message}`;
            bubble.style.borderColor = 'var(--alert)';
            bubble.style.color = 'var(--alert)';
          } else if (parsed.choices) {
            const choice = parsed.choices[0];
            if (choice?.finish_reason) lastFinishReason = choice.finish_reason;
            const delta = choice?.delta?.content;
            if (delta) {
              if (!firstTokenTime) firstTokenTime = performance.now();
              genTokens++;
              assistantText += delta;
              updateStreamBubble(bubble, assistantText);
              getChatEl().scrollTop = getChatEl().scrollHeight;
            }
          }
        } catch (_) {}
      }
    }

    // Auto-continue if response was truncated by max_tokens
    const MAX_CONTINUATIONS = 3;
    const agentMode = isAgentEnabled();
    let continuations = 0;
    while (
      lastFinishReason === 'length' &&
      continuations < MAX_CONTINUATIONS
    ) {
      continuations++;
      const contMessages = [
        ...messages,
        { role: 'assistant', content: assistantText },
        { role: 'user', content: 'Continue.' },
      ];
      const contReader = await chatStream(contMessages, {
        maxTokens:   state.settings.maxTokens,
        temperature: state.settings.temperature,
        topP:        state.settings.topP,
        thinking:    false,
        agent:       agentMode,
      });
      lastFinishReason = null;
      while (true) {
        const { done, value } = await contReader.read();
        if (done) break;
        const chunk = dec.decode(value, { stream: true });
        for (const line of chunk.split('\n')) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const data = trimmed.slice(5).trim();
          if (data === '[DONE]') break;
          try {
            const parsed = JSON.parse(data);
            if (parsed.type === 'content') {
              // Agent continuation content
              genTokens++;
              assistantText += parsed.delta || '';
              updateStreamBubble(bubble, assistantText);
              getChatEl().scrollTop = getChatEl().scrollHeight;
            } else if (parsed.type === 'done') {
              lastFinishReason = parsed.finish_reason === 'length' ? 'length' : 'stop';
            } else if (parsed.choices) {
              // Non-agent continuation
              const choice = parsed.choices[0];
              if (choice?.finish_reason) lastFinishReason = choice.finish_reason;
              const delta = choice?.delta?.content;
              if (delta) {
                genTokens++;
                assistantText += delta;
                updateStreamBubble(bubble, assistantText);
                getChatEl().scrollTop = getChatEl().scrollHeight;
              }
            }
          } catch (_) {}
        }
      }
    }

    const cursor = bubble.querySelector('.cursor');
    if (cursor) cursor.remove();
    cancelStreamRender();  // Cancel any pending RAF before final render
    renderAssistantContent(bubble, assistantText);

    // Debug stats
    if (state.settings.debug && genTokens > 0) {
      const totalMs = performance.now() - streamStart;
      const genMs   = firstTokenTime ? (performance.now() - firstTokenTime) : totalMs;
      const ttft    = firstTokenTime ? (firstTokenTime - streamStart) : 0;
      const tps     = genTokens / (genMs / 1000);
      const debugEl = document.createElement('div');
      debugEl.className = 'debug-stats';
      debugEl.textContent =
        `⚡ ${tps.toFixed(1)} tok/s · ${genTokens} tokens · ` +
        `TTFT: ${ttft.toFixed(0)}ms · ${(totalMs / 1000).toFixed(1)}s total`;
      asstWrapper.appendChild(debugEl);
    }

    state.history.push({ role: 'assistant', content: assistantText });
    state.totalTokens += assistantText.split(/\s+/).length;
    updateTokenCount();
    updateContextMeter();

    // Persist assistant message
    if (window._tqSaveMessage) window._tqSaveMessage('assistant', assistantText);

  } catch (err) {
    const cursor = bubble.querySelector('.cursor');
    if (cursor) cursor.remove();
    const errMsg = err.code === 'context_exceeded'
      ? err.message + ' Clear the conversation to free space.'
      : err.message;
    bubble.textContent = `[ERROR] ${errMsg}`;
    bubble.style.borderColor = 'var(--alert)';
    bubble.style.color = 'var(--alert)';
    showToast(errMsg);
    state.history.pop();
  } finally {
    setStreaming(false);
  }
}

function getDocIcon(mimeType) {
  const icons = {
    'application/pdf': '📄',
    'text/plain': '📝',
    'text/markdown': '📝',
    'text/csv': '📊',
    'application/json': '📋',
    'text/x-yaml': '⚙️',
    'text/x-python': '🐍',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '📃',
  };
  return icons[mimeType] || '📎';
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function clearSession() {
  state.history = [];
  state.totalTokens = 0;
  updateTokenCount();
  updateContextMeter();
  const chatEl = getChatEl();
  chatEl.innerHTML = '';
  chatEl.appendChild(getWelcomeEl());
}
