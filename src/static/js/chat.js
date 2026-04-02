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
} from './ui.js';
import {
  isAgentEnabled,
  createToolCallCard,
  createToolResultCard,
  markToolCompleted,
  createAgentSummary,
} from './agent.js';

const inputEl = document.getElementById('user-input');

/** Build the messages array for the API, including images if present. */
function buildUserContent(text, images) {
  if (!images || images.length === 0) return text;
  // OpenAI multimodal format
  const parts = [{ type: 'text', text }];
  for (const img of images) {
    parts.push({
      type: 'image_url',
      image_url: { url: img.dataUrl },
    });
  }
  return parts;
}

export async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text && state.pendingImages.length === 0) return;

  inputEl.value = '';
  inputEl.style.height = 'auto';

  // Capture and clear pending images
  const images = [...state.pendingImages];
  state.pendingImages = [];
  // Dispatch event so upload module clears previews
  document.dispatchEvent(new CustomEvent('tq:images-consumed'));

  const userContent = buildUserContent(text, images);
  state.history.push({ role: 'user', content: userContent });

  // Persist user message
  const textContent = typeof userContent === 'string' ? userContent : text;
  if (window._tqSaveMessage) window._tqSaveMessage('user', textContent);

  // Render user bubble (with optional image thumbnails)
  const { wrapper: userWrapper } = appendMessage('user', text || '[image]');
  if (images.length > 0) {
    const imgBar = document.createElement('div');
    imgBar.className = 'msg-images';
    for (const img of images) {
      const el = document.createElement('img');
      el.src = img.dataUrl;
      imgBar.appendChild(el);
    }
    userWrapper.querySelector('.msg-bubble').prepend(imgBar);
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
            asstWrapper.appendChild(card);
            getChatEl().scrollTop = getChatEl().scrollHeight;
          } else if (parsed.type === 'tool_result') {
            const callCard = activeToolCards[parsed.id];
            if (callCard) markToolCompleted(callCard);
            const resultCard = createToolResultCard(parsed.name, parsed.content || '');
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

export function clearSession() {
  state.history = [];
  state.totalTokens = 0;
  updateTokenCount();
  updateContextMeter();
  const chatEl = getChatEl();
  chatEl.innerHTML = '';
  chatEl.appendChild(getWelcomeEl());
}
