/** API layer — all server communication goes through here. */

export async function fetchHealth() {
  const r = await fetch('/health');
  if (!r.ok) throw new Error(`Health: ${r.status}`);
  return r.json();
}

export async function fetchAvailableModels() {
  const r = await fetch('/v1/available-models');
  if (!r.ok) throw new Error(`Models: ${r.status}`);
  return r.json();
}

export async function switchMode(mode) {
  const r = await fetch('/v1/switch-mode', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || r.statusText);
  }
  return r.json();
}

export async function switchModel(filename) {
  const r = await fetch('/v1/switch-model', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: filename }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || r.statusText);
  }
  return r.json();
}

export async function switchProvider(provider, apiKey) {
  const body = { provider };
  if (apiKey) body.api_key = apiKey;
  const r = await fetch('/v1/switch-provider', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || r.statusText);
  }
  return r.json();
}

export async function fetchCloudProviders() {
  const r = await fetch('/v1/cloud-providers');
  if (!r.ok) throw new Error(`Providers: ${r.status}`);
  return r.json();
}

export async function fetchProviderModels(provider) {
  const r = await fetch(`/v1/cloud-providers/${encodeURIComponent(provider)}/models`);
  if (!r.ok) throw new Error(`Provider models: ${r.status}`);
  return r.json();
}

export async function switchCloudModel(model) {
  const r = await fetch('/v1/switch-cloud-model', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || r.statusText);
  }
  return r.json();
}

/**
 * Upload a file to the server.
 * @param {File} file - The file to upload
 * @param {string} sessionId - Current session ID
 * @param {string} token - JWT auth token
 * @returns {Promise<{id: string, original_name: string, mime_type: string, size_bytes: number, type: string}>}
 */
export async function uploadFile(file, sessionId, token) {
  const form = new FormData();
  form.append('file', file);
  form.append('session_id', sessionId);
  const headers = {};
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const r = await fetch('/v1/upload', {
    method: 'POST',
    headers,
    body: form,
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || r.statusText);
  }
  return r.json();
}

/**
 * Start a streaming chat completion. Returns the ReadableStream reader.
 * @param {Array} messages - Chat messages
 * @param {Object} opts - { maxTokens, temperature, topP, thinking }
 * @returns {ReadableStreamDefaultReader}
 */
export async function chatStream(messages, opts) {
  const r = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages,
      max_tokens:  opts.maxTokens,
      temperature: opts.temperature,
      top_p:       opts.topP,
      stream:      true,
      thinking:    opts.thinking,
      tools:       opts.agent || false,
    }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    const detail = err.detail;
    // Structured error (e.g. context_exceeded) — extract .message
    if (detail && typeof detail === 'object') {
      const e = new Error(detail.message || JSON.stringify(detail));
      e.code = detail.error;
      e.contextMax = detail.context_max;
      throw e;
    }
    throw new Error(detail || r.statusText);
  }
  return r.body.getReader();
}
