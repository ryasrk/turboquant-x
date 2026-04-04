/** State management for TurboQuant-X UI. */

const STORAGE_KEY = 'tq-settings';

function loadSettings() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) return JSON.parse(saved);
  } catch (_) {}
  return {
    systemPrompt: '',
    maxTokens: 2048,
    temperature: 0.7,
    topP: 0.95,
    inferenceMode: 'standard',
    debug: false,
  };
}

export const state = {
  history: [],
  settings: loadSettings(),
  streaming: false,
  totalTokens: 0,
  contextMax: 0,        // populated from /health
  contextUsed: 0,       // populated from /health
  thinking: true,
  agent: false,
  pendingAttachments: [], // { id: string|null, type: 'image'|'document', dataUrl: string, file: File, mimeType: string, name: string, size: number }
};

export function saveSettings() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state.settings));
}
