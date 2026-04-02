/** Image upload — pick files, preview thumbnails, attach to messages. */

import { state } from './state.js';

const uploadBtn    = document.getElementById('upload-btn');
const fileInput    = document.getElementById('image-input');
const previewBar   = document.getElementById('image-preview-bar');

const MAX_IMAGES   = 4;
const MAX_SIZE_MB  = 10;
const ALLOWED_TYPES = ['image/png', 'image/jpeg', 'image/gif', 'image/webp'];

/** Enable or disable the upload button based on model vision support. */
export function setVisionEnabled(enabled) {
  if (!uploadBtn) return;
  uploadBtn.disabled = !enabled;
  uploadBtn.style.opacity = enabled ? '' : '0.3';
  uploadBtn.style.pointerEvents = enabled ? '' : 'none';
  uploadBtn.title = enabled ? 'Attach image' : 'Current model does not support images';
}

export function initUpload() {
  if (!uploadBtn || !fileInput) return;

  uploadBtn.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', () => {
    const files = Array.from(fileInput.files);
    fileInput.value = '';   // reset so same file can be re-selected
    for (const file of files) {
      if (state.pendingImages.length >= MAX_IMAGES) break;
      if (!ALLOWED_TYPES.includes(file.type)) continue;
      if (file.size > MAX_SIZE_MB * 1024 * 1024) continue;
      readAndAdd(file);
    }
  });

  // Listen for chat consuming the images
  document.addEventListener('tq:images-consumed', () => {
    state.pendingImages = [];
    renderPreviews();
  });
}

function readAndAdd(file) {
  const reader = new FileReader();
  reader.onload = () => {
    state.pendingImages.push({ dataUrl: reader.result, file });
    renderPreviews();
  };
  reader.readAsDataURL(file);
}

function renderPreviews() {
  if (!previewBar) return;
  previewBar.innerHTML = '';
  if (state.pendingImages.length === 0) {
    previewBar.classList.remove('has-images');
    return;
  }
  previewBar.classList.add('has-images');

  state.pendingImages.forEach((img, i) => {
    const wrap = document.createElement('div');
    wrap.className = 'image-preview';
    const el = document.createElement('img');
    el.src = img.dataUrl;
    const rm = document.createElement('button');
    rm.className = 'remove-img';
    rm.textContent = '×';
    rm.addEventListener('click', () => {
      state.pendingImages.splice(i, 1);
      renderPreviews();
    });
    wrap.appendChild(el);
    wrap.appendChild(rm);
    previewBar.appendChild(wrap);
  });
}
