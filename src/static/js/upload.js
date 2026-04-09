/** File upload — pick files, preview thumbnails & document chips, attach to messages. */

import { state } from './state.js';
import { uploadFile } from './api.js';
import { getToken } from './auth.js';
import { getCurrentSessionId, createSession } from './sessions.js';
import { isLoggedIn } from './auth.js';

const uploadBtn    = document.getElementById('upload-btn');
const fileInput    = document.getElementById('file-input');
const previewBar   = document.getElementById('image-preview-bar');

const MAX_ATTACHMENTS = 8;
const MAX_SIZE_MB = 20;
const ALLOWED_TYPES = [
  'image/png', 'image/jpeg', 'image/gif', 'image/webp',
  'application/pdf', 'text/plain', 'text/markdown', 'text/csv',
  'application/json', 'text/x-yaml', 'text/x-python',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
];
const IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/gif', 'image/webp'];

/** Enable or disable upload capabilities based on model support. */
let _visionEnabled = false;

export function setUploadCapabilities(vision, attachments) {
  _visionEnabled = !!vision;
  window.tqVisionEnabled = _visionEnabled;
  if (!uploadBtn) return;
  uploadBtn.disabled = !attachments;
  uploadBtn.style.opacity = attachments ? '' : '0.3';
  uploadBtn.style.pointerEvents = attachments ? '' : 'none';
  uploadBtn.title = 'Attach files';
  // Update image option in menu if it exists
  const imgOpt = document.getElementById('upload-menu-image');
  if (imgOpt) {
    imgOpt.classList.toggle('disabled', !_visionEnabled);
    imgOpt.title = _visionEnabled ? 'Upload images' : 'Image upload requires cloud model with vision support';
    const hint = imgOpt.querySelector('.upload-menu-hint');
    if (hint && _visionEnabled) hint.remove();
    if (!hint && !_visionEnabled) {
      const span = document.createElement('span');
      span.className = 'upload-menu-hint';
      span.textContent = '(cloud only)';
      imgOpt.appendChild(span);
    }
  }
}

/** Backward compatibility alias */
export function setVisionEnabled(enabled) {
  setUploadCapabilities(enabled, true);
}

/** Wait for all pending attachment uploads to complete. */
export async function waitForUploads() {
  const promises = state.pendingAttachments
    .filter(a => a._uploadPromise)
    .map(a => a._uploadPromise);
  if (promises.length > 0) await Promise.allSettled(promises);
}

function _createUploadMenu() {
  const menu = document.createElement('div');
  menu.id = 'upload-menu';
  menu.className = 'upload-menu';
  menu.innerHTML = `
    <button id="upload-menu-image" class="upload-menu-item${_visionEnabled ? '' : ' disabled'}" type="button"
      title="${_visionEnabled ? 'Upload images' : 'Image upload requires cloud model with vision support'}">
      <span class="upload-menu-icon">🖼️</span>
      <span class="upload-menu-label">Upload Image</span>
      ${_visionEnabled ? '' : '<span class="upload-menu-hint">(cloud only)</span>'}
    </button>
    <button id="upload-menu-doc" class="upload-menu-item" type="button" title="Upload documents">
      <span class="upload-menu-icon">📄</span>
      <span class="upload-menu-label">Upload Document</span>
    </button>
  `;
  uploadBtn.parentElement.style.position = 'relative';
  uploadBtn.parentElement.appendChild(menu);
  return menu;
}

export function initUpload() {
  if (!uploadBtn || !fileInput) return;

  const menu = _createUploadMenu();
  let menuOpen = false;

  function toggleMenu() {
    menuOpen = !menuOpen;
    menu.classList.toggle('open', menuOpen);
  }
  function closeMenu() {
    menuOpen = false;
    menu.classList.remove('open');
  }

  uploadBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    toggleMenu();
  });

  // Close menu when clicking outside
  document.addEventListener('click', (e) => {
    if (menuOpen && !menu.contains(e.target) && e.target !== uploadBtn) closeMenu();
  });

  // Image option
  menu.querySelector('#upload-menu-image').addEventListener('click', () => {
    if (!_visionEnabled) {
      showToast('Image upload requires a cloud model with vision support');
      return;
    }
    fileInput.accept = IMAGE_TYPES.map(t => '.' + t.split('/')[1]).join(',');
    fileInput.click();
    closeMenu();
  });

  // Document option
  menu.querySelector('#upload-menu-doc').addEventListener('click', () => {
    fileInput.accept = '.pdf,.txt,.md,.json,.csv,.yaml,.yml,.py,.docx';
    fileInput.click();
    closeMenu();
  });

  fileInput.addEventListener('change', () => {
    const files = Array.from(fileInput.files);
    fileInput.value = '';   // reset so same file can be re-selected
    for (const file of files) {
      if (state.pendingAttachments.length >= MAX_ATTACHMENTS) break;
      if (!ALLOWED_TYPES.includes(file.type)) continue;
      if (file.size > MAX_SIZE_MB * 1024 * 1024) continue;
      if (IMAGE_TYPES.includes(file.type) && !_visionEnabled) {
        showToast('Images not supported by current model');
        continue;
      }
      readAndAdd(file);
    }
  });

  // Listen for chat consuming the attachments
  document.addEventListener('tq:attachments-consumed', () => {
    state.pendingAttachments = [];
    renderPreviews();
  });
}

function readAndAdd(file) {
  const reader = new FileReader();
  reader.onload = () => {
    const type = IMAGE_TYPES.includes(file.type) ? 'image' : 'document';
    const attachment = {
      id: null,
      type,
      dataUrl: reader.result,
      file,
      mimeType: file.type,
      name: file.name,
      size: file.size
    };
    state.pendingAttachments.push(attachment);
    renderPreviews();
    
    // Upload to server to get persistent attachment ID
    attachment._uploadPromise = (async () => {
      const token = getToken();
      if (!token || !isLoggedIn()) {
        showToast('Log in for full document reading support');
        return;
      }

      // Auto-create session if none exists
      let sessionId = getCurrentSessionId();
      if (!sessionId) {
        try {
          await createSession('New Chat');
          sessionId = getCurrentSessionId();
        } catch (e) {
          console.warn('Failed to create session for upload:', e);
          return;
        }
      }
      if (!sessionId) return;

      try {
        const response = await uploadFile(file, sessionId, token);
        attachment.id = response.id;
        attachment.serverName = response.original_name;
        renderPreviews();
      } catch (error) {
        console.warn(`Upload failed for ${file.name}: ${error.message}`);
        showToast(`Upload failed: ${error.message}`);
        attachment._uploadFailed = true;
      }
    })();
  };
  reader.readAsDataURL(file);
}

function renderPreviews() {
  if (!previewBar) return;
  previewBar.innerHTML = '';
  if (state.pendingAttachments.length === 0) {
    previewBar.classList.remove('has-attachments');
    return;
  }
  previewBar.classList.add('has-attachments');

  state.pendingAttachments.forEach((attachment, i) => {
    if (attachment.type === 'image') {
      renderImagePreview(attachment, i);
    } else {
      renderDocumentChip(attachment, i);
    }
  });
}

function renderImagePreview(attachment, index) {
  const wrap = document.createElement('div');
  wrap.className = 'image-preview';
  if (attachment.id) wrap.classList.add('uploaded');
  const el = document.createElement('img');
  el.src = attachment.dataUrl;
  const rm = document.createElement('button');
  rm.className = 'remove-img';
  rm.textContent = '×';
  rm.addEventListener('click', () => {
    state.pendingAttachments.splice(index, 1);
    renderPreviews();
  });
  wrap.appendChild(el);
  wrap.appendChild(rm);
  previewBar.appendChild(wrap);
}

function renderDocumentChip(attachment, index) {
  const chip = document.createElement('div');
  chip.className = 'attachment-chip';
  if (attachment.id) chip.classList.add('uploaded');
  if (attachment._uploadFailed) chip.classList.add('failed');
  
  const icon = document.createElement('span');
  icon.className = 'chip-icon';
  icon.textContent = attachment.id ? '✅' : attachment._uploadFailed ? '❌' : getFileIcon(attachment.mimeType);
  
  const name = document.createElement('span');
  name.className = 'chip-name';
  name.textContent = truncateFilename(attachment.name, 20);
  
  const size = document.createElement('span');
  size.className = 'chip-size';
  size.textContent = attachment.id ? 'uploaded' : attachment._uploadFailed ? 'failed' : formatFileSize(attachment.size);
  
  const rm = document.createElement('button');
  rm.className = 'remove-att';
  rm.textContent = '×';
  rm.addEventListener('click', () => {
    state.pendingAttachments.splice(index, 1);
    renderPreviews();
  });
  
  chip.appendChild(icon);
  chip.appendChild(name);
  chip.appendChild(size);
  chip.appendChild(rm);
  previewBar.appendChild(chip);
}

function getFileIcon(mimeType) {
  const iconMap = {
    'application/pdf': '📄',
    'text/plain': '📝',
    'text/markdown': '📝',
    'text/csv': '📊',
    'application/json': '📋',
    'text/x-yaml': '⚙️',
    'text/x-python': '🐍',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '📃'
  };
  return iconMap[mimeType] || '📄';
}

function truncateFilename(filename, maxLength) {
  if (filename.length <= maxLength) return filename;
  const extension = filename.split('.').pop();
  const nameWithoutExt = filename.slice(0, -(extension.length + 1));
  const truncated = nameWithoutExt.slice(0, maxLength - extension.length - 4) + '...';
  return truncated + '.' + extension;
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

function showToast(message) {
  // Simple toast implementation - can be enhanced
  const toast = document.createElement('div');
  toast.style.cssText = 'position:fixed;top:20px;right:20px;background:#333;color:white;padding:10px;border-radius:4px;z-index:9999';
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}
