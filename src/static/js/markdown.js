/** Lightweight Markdown → HTML renderer (XSS-safe).
 *
 * Escapes all HTML first, then applies markdown patterns.
 * Only produces controlled HTML elements — no raw HTML passthrough.
 * Supports KaTeX math: $inline$ and $$block$$.
 */

/** Placeholder prefix unlikely to collide with content. */
const _PH = '\x00MATH';
let _mathSlots = [];

/** Extract math expressions before HTML escaping and replace with placeholders. */
function extractMath(text) {
  _mathSlots = [];
  // Block math: $$...$$  (may span multiple lines)
  text = text.replace(/\$\$([\s\S]+?)\$\$/g, (_, expr) => {
    const idx = _mathSlots.length;
    _mathSlots.push({ expr: expr.trim(), displayMode: true });
    return `${_PH}${idx}${_PH}`;
  });
  // Inline math: $...$  (single line, not empty, not starting/ending with space)
  text = text.replace(/\$([^\n$]+?)\$/g, (_, expr) => {
    const idx = _mathSlots.length;
    _mathSlots.push({ expr: expr.trim(), displayMode: false });
    return `${_PH}${idx}${_PH}`;
  });
  return text;
}

/** Restore math placeholders with KaTeX-rendered HTML. */
function restoreMath(html) {
  for (let i = 0; i < _mathSlots.length; i++) {
    const { expr, displayMode } = _mathSlots[i];
    let rendered;
    try {
      if (typeof katex !== 'undefined') {
        rendered = katex.renderToString(expr, {
          displayMode,
          throwOnError: false,
          trust: false,
        });
      } else {
        // KaTeX not loaded — fallback to styled plain text
        rendered = displayMode
          ? `<div class="math-fallback">$$${escapeHtml(expr)}$$</div>`
          : `<span class="math-fallback">$${escapeHtml(expr)}$</span>`;
      }
    } catch {
      rendered = displayMode
        ? `<div class="math-fallback">$$${escapeHtml(expr)}$$</div>`
        : `<span class="math-fallback">$${escapeHtml(expr)}$</span>`;
    }
    html = html.replace(`${_PH}${i}${_PH}`, rendered);
  }
  return html;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Render inline markdown patterns within a single line.
 * Order matters — process code spans first to avoid conflicts.
 */
function renderInline(line) {
  // Inline code: `code`
  line = line.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Bold + italic: ***text***
  line = line.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  // Bold: **text**
  line = line.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic: *text*
  line = line.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, '<em>$1</em>');
  // Links: [text](url)
  line = line.replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  // Bare URLs: https://...
  line = line.replace(/(?<!")(?<!=)(https?:\/\/[^\s<"]+)/g,
    '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
  return line;
}

/**
 * Convert markdown text to safe HTML.
 * Supports: headings, bold, italic, code, links, lists, blockquotes, code blocks.
 */
export function renderMarkdown(text) {
  // Extract math before escaping HTML (KaTeX needs raw LaTeX)
  text = extractMath(text);
  // Then escape all HTML
  text = escapeHtml(text);

  const lines = text.split('\n');
  const out = [];
  let inCodeBlock = false;
  let inList = null; // 'ul' | 'ol' | null
  let codeLines = [];

  function closeList() {
    if (inList) {
      out.push(`</${inList}>`);
      inList = null;
    }
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Fenced code blocks: ```
    if (line.trimStart().startsWith('```')) {
      if (!inCodeBlock) {
        closeList();
        inCodeBlock = true;
        codeLines = [];
      } else {
        out.push(`<pre><code>${codeLines.join('\n')}</code></pre>`);
        inCodeBlock = false;
      }
      continue;
    }
    if (inCodeBlock) {
      codeLines.push(line);
      continue;
    }

    // Empty line → close list, add break
    if (line.trim() === '') {
      closeList();
      out.push('');
      continue;
    }

    // Headings: # ## ### ####
    const headingMatch = line.match(/^(#{1,4})\s+(.+)$/);
    if (headingMatch) {
      closeList();
      const level = headingMatch[1].length;
      out.push(`<h${level + 1}>${renderInline(headingMatch[2])}</h${level + 1}>`);
      continue;
    }

    // Unordered list: * item, - item, + item
    const ulMatch = line.match(/^(\s*)[*\-+]\s+(.+)$/);
    if (ulMatch) {
      if (inList !== 'ul') {
        closeList();
        inList = 'ul';
        out.push('<ul>');
      }
      out.push(`<li>${renderInline(ulMatch[2])}</li>`);
      continue;
    }

    // Ordered list: 1. item
    const olMatch = line.match(/^(\s*)\d+\.\s+(.+)$/);
    if (olMatch) {
      if (inList !== 'ol') {
        closeList();
        inList = 'ol';
        out.push('<ol>');
      }
      out.push(`<li>${renderInline(olMatch[2])}</li>`);
      continue;
    }

    // Blockquote: > text
    const bqMatch = line.match(/^>\s?(.*)$/);
    if (bqMatch) {
      closeList();
      out.push(`<blockquote>${renderInline(bqMatch[1])}</blockquote>`);
      continue;
    }

    // Horizontal rule: --- or ***
    if (/^[-*_]{3,}$/.test(line.trim())) {
      closeList();
      out.push('<hr>');
      continue;
    }

    // Regular paragraph line
    closeList();
    out.push(renderInline(line));
  }

  // Close any open blocks
  closeList();
  if (inCodeBlock) {
    out.push(`<pre><code>${codeLines.join('\n')}</code></pre>`);
  }

  // Join and wrap consecutive text lines into paragraphs
  return restoreMath(out
    .join('\n')
    .replace(/\n{2,}/g, '</p><p>')
    .replace(/^/, '<p>')
    .replace(/$/, '</p>')
    .replace(/<p><\/p>/g, '')
    .replace(/<p>(<(?:h[1-6]|ul|ol|pre|blockquote|hr)[^]*?<\/(?:h[1-6]|ul|ol|pre|blockquote)>|<hr>)<\/p>/g, '$1'));
}
