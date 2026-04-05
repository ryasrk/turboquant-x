"""Smart TF-IDF search for n8n nodes and templates.

Uses lightweight TF-IDF + cosine similarity (numpy only, no scikit-learn)
to find the most relevant nodes and templates for a user's prompt.
Returns only what matters — saves context window for the LLM.
"""

from __future__ import annotations

import math
import re
import time
import logging
from collections import Counter
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Tokenizer ────────────────────────────────────────────────────────

_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than",
    "too", "very", "just", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "through",
    "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "this", "that", "these", "those", "i", "me", "my", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "it", "its",
    "they", "them", "their", "what", "which", "who", "whom",
    "n8n", "nodes", "base", "node", "type", "version",
})

# ── Synonym / Alias Expansion ────────────────────────────────────────
# Maps user-friendly terms → canonical n8n terms.
# Both query and document text get expanded so synonyms match.
_SYNONYMS: dict[str, list[str]] = {
    "reminder": ["schedule", "cron", "timer", "interval"],
    "notify": ["send", "message", "alert", "notification"],
    "alert": ["notify", "message", "send"],
    "email": ["mail", "smtp", "gmail", "outlook"],
    "mail": ["email", "smtp"],
    "bot": ["chat", "agent", "automation"],
    "chat": ["message", "conversation", "bot"],
    "database": ["db", "postgres", "mysql", "sql", "supabase"],
    "db": ["database", "postgres", "mysql", "sql"],
    "api": ["http", "request", "rest", "webhook"],
    "webhook": ["http", "trigger", "hook"],
    "llm": ["openai", "gpt", "anthropic", "model", "langchain"],
    "ai": ["openai", "langchain", "model", "agent"],
    "gpt": ["openai", "chat", "model"],
    "cron": ["schedule", "timer", "interval", "trigger"],
    "schedule": ["cron", "timer", "interval"],
    "timer": ["schedule", "cron", "interval"],
    "file": ["read", "write", "local", "ftp", "sftp"],
    "spreadsheet": ["google", "sheets", "excel", "csv"],
    "sheets": ["spreadsheet", "google", "excel"],
    "storage": ["s3", "minio", "gcs", "blob", "bucket"],
    "transform": ["set", "map", "convert", "edit"],
    "filter": ["if", "switch", "condition", "route"],
    "loop": ["splitinbatches", "batch", "iterate"],
    "wait": ["delay", "pause", "sleep"],
    "merge": ["combine", "join", "append"],
    "split": ["separate", "batch", "divide"],
    "error": ["catch", "stop", "throw", "handle"],
    "code": ["function", "javascript", "python", "script"],
    "image": ["binary", "convert", "resize", "crop"],
    "pdf": ["document", "extract", "convert"],
}


def _expand_synonyms(tokens: list[str]) -> list[str]:
    """Expand tokens with synonym aliases (single pass, no infinite loops)."""
    expanded = list(tokens)
    for t in tokens:
        if t in _SYNONYMS:
            for syn in _SYNONYMS[t]:
                if syn not in expanded and syn not in _STOP_WORDS:
                    expanded.append(syn)
    return expanded


def _tokenize(text: str, expand: bool = True) -> list[str]:
    """Tokenize text into lowercase words, splitting camelCase and filtering stops."""
    # Split camelCase: "httpRequest" -> "http request"
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Split on non-alphanumeric
    words = _SPLIT_RE.split(text.lower())
    result = []
    for w in words:
        if not w or len(w) <= 1 or w in _STOP_WORDS:
            continue
        result.append(w)
        # Lightweight stemming — add root form for common suffixes
        stemmed = _stem(w)
        if stemmed != w and stemmed not in _STOP_WORDS and len(stemmed) > 1:
            result.append(stemmed)

    # Bigram indexing — capture multi-word phrases
    if len(result) >= 2:
        for i in range(len(result) - 1):
            result.append(f"{result[i]}_{result[i+1]}")

    # Synonym expansion
    if expand:
        result = _expand_synonyms(result)

    return result


def _stem(word: str) -> str:
    """Lightweight suffix stripping for better token matching."""
    for suffix in ("ation", "ting", "ment", "ness", "ally", "ious", "ical",
                   "ify", "ing", "ion", "ies", "ive", "ous", "ful",
                   "age", "ble", "ity", "ist", "ism", "ize",
                   "ly", "ed", "er", "es", "al", "ty"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
        return word[:-1]
    return word


# ── TF-IDF Index ─────────────────────────────────────────────────────

class TfIdfIndex:
    """Lightweight TF-IDF index using numpy."""

    def __init__(self):
        self._docs: list[dict[str, Any]] = []  # original data
        self._doc_tokens: list[list[str]] = []
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self._tfidf_matrix: np.ndarray | None = None
        self._built = False

    def add(self, data: dict[str, Any], text: str, boost_text: str = "") -> None:
        """Add a document to the index.

        Args:
            data: Original document data.
            text: Main searchable text.
            boost_text: High-importance text (e.g. displayName). Tokens
                        from this field are added 2x for higher weight.
        """
        tokens = _tokenize(text, expand=False)  # No synonym expansion for documents
        if boost_text:
            boost_tokens = _tokenize(boost_text, expand=False)
            tokens = boost_tokens + boost_tokens + tokens  # 2x weight
        self._docs.append(data)
        self._doc_tokens.append(tokens)
        self._built = False

    def build(self) -> None:
        """Build the TF-IDF matrix."""
        if not self._docs:
            return

        # Build vocabulary
        vocab: dict[str, int] = {}
        for tokens in self._doc_tokens:
            for t in set(tokens):
                if t not in vocab:
                    vocab[t] = len(vocab)
        self._vocab = vocab

        n_docs = len(self._docs)
        n_terms = len(vocab)
        if n_terms == 0:
            return

        # Document frequency
        df = np.zeros(n_terms, dtype=np.float32)
        for tokens in self._doc_tokens:
            for t in set(tokens):
                df[vocab[t]] += 1

        # IDF with smoothing
        self._idf = np.log((n_docs + 1) / (df + 1)) + 1

        # TF-IDF matrix (n_docs x n_terms)
        self._tfidf_matrix = np.zeros((n_docs, n_terms), dtype=np.float32)
        for i, tokens in enumerate(self._doc_tokens):
            if not tokens:
                continue
            tf = Counter(tokens)
            max_tf = max(tf.values())
            for term, count in tf.items():
                j = vocab[term]
                # Augmented TF to prevent bias toward longer documents
                self._tfidf_matrix[i, j] = (0.5 + 0.5 * count / max_tf) * self._idf[j]

        # L2 normalize rows
        norms = np.linalg.norm(self._tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self._tfidf_matrix /= norms

        self._built = True

    def search(self, query: str, top_k: int = 10) -> list[tuple[dict[str, Any], float]]:
        """Search the index. Returns list of (doc_data, score) sorted by relevance."""
        if not self._built or self._tfidf_matrix is None:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        # Build query vector
        q_vec = np.zeros(len(self._vocab), dtype=np.float32)
        tf = Counter(tokens)
        max_tf = max(tf.values()) if tf else 1
        for term, count in tf.items():
            if term in self._vocab:
                j = self._vocab[term]
                q_vec[j] = (0.5 + 0.5 * count / max_tf) * self._idf[j]

        # L2 normalize
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec /= norm

        # Cosine similarity
        scores = self._tfidf_matrix @ q_vec

        # Top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            s = float(scores[idx])
            if s > 0.01:  # threshold to filter noise
                results.append((self._docs[idx], s))

        return results


# ── Node Index (singleton, cached) ───────────────────────────────────

_node_index: TfIdfIndex | None = None
_node_index_ts: float = 0
_NODE_INDEX_TTL = 300  # 5 minutes


async def get_node_index() -> TfIdfIndex:
    """Get or build the node type TF-IDF index (cached 5 min)."""
    global _node_index, _node_index_ts

    if _node_index is not None and (time.time() - _node_index_ts) < _NODE_INDEX_TTL:
        return _node_index

    from src.server.n8n_manager import get_available_nodes
    nodes = await get_available_nodes()

    idx = TfIdfIndex()
    for n in nodes:
        name = n.get("name", "")
        display = n.get("displayName", "")
        desc = n.get("description", "")
        group = " ".join(n.get("group", []))
        cred_names = ""
        creds = n.get("credentials", [])
        if isinstance(creds, list):
            cred_names = " ".join(c.get("name", "") for c in creds if isinstance(c, dict))

        # Build searchable text from all relevant fields
        search_text = f"{name} {desc} {group} {cred_names}"
        # Boost displayName (2x weight) — it's the most distinctive field
        idx.add(n, search_text, boost_text=display)

    idx.build()
    _node_index = idx
    _node_index_ts = time.time()
    logger.info("Built node TF-IDF index: %d nodes", len(nodes))
    return idx


async def search_nodes(query: str, top_k: int = 15) -> list[dict[str, Any]]:
    """Search for relevant n8n node types using TF-IDF similarity.

    Deduplicates by displayName — keeps the highest-scored version.

    Args:
        query: Natural language query (user's workflow description).
        top_k: Maximum number of nodes to return.

    Returns:
        List of node dicts sorted by relevance.
    """
    idx = await get_node_index()
    # Fetch extra to account for dedup losses
    results = idx.search(query, top_k=top_k * 3)
    seen: dict[str, dict[str, Any]] = {}
    for doc, score in results:
        key = doc.get("displayName", doc.get("name", "")).strip()
        if key not in seen:
            seen[key] = doc
    return list(seen.values())[:top_k]


# ── Template Index (singleton, cached) ───────────────────────────────

_template_index: TfIdfIndex | None = None
_template_index_ts: float = 0
_TEMPLATE_INDEX_TTL = 600  # 10 minutes


async def get_template_index() -> TfIdfIndex:
    """Get or build the template TF-IDF index (cached 10 min)."""
    global _template_index, _template_index_ts

    if _template_index is not None and (time.time() - _template_index_ts) < _TEMPLATE_INDEX_TTL:
        return _template_index

    try:
        from src.server.n8n_templates import get_template_index as get_tpl_index
        templates = get_tpl_index()
    except Exception:
        logger.debug("Could not load templates for index")
        return TfIdfIndex()

    idx = TfIdfIndex()
    for t in templates:
        name = t.get("name", "")
        desc = t.get("description", "")
        category = t.get("category", "")
        nodes = " ".join(t.get("node_types", []))
        search_text = f"{desc} {category} {nodes}"
        idx.add(t, search_text, boost_text=name)

    idx.build()
    _template_index = idx
    _template_index_ts = time.time()
    logger.info("Built template TF-IDF index: %d templates", len(templates))
    return idx


async def search_templates_smart(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Search for relevant templates using TF-IDF similarity.

    Deduplicates by template name.

    Args:
        query: Natural language query.
        top_k: Maximum number of templates to return.

    Returns:
        List of template dicts sorted by relevance.
    """
    idx = await get_template_index()
    results = idx.search(query, top_k=top_k * 2)
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for doc, score in results:
        name = doc.get("name", "")
        if name not in seen:
            seen.add(name)
            deduped.append(doc)
    return deduped[:top_k]
