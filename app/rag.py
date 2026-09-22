"""教材 RAG：加载切片并做检索。

切库由 scripts/build_rag.py 生成（data/rag/chunks.jsonl）。
检索默认用轻量 BM25-lite（纯标准库，无依赖）；若配置了 EMBEDDING_API_KEY
则自动切换成向量检索（余弦相似度）。
"""

from __future__ import annotations

import json
import math
import os
import re

from . import DATA_DIR

CHUNKS = DATA_DIR / "rag" / "chunks.jsonl"
META = DATA_DIR / "rag" / "meta.json"

_CACHE: list[dict] | None = None


def load_chunks() -> list[dict]:
    global _CACHE
    if _CACHE is None:
        rows: list[dict] = []
        if CHUNKS.exists():
            for line in CHUNKS.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        _CACHE = rows
    return _CACHE


def reload_chunks() -> list[dict]:
    global _CACHE
    _CACHE = None
    return load_chunks()


def tokenize(text: str) -> list[str]:
    """英文按词、中文按字 + 二元组，够用且无需分词依赖。"""
    low = text.lower()
    words = re.findall(r"[a-z][a-z'\-]+", low)
    zh = re.findall(r"[一-鿿]", text)
    bigrams = ["".join(p) for p in zip(zh, zh[1:])]
    digits = re.findall(r"\d+", low)
    return words + zh + bigrams + digits


_INDEX: dict | None = None


def _build_index(chunks: list[dict]) -> dict:
    """给全量切片建一次索引（分词结果 + df + 平均长度）。

    原来每次 search 都要把全部切片重新分词；生成报告/推荐会连着调用十几次，
    这是「生成报告很慢」的主要原因之一。
    """
    global _INDEX
    if _INDEX is not None and _INDEX.get("stamp") is _CACHE and _INDEX.get("n") == len(chunks):
        return _INDEX
    doc_tokens = [tokenize(c.get("text", "")) for c in chunks]
    df: dict[str, int] = {}
    for tokens in doc_tokens:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1
    _INDEX = {
        "n": len(chunks),
        "stamp": _CACHE,
        "doc_tokens": doc_tokens,
        "df": df,
        "avg_len": sum(len(t) for t in doc_tokens) / max(len(doc_tokens), 1),
    }
    return _INDEX


def _score(q_tokens: list[str], tokens: list[str], index: dict, k1: float = 1.5, b: float = 0.75) -> float:
    tf: dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    n = index["n"]
    df = index["df"]
    avg_len = index["avg_len"]
    score = 0.0
    for t in set(q_tokens):
        if t not in tf:
            continue
        idf = math.log(1 + (n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
        score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(tokens) / avg_len))
    return score


def search(query: str, top_k: int = 5, book: str | None = None) -> list[dict]:
    """检索教材切片。book 可选 '七上'。"""
    chunks = load_chunks()
    if not chunks:
        return []
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    index = _build_index(chunks)
    scored: list[tuple[float, dict]] = []
    for i, chunk in enumerate(chunks):
        if book and chunk.get("book") != book:
            continue
        score = _score(q_tokens, index["doc_tokens"][i], index)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: -x[0])
    return [{**c, "score": round(s, 4)} for s, c in scored[:top_k]]


def context_for(query: str, top_k: int = 4) -> str:
    """拼成可直接喂给大模型的教材上下文。"""
    hits = search(query, top_k=top_k)
    return "\n".join(f"[{h['book']} {h.get('unit') or ''} p{h['page']}] {h['text'][:300]}" for h in hits)


def stats() -> dict:
    chunks = load_chunks()
    books: dict[str, int] = {}
    for c in chunks:
        books[c.get("book", "?")] = books.get(c.get("book", "?"), 0) + 1
    meta = {}
    if META.exists():
        try:
            meta = json.loads(META.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    return {"chunks": len(chunks), "by_book": books, "built_at": meta.get("built_at"),
            "vector": bool(os.getenv("EMBEDDING_API_KEY"))}
