"""按知识点补语法题：教材词汇约束 + 本地校验 + 落盘缓存。

为什么需要它：语料里的知识点题量极不均（be动词 25 道，「人称代词与物主代词」只有 1 道）。
当学生的薄弱点恰好是小题量知识点时，「本轮完成，再来一组」会无题可出、只能跳点。
所以允许大模型针对单个知识点补题，但必须守住两条底线：

  1. 不超纲 —— 题干与选项里的实词要落在教材词表内（超纲占比阈值 20%）；
  2. 可判分 —— 结构过 grammar.validate_item，答案唯一、选择题干扰项合法。

补出来的题落盘到 data/gen_cache/point_*.json，corpus.iter_items() 会自动并入，
下次启动仍然可用，不会反复调用大模型。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from . import DATA_DIR
from .grammar import GRAMMAR_POINTS, RULE_KINDS, validate_item
from .llm import LLM, last_error

CACHE_DIR = DATA_DIR / "gen_cache"

_WORD_RE = re.compile(r"[a-z][a-z'\-]+")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
OFF_TOPIC_RATIO = 0.2      # 超纲实词占比超过它就不要（比离线脚本的 0.3 更严）
MAX_PER_CALL = 6           # 单次补题上限

SYSTEM = (
    "你是人教版初中英语教研员，要为七年级（12-13 岁）学生补编语法练习题。\n"
    "硬约束：\n"
    "1. 只考「{point}」这一个知识点；其余用词与句型必须保持七年级水平，"
    "不要出现现在完成时、被动语态、从句、非谓语等超纲语法；\n"
    "2. 只能用下面【教材例句】里出现过的单词与短语；\n"
    "3. cloze（填空）题干必须有且只有一个 ___，answer 是唯一正确答案；\n"
    "4. choice（选择）给 3-4 个 options，answer 必须在 options 里，"
    "干扰项要是同学段常见混淆；\n"
    "5. 句子不超过 12 个单词，level 按难度填 1/2/3；\n"
    "6. hint 用中文写一句话解析（为什么是这个答案），面向 12-13 岁学生；\n"
    "7. 题干不要与下面【已有题】重复，也不要只换一个词。\n"
    "只输出 JSON，不要解释："
    '{{"items":[{{"kind":"cloze","text":"This ___ my sister.","zh":"这是我姐姐。",'
    '"answer":"is","accept":["is"],"options":["am","is","are"],"level":1,'
    '"hint":"this 是单数，用 is。"}}]}}'
)

USER = (
    "知识点：{point}\n"
    "【教材例句】\n{context}\n\n"
    "【已有题】\n{existing}\n\n"
    "请出 {n} 道题（cloze 与 choice 各占一半左右）。"
)

_CACHE: list[dict[str, Any]] | None = None
_SIG: tuple[tuple[str, int], ...] | None = None


def point_slug(point: str) -> str:
    """知识点 → 文件名片段。"""
    slug = _SLUG_RE.sub("_", str(point or "").strip().lower()).strip("_")
    return slug or "unknown"


def cache_file(point: str) -> Path:
    return CACHE_DIR / f"point_{point_slug(point)}.json"


def _signature() -> tuple[tuple[str, int], ...]:
    if not CACHE_DIR.exists():
        return ()
    return tuple(sorted((p.name, p.stat().st_mtime_ns) for p in CACHE_DIR.glob("point_*.json")))


def load_generated(force: bool = False) -> list[dict[str, Any]]:
    """读取所有补题（用「文件名 + mtime」签名做进程内缓存）。

    签名比对只是 stat：离线脚本在另一个进程里补了题，正在跑的服务也能自动感知，
    不必重启；平时也不会每次遍历题库都去读盘。
    """
    global _CACHE, _SIG
    sig = _signature()
    if _CACHE is not None and not force and sig == _SIG:
        return _CACHE
    out: list[dict[str, Any]] = []
    for path in sorted(CACHE_DIR.glob("point_*.json")) if CACHE_DIR.exists() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue          # 坏文件不影响主流程
        if isinstance(data, list):
            out.extend(x for x in data if isinstance(x, dict))
    _CACHE, _SIG = out, sig
    return out


def merge_generated(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把补题并入题目列表：同 id 或同题干的不重复计入。"""
    from .corpus import norm_text

    seen_ids = {str(i.get("id")) for i in items}
    seen_text = {norm_text(i.get("text")) for i in items}
    for item in load_generated():
        iid = str(item.get("id") or "")
        key = norm_text(item.get("text"))
        if not iid or iid in seen_ids or (key and key in seen_text):
            continue
        seen_ids.add(iid)
        seen_text.add(key)
        items.append(item)
    return items


def textbook_vocab() -> set[str]:
    """教材词汇基准：题库里出现过的英文词（题库本身是教材抽取 + 校验过的）。"""
    from .corpus import iter_items

    words: set[str] = set()
    for it in iter_items():
        words |= {w for w in _WORD_RE.findall(str(it.get("text", "")).lower()) if len(w) > 1}
    return words


def off_topic(item: dict[str, Any], vocab: set[str]) -> str:
    """超纲检查：题干与选项里的实词必须大部分落在教材词表里。"""
    text = f"{item.get('text', '')} {' '.join(str(o) for o in (item.get('options') or []))}"
    words = [w for w in _WORD_RE.findall(text.lower()) if len(w) > 2]
    if not words:
        return ""
    miss: list[str] = []
    for w in words:
        if w in vocab:
            continue
        # where's / brothers 这类屈折形式不算超纲
        stem = w.rstrip("'").rstrip("s").rstrip("'").rstrip("e")
        if stem in vocab or stem.strip("-") in vocab:
            continue
        miss.append(w)
    if len(miss) / len(words) > OFF_TOPIC_RATIO:
        return f"超纲词汇过多：{', '.join(miss[:6])}"
    return ""


def _context_for(point: str, existing: list[dict[str, Any]], top_k: int = 4) -> str:
    """教材线索：优先检索该知识点的教材原句，检索不到就用已有题的题干。"""
    lines: list[str] = []
    try:
        from .rag import search

        hits = search(f"{point} Grammar Focus", top_k=top_k, book="七上") or []
        if not hits:
            hits = search(point, top_k=top_k, book="七上") or []
        lines = [f"[p{h.get('page')}] {str(h.get('text', ''))[:300]}" for h in hits]
    except Exception:
        lines = []
    if not lines:
        lines = [str(x.get("text", "")) for x in existing[:top_k]]
    return "\n".join(lines) or "（无教材线索，请只用七年级常见基础句型）"


def _save(point: str, new_items: list[dict[str, Any]]) -> None:
    """把新题写回该知识点的缓存文件（同 id 覆盖）。"""
    global _CACHE
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_file(point)
    merged: dict[str, dict[str, Any]] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                merged = {str(x.get("id")): x for x in data if isinstance(x, dict)}
        except Exception:
            merged = {}
    for x in new_items:
        merged[str(x.get("id"))] = x
    path.write_text(
        json.dumps(list(merged.values()), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _CACHE = None          # 下次读取时重载，本进程立即能看到新题
    _SIG = None


def generate_for_point(
    point: str,
    need: int,
    items: list[dict[str, Any]] | None = None,
    save: bool = True,
) -> tuple[list[dict[str, Any]], list[str], str]:
    """让大模型针对一个知识点补题，本地校验后落盘。

    返回 (可用新题, 被剔除原因, 错误信息)。
    """
    from .corpus import iter_items, norm_text

    llm = LLM()
    if not llm.available:
        return [], [], "没有配置 LLM_API_KEY"
    pool = list(items if items is not None else iter_items())
    existing = [i for i in pool if str(i.get("grammar_point") or "") == point]
    need = max(1, min(int(need or 1), MAX_PER_CALL))
    data = llm.chat_json(
        SYSTEM.format(point=point),
        USER.format(
            point=point,
            context=_context_for(point, existing),
            existing="\n".join(f"- {i.get('text')}" for i in existing[:8]) or "（还没有这类题）",
            n=need,
        ),
        temperature=0.5,
    )
    if not data:
        return [], [], f"大模型没返回可用 JSON（{last_error() or '未知原因'}）"
    raw = data.get("items")
    if not isinstance(raw, list):
        return [], [], "返回里没有 items 数组"

    vocab = textbook_vocab()
    seen = {norm_text(i.get("text")) for i in pool}
    base = len(existing)
    ok: list[dict[str, Any]] = []
    dropped: list[str] = []
    for n, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            dropped.append(f"第{n}题不是对象")
            continue
        item["kind"] = str(item.get("kind") or "").strip().lower()
        if item["kind"] not in RULE_KINDS:
            dropped.append(f"第{n}题 kind={item['kind']} 不支持")
            continue
        reason = validate_item(item)
        if reason:
            dropped.append(f"第{n}题 {reason}")
            continue
        reason = off_topic(item, vocab)
        if reason:
            dropped.append(f"第{n}题 {reason}")
            continue
        key = norm_text(item.get("text"))
        if not key or key in seen:
            dropped.append(f"第{n}题 与已有题重复")
            continue
        seen.add(key)
        item["grammar_point"] = point            # 强制归到目标知识点，避免画像被打散
        item["id"] = f"topup_{point_slug(point)}_{base + len(ok) + 1:02d}"
        item["origin"] = "大模型按知识点补题（教材约束）"
        item["level"] = int(item.get("level") or 1)
        item["accept"] = [str(a) for a in (item.get("accept") or []) if str(a).strip()]
        item["book"] = "七上"
        ok.append(item)
    if ok and save:
        _save(point, ok)
    return ok, dropped, ""


def point_counts(items: list[dict[str, Any]] | None = None) -> dict[str, int]:
    """各知识点可用题量（含补题）。"""
    from .corpus import iter_items

    counts = {p: 0 for p in GRAMMAR_POINTS}
    for it in items if items is not None else iter_items():
        if str(it.get("kind")) not in RULE_KINDS or it.get("verified") is False:
            continue
        point = str(it.get("grammar_point") or "")
        if point:
            counts[point] = counts.get(point, 0) + 1
    return counts


def short_points(
    items: list[dict[str, Any]] | None = None, min_count: int = 6
) -> list[tuple[str, int]]:
    """题量不足的知识点，少的在前。"""
    return sorted(
        ((p, n) for p, n in point_counts(items).items() if n < min_count),
        key=lambda x: (x[1], x[0]),
    )


def ensure_topup(point: str, need: int = 3, min_count: int = 6) -> list[dict[str, Any]]:
    """题量不足时才补题（在线兜底，受 GRAMMAR_TOPUP_ONLINE 开关控制）。"""
    from . import config

    if not point or not config.grammar_topup_online():
        return []
    lack = max(0, min_count - point_counts().get(point, 0))
    if lack <= 0:
        return []
    ok, dropped, err = generate_for_point(point, min(int(need or 1), lack))
    if err:
        print(f"[topup] 「{point}」补题失败：{err}", file=sys.stderr)
    else:
        if dropped:
            print(
                f"[topup] 「{point}」剔除 {len(dropped)} 题：{'；'.join(dropped[:3])}",
                file=sys.stderr,
            )
        if ok:
            print(f"[topup] 「{point}」补了 {len(ok)} 道题", file=sys.stderr)
    return ok
