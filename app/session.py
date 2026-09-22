"""一次定级测评的完整流程：组卷 → 选工具 → 逐题评测 → 汇总分析 → 推练习。"""

from __future__ import annotations

import base64
import json
import time
import uuid
from pathlib import Path
from typing import Any

from . import DATA_DIR, config, tools
from .chivox import ChivoxMCPError
from .grammar import judge
from .corpus import (
    build_placement,
    chain_kind_of,
    find_item,
    item_phones,
    norm_text,
    recommend_pool,
)
from .llm import LLM
from .phonemes import issue_tip, tip_for
from .rag import search as rag_search
from .scoring import aggregate, summarize

HISTORY_FILE = DATA_DIR / "history.jsonl"

class SessionError(RuntimeError):
    pass


class SessionStore:
    """进程内存放会话；重启即失效，够演示用。"""

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    def create(self, student: str, size: int = 3) -> dict[str, Any]:
        size = max(2, min(int(size), 5))  # size = 单词题数量
        skip_paragraph = (
            "en_paragraph_eval" in config.chivox_skip_tools()
            or not tools.usable("en_paragraph_eval")
        )
        items = build_placement(size=size, skip_paragraph=skip_paragraph)
        session: dict[str, Any] = {
            "id": uuid.uuid4().hex[:12],
            "student": (student or "小朋友").strip(),
            "created_at": time.time(),
            "items": [],
            "results": [],           # 定级卷结果
            "practice_results": [],  # 练习结果（闭环：回流后才能出下一轮）
            "seen_items": [],        # 出过卷 / 推荐过的题，避免下一轮重复出现
            "round": 0,              # 第几轮练习（0 = 还没开始练）
            "report": None,
        }
        for item in items:
            item_chain = str(item.get("chain") or "word")
            session["items"].append(
                {
                    "index": len(session["items"]),
                    "item": item,
                    "group": item.get("group", "word"),
                    "chain": item_chain,
                    "tool": _tool_of(item_chain),
                    "ref_text": _default_ref_text(item),
                }
            )
        self._sessions[session["id"]] = session
        return session

    def get(self, session_id: str) -> dict[str, Any]:
        try:
            return self._sessions[session_id]
        except KeyError:
            raise SessionError(f"会话不存在或已过期：{session_id}") from None

    def list_ids(self) -> list[str]:
        return list(self._sessions.keys())


# 全局会话单例：main.py 与 record_practice / next_round 共用同一个，
# 否则练习结果回写会写到另一个空实例里。
STORE = SessionStore()


def _tool_of(chain: str) -> str:
    """题型链 → 展示用的工具名。

    语法题（chain=rule）不调驰声，由 app.grammar 本地判分，
    这里返回 "rule"；否则会被错误地显示成链首的 en_word_eval。
    """
    return "rule" if chain == "rule" else tools.chain_for(chain)[0]


def _default_ref_text(item: dict[str, Any]) -> str:
    text = str(item.get("text") or "")
    if item.get("kind") == "semi_open":
        return str(item.get("sample") or item.get("answer") or text)
    if item.get("kind") == "choice":
        return text.replace(" ", "")
    return text


def make_entry(kind: str, text: str, index: int = 0, group: str = "word") -> dict[str, Any]:
    """推荐练习用的临时题（可能不在语料里，比如词表配对题）。"""
    item = {"id": f"ad_{index}", "kind": kind, "text": text, "group": group, "chain": kind}
    return {
        "index": index,
        "item": item,
        "group": group,
        "chain": kind,
        "tool": _tool_of(kind),
        "ref_text": text,
    }


def evaluate_entry(
    entry: dict[str, Any],
    audio_base64: str | None = None,
    audio_url: str | None = None,
    answer: str | None = None,
) -> dict[str, Any]:
    """评测一道题：口语题走工具链（失败自动降级），语法题走本地规则判分。"""
    if str(entry.get("chain") or "word") == "rule":
        item = entry["item"]
        summary = judge(item, answer)
        return {
            "index": entry["index"],
            "group": "grammar",
            "chain": "rule",
            "tool": "rule",
            "tried": [{"tool": "rule", "ok": True}],
            "ref_text": str(item.get("text", "")),
            "item": item,
            "summary": summary,
            "raw": summary.get("raw") or {},
        }
    out = tools.evaluate_chain(
        kind=str(entry.get("chain") or "word"),
        ref_text=entry["ref_text"],
        audio_base64=audio_base64,
        audio_url=audio_url,
    )
    return {
        "index": entry["index"],
        "group": entry.get("group", "word"),
        "chain": entry.get("chain", "word"),
        "tool": out["tool"],
        "tried": out["tried"],
        "ref_text": entry["ref_text"],
        "item": entry["item"],
        "summary": out["summary"],
        "raw": out["raw"],
    }


def evaluate_single(
    item_id: str | None = None,
    audio_base64: str | None = None,
    audio_url: str | None = None,
    kind: str | None = None,
    text: str | None = None,
    answer: str | None = None,
) -> dict[str, Any]:
    """练习题评测：教材题给 item_id；配对/临时题给 kind + text。"""
    if item_id:
        item = dict(find_item(item_id))
    elif kind and text:
        item = {"id": f"ad_{abs(hash(text)) % 100000}", "kind": kind, "text": text}
    else:
        raise ValueError("需要 item_id 或 (kind, text) 之一")
    chain = str(item.get("chain") or "") or chain_kind_of(item)
    item["chain"] = chain
    entry = {
        "index": 0,
        "group": item.get("group", "word"),
        "chain": chain,
        "tool": _tool_of(chain),
        "ref_text": _default_ref_text(item),
        "item": item,
    }
    return evaluate_entry(
        entry, audio_base64=audio_base64, audio_url=audio_url, answer=answer
    )


def _rule_report(agg: dict[str, Any]) -> dict[str, Any]:
    """没有大模型时的规则版学生报告：结论 / 证据 / 动作。"""
    dims = {k: v for k, v in (agg.get("dims") or {}).items() if v is not None}
    overall = agg.get("overall")
    best = max(dims, key=lambda k: dims[k]) if dims else None
    worst = agg.get("weakest_dim")
    weak_ph = agg.get("weak_phonemes") or []
    focus = weak_ph[0]["phoneme"] if weak_ph else None
    if overall is None:
        conclusion = "这次没有拿到有效分数，换个安静的地方再录一次吧。"
    else:
        conclusion = f"这次 {overall} 分（{agg.get('level', '')}）。"
        if best and worst and best != worst:
            conclusion += f"{best} 比 {worst} 好（{dims[best]}% vs {dims[worst]}%）。"
    ev: list[str] = []
    profile = agg.get("error_profile") or {}
    confusions = agg.get("sound_confusions") or []
    if weak_ph:
        hit = weak_ph[0]
        words = "、".join(hit.get("words") or []) or (hit.get("label") or "")
        detail = hit.get("kinds_text") or f"{hit.get('count')} 次问题"
        ev.append(f"/{focus}/ 还没定型（{detail}，如 {words}）。")
    if profile:
        line = "错误画像：" + "、".join(f"{k} {v} 处" for k, v in profile.items())
        if confusions:
            top = confusions[0]
            line += f"；最典型的是 {top['label']}（{top['count']} 次）"
        ev.append(line + "。")
    if "流利度" in dims and dims["流利度"] < 70:
        ev.append(f"流利度 {dims['流利度']}%，读整句时断得比较多。")
    if "内容命中" in dims and dims["内容命中"] < 60:
        ev.append("有题没答到点子上，意思还没说全。")
    if agg.get("invalid_count"):
        ev.append(f"有 {agg['invalid_count']} 题没答到点上，本次不计入发音统计。")
    if not ev:
        ev.append("这次没有抓到明显问题，读音和节奏都挺稳。")
    if confusions:
        top = confusions[0]
        action = f"今天只练 {top['label']}：{issue_tip(top['expected'], top['actual'], 'mispron')}"
    elif focus:
        action = f"今天只练 /{focus}/：{tip_for(focus)}读 5 遍就休息，三天后用新词再测一次。"
    else:
        action = "今天挑一句教材短句，慢速跟读 5 遍，注意一口气读完。"
    return {"conclusion": conclusion, "evidence": ev[:3], "action": action, "source": "rule"}


GROUP_LIMIT = {"word": 3, "sentence": 3, "paragraph": 1, "grammar": 3}


def recommend_groups(
    agg: dict[str, Any],
    exclude_ids: set[str] | None = None,
    exclude_texts: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """三类练习推荐：先按薄弱点检索候选，再让大模型在候选里挑并写针对性理由。

    约束：
    1. 大模型只能从候选里挑 id —— 候选全部来自课本语料，不允许自造单词（超纲即失效）；
    2. 候选已按题号与文本双重排除定级测评考过的题；
    3. 大模型不可用或挑选失败时，保留规则候选，保证三栏都有题。
    """
    groups = recommend_pool(agg, exclude_ids=exclude_ids, exclude_texts=exclude_texts)
    # 补教材出处
    for items in groups.values():
        for it in items:
            hits = rag_search(str(it.get("text", "")), top_k=1)
            it["ref"] = (
                f"{hits[0].get('book')} {hits[0].get('unit') or ''} p{hits[0].get('page')}".strip()
                if hits
                else ""
            )
            it.setdefault("source", "rule")

    llm = LLM()
    if (
        not llm.available
        or not config.llm_enabled()
        or not config.recommend_use_llm()
        or not any(groups.values())
    ):
        return groups      # 关掉大模型挑题（或没配 key）时，直接用规则候选

    candidates: list[dict[str, Any]] = []
    for group, items in groups.items():
        for it in items:
            candidates.append(
                {
                    "id": str(it.get("id")),
                    "group": group,
                    "kind": it.get("kind"),
                    "text": it.get("text"),
                    "zh": it.get("zh"),
                    "level": it.get("level"),
                    "unit": it.get("unit_name"),
                    "phones": item_phones(it),
                    "origin": it.get("origin"),
                    "point": it.get("point") or it.get("grammar_point"),
                }
            )
    if not candidates:
        return groups

    payload = {
        "grade": 7,
        "overall": agg.get("overall"),
        "level": agg.get("level"),
        "weak_phonemes": [
            {
                "phoneme": w.get("phoneme"),
                "kinds": w.get("kinds_text") or f"{w.get('count')} 次",
                "misread_as": w.get("misread_text") or "",
                "words": w.get("words"),
            }
            for w in (agg.get("weak_phonemes") or [])
        ],
        "weak_words": agg.get("weak_words"),
        "error_profile": agg.get("error_profile"),
        "sound_confusions": agg.get("sound_confusions"),
        "grammar_score": agg.get("grammar_score"),
        "weak_grammar": agg.get("weak_grammar"),
        "candidates": candidates[:10],
    }
    try:
        picks = llm.recommend(payload) or []
    except Exception:
        return groups
    if not picks:
        return groups

    by_id = {str(c["id"]): c for c in candidates}
    order: list[str] = []
    for p in picks:
        cid = str(p.get("id"))
        if not isinstance(p, dict) or cid not in by_id or cid in order:
            continue        # 不在候选里的 id 一律丢弃 → 保证不超出课本
        order.append(cid)
        item = by_id[cid]
        if p.get("reason"):
            item["_llm_reason"] = str(p["reason"])
        if p.get("focus"):
            item["_llm_focus"] = str(p["focus"])
    if not order:
        return groups

    # 大模型挑的排前面，规则候选补足，保证每组题量
    final: dict[str, list[dict[str, Any]]] = {
        "word": [],
        "sentence": [],
        "paragraph": [],
        "grammar": [],
    }
    for cid in order:
        cand = by_id[cid]
        group = str(cand.get("group") or "word")
        if group not in final or len(final[group]) >= GROUP_LIMIT.get(group, 3):
            continue
        item = next((it for it in groups[group] if str(it.get("id")) == cid), None)
        if item is None:
            continue
        if cand.get("_llm_reason"):
            item["reason"] = cand["_llm_reason"]
        if cand.get("_llm_focus"):
            item["dimension"] = cand["_llm_focus"]
        item["source"] = "llm"
        final[group].append(item)
    for group, items in groups.items():
        for it in items:
            if len(final[group]) >= GROUP_LIMIT.get(group, 3):
                break
            if all(str(x.get("id")) != str(it.get("id")) for x in final[group]):
                final[group].append(it)
    return final


def _remember(session: dict[str, Any], practice: dict[str, list[dict[str, Any]]]) -> None:
    """记住本轮推荐过的题：下一轮不再出现（哪怕学生没做）。"""
    seen = session.setdefault("seen_items", [])
    known = {str(x.get("id")) for x in seen}
    for items in practice.values():
        for it in items:
            key = str(it.get("id") or "")
            if key and key not in known:
                seen.append({"id": key, "text": str(it.get("text", ""))})
                known.add(key)


def record_practice(session_id: str, result: dict[str, Any]) -> None:
    """闭环一期①：把练习结果回写到会话（否则服务端不知道学生练得怎么样）。"""
    session = STORE.get(session_id)
    results = session.setdefault("practice_results", [])
    results.append(result)


def next_round(session_id: str) -> dict[str, Any]:
    """闭环一期②③：用「定级 + 练习」的全部结果重新画像，出下一组题。

    与定级报告的区别：
      - 画像按全部历史结果重算（练习会影响弱音素 / 弱语法点）；
      - 排除范围扩大到「定级考过 + 之前练过」的题号与文本，避免重复。
    """
    session = STORE.get(session_id)
    base = list(session.get("results") or [])
    practiced = list(session.get("practice_results") or [])
    results = base + practiced
    if not results:
        raise SessionError("还没有任何评测记录，先做定级测评")

    agg = aggregate(results)
    used_ids = {str(r["item"].get("id")) for r in results if r.get("item")}
    used_texts = {norm_text(r.get("ref_text")) for r in results}
    used_texts |= {norm_text((r.get("item") or {}).get("text", "")) for r in results}
    # 定级卷里没做的题、以及之前推荐过的题，都不再出现
    for entry in session.get("items") or []:
        item = entry.get("item") or {}
        if item.get("id"):
            used_ids.add(str(item["id"]))
        if item.get("text"):
            used_texts.add(norm_text(item.get("text")))
    for past in session.get("seen_items") or []:
        if past.get("id"):
            used_ids.add(str(past["id"]))
        if past.get("text"):
            used_texts.add(norm_text(past.get("text")))
    used_texts.discard("")

    practice = recommend_groups(agg, exclude_ids=used_ids, exclude_texts=used_texts)
    _remember(session, practice)
    session["round"] = int(session.get("round") or 0) + 1

    # 更新报告里与练习直接相关的部分（报告文字不重算，避免重复调用大模型）
    report = session.get("report") or {}
    report["practice"] = practice
    report["round"] = session["round"]
    report["dims"] = agg.get("dims")
    report["weakest_dim"] = agg.get("weakest_dim")
    report["overall"] = agg.get("overall")
    report["grammar_score"] = agg.get("grammar_score")
    report["grammar_profile"] = agg.get("grammar_profile")
    report["weak_phonemes"] = agg.get("weak_phonemes")
    report["weak_grammar"] = agg.get("weak_grammar")
    session["report"] = report

    return {
        "round": session["round"],
        "practice": practice,
        "overall": agg.get("overall"),
        "grammar_score": agg.get("grammar_score"),
        "dims": agg.get("dims"),
        "weakest_dim": agg.get("weakest_dim"),
        "weak_phonemes": agg.get("weak_phonemes"),
        "weak_grammar": agg.get("weak_grammar"),
        "answered": len(results),
        "practiced": len(practiced),
    }


def build_report(session: dict[str, Any]) -> dict[str, Any]:
    results = session.get("results") or []
    if not results:
        raise SessionError("还没有评测记录，先录几道题再生成报告。")
    agg = aggregate(results)
    llm = LLM()
    report: dict[str, Any] | None = None
    used_ids = {str(r["item"].get("id")) for r in results}
    # 推荐题不能与定级测评考过的文本重复：题号 + 文本双重排除
    used_texts = {norm_text(r.get("ref_text")) for r in results}
    used_texts |= {norm_text(r["item"].get("text", "")) for r in results}
    used_texts.discard("")

    def _make_practice() -> dict[str, list[dict[str, Any]]]:
        groups = recommend_groups(agg, exclude_ids=used_ids, exclude_texts=used_texts)
        _remember(session, groups)
        return groups

    analyze_payload = {
        "grade": 7,
        "student": session.get("student"),
        "overall": agg.get("overall"),
        "level_hint": agg.get("level"),
        "dims": agg.get("dims"),
        "weak_phonemes": agg.get("weak_phonemes"),
        "weak_words": agg.get("weak_words"),
        "error_profile": agg.get("error_profile"),
        "sound_confusions": agg.get("sound_confusions"),
        "grammar_score": agg.get("grammar_score"),
        "weak_grammar": agg.get("weak_grammar"),
        "items": [
            {
                "kind": r["item"].get("kind"),
                "text": str(r["ref_text"])[:60],
                "overall": r["summary"].get("overall"),
                # 只给最关键的几处错误：JSON 越小，模型回得越快
                "top_errors": [
                    e.get("expected") or e.get("phoneme")
                    for e in (r["summary"].get("errors") or [])[:3]
                ],
            }
            for r in results
        ],
    }
    if llm.available and config.llm_enabled():
        # 串行 + 失败重试：实测并发两次调用时偶发「HTTP 200 但 content 为空」，
        # 结果会静默降级成规则版报告；串行总耗时并不比并发差。
        for _ in range(2):
            try:
                report = llm.analyze(analyze_payload)
            except Exception:
                report = None
            if report:
                break
    practice = _make_practice()
    if not practice:
        practice = _make_practice()
    if not report:
        report = _rule_report(agg)

    final = {
        "student": session.get("student"),
        "created_at": session.get("created_at"),
        "overall": agg.get("overall"),
        "level": report.get("level") or agg.get("level"),
        "dims": agg.get("dims"),
        "weakest_dim": agg.get("weakest_dim"),
        "invalid_count": agg.get("invalid_count"),
        "weak_phonemes": agg.get("weak_phonemes"),
        "weak_words": agg.get("weak_words"),
        "error_profile": agg.get("error_profile"),
        "error_total": agg.get("error_total"),
        "sound_confusions": agg.get("sound_confusions"),
        "grammar_score": agg.get("grammar_score"),
        "grammar_profile": agg.get("grammar_profile"),
        "weak_grammar": agg.get("weak_grammar"),
        "analysis": report,
        "practice": practice,
        "per_item": [
            {
                "index": r["index"],
                "group": r.get("group", "word"),
                "kind": r["item"].get("kind"),
                "text": r["ref_text"],
                "zh": r["item"].get("zh"),
                "tool": r["tool"],
                "overall": r["summary"].get("overall"),
                "multi_dim": r["summary"].get("multi_dim"),
                "dims": r["summary"].get("dims"),
                "errors": r["summary"].get("errors"),
                "words": r["summary"].get("words"),
            }
            for r in results
        ],
    }
    session["report"] = final
    _append_history(session, final)
    return final


def _append_history(session: dict[str, Any], report: dict[str, Any]) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "student": session.get("student"),
            "session_id": session.get("id"),
            "overall": report.get("overall"),
            "level": report.get("level"),
            "weak_phonemes": [w["phoneme"] for w in report.get("weak_phonemes", [])],
            "practice": [
                i.get("id")
                for g in (report.get("practice") or {}).values()
                for i in g
            ],
        }
        with HISTORY_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


def read_history(limit: int = 50) -> list[dict[str, Any]]:
    if not Path(HISTORY_FILE).exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]


def encode_audio(raw: bytes, suffix: str = ".wav") -> str:
    """转成评测用 audio_base64。

    注意：驰声 MCP 的 base64 通道对 wav 会卡死超时，必须用 mp3（见 audio.to_eval_audio）。
    """
    from .audio import to_eval_audio

    data, _ = to_eval_audio(raw, suffix)
    return base64.b64encode(data).decode("ascii")
