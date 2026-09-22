"""驰声工具链：按题型选择优先级链，失败自动降级，并对 core 做熔断。

题型 → 工具优先级（左优先）：
    word            单词        en_word_correction → en_word_eval
    word_pair       配对单词    en_word_eval → en_word_correction
    sentence        短句跟读    en_sentence_correction → en_sentence_eval
    sentence_answer 短句应答    en_semi_open_eval → en_sentence_correction → en_sentence_eval
    paragraph       段落        en_paragraph_eval（无降级）
"""

from __future__ import annotations

import json
import time
from typing import Any

from .chivox import ChivoxMCP, ChivoxMCPError

CHAINS: dict[str, list[str]] = {
    "word": ["en_word_correction", "en_word_eval"],
    "word_pair": ["en_word_eval", "en_word_correction"],
    "sentence": ["en_sentence_correction", "en_sentence_eval"],
    "sentence_answer": ["en_semi_open_eval", "en_sentence_correction", "en_sentence_eval"],
    "paragraph": ["en_paragraph_eval"],
    "choice": ["en_choice_eval"],
    # "rule" 不在这里：语法题（填空/选择）不调驰声，由 app.grammar 本地判分
}

COOLDOWN = 300.0  # 某个 core 失败后的冷却秒数，避免每题都白等超时
_state: dict[str, dict[str, float]] = {}


def chain_for(kind: str) -> list[str]:
    return CHAINS.get(kind, ["en_word_eval"])


def usable(tool: str) -> bool:
    st = _state.get(tool)
    if not st:
        return True
    return (time.time() - st["fail_at"]) > COOLDOWN


def mark_fail(tool: str) -> None:
    st = _state.setdefault(tool, {"fail_at": 0.0, "fail_count": 0.0})
    st["fail_at"] = time.time()
    st["fail_count"] += 1


def mark_ok(tool: str) -> None:
    _state.pop(tool, None)


def core_status() -> dict[str, Any]:
    now = time.time()
    return {
        tool: {
            "fails": int(st["fail_count"]),
            "cooldown_left": max(0, int(COOLDOWN - (now - st["fail_at"]))),
        }
        for tool, st in _state.items()
    }


def build_ref(tool: str, ref_text: str) -> str:
    """semi_open 的 ref_text 必须是 JSON 字符串（双引号），否则 core 起不来（51000）。"""
    if tool == "en_semi_open_eval":
        return json.dumps({"lm": [{"text": ref_text}]}, ensure_ascii=False)
    return ref_text


def evaluate_chain(
    kind: str,
    ref_text: str,
    audio_base64: str | None = None,
    audio_url: str | None = None,
    client: ChivoxMCP | None = None,
    fill_dims: bool = True,
) -> dict[str, Any]:
    """按链逐个尝试，返回第一个成功的结果；失败的工具进冷却。"""
    from .scoring import summarize

    own = client is None
    mcp = client or ChivoxMCP()
    tried: list[dict[str, Any]] = []
    last_err = ""
    try:
        for tool in chain_for(kind):
            if not usable(tool):
                tried.append({"tool": tool, "skipped": "cooldown"})
                continue
            try:
                raw = mcp.evaluate(
                    tool=tool,
                    ref_text=build_ref(tool, ref_text),
                    audio_base64=audio_base64,
                    audio_url=audio_url,
                )
            except ChivoxMCPError as exc:
                last_err = str(exc)
                mark_fail(tool)
                tried.append({"tool": tool, "error": last_err[:120]})
                continue
            summary = summarize(raw)
            if summary.get("overall") is None and isinstance(raw, str):
                last_err = str(raw)[:150]
                mark_fail(tool)
                tried.append({"tool": tool, "error": last_err})
                continue
            mark_ok(tool)
            tried.append({"tool": tool, "ok": True})
            # 纠音类工具（*_correction）只给总分与建议，缺维度分时用链上其它工具补一次
            if fill_dims and not summary.get("dims") and not summary.get("multi_dim"):
                for extra in chain_for(kind)[1:]:
                    if extra == tool or not usable(extra):
                        continue
                    try:
                        raw2 = mcp.evaluate(
                            tool=extra,
                            ref_text=build_ref(extra, ref_text),
                            audio_base64=audio_base64,
                            audio_url=audio_url,
                        )
                    except ChivoxMCPError as exc:
                        mark_fail(extra)
                        tried.append({"tool": extra, "error": str(exc)[:80]})
                        continue
                    s2 = summarize(raw2)
                    if s2.get("dims") or s2.get("multi_dim"):
                        summary["dims"] = s2.get("dims") or summary.get("dims") or {}
                        summary["multi_dim"] = s2.get("multi_dim") or summary.get("multi_dim") or {}
                        # 纠音工具给的逐音素对齐（错读/漏读/多读）不能被覆盖，只做合并
                        if not summary.get("words"):
                            summary["words"] = s2.get("words") or []
                        merged_errors = list(summary.get("errors") or [])
                        for err in s2.get("errors") or []:
                            if err not in merged_errors:
                                merged_errors.append(err)
                        summary["errors"] = merged_errors
                        if not summary.get("phones_detail"):
                            summary["phones_detail"] = s2.get("phones_detail") or []
                        mark_ok(extra)
                        tried.append({"tool": extra, "ok": True, "fill_dims": True})
                        break
            return {"tool": tool, "summary": summary, "raw": raw, "tried": tried}
        raise ChivoxMCPError(last_err or "该题型所有评测工具当前都不可用，请稍后重试")
    finally:
        if own:
            mcp.close()
