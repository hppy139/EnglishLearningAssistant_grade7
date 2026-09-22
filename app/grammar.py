"""语法题（文本填空 / 选择题）的规则判分。

这两类题不经过驰声，纯规则判定对错。判分结果的**结构与 scoring.summarize() 同构**，
因此可以直接汇进 scoring.aggregate()，汇总逻辑不需要分支。

约定（见 docs/语料字段）：
    cloze  : {"text":"This ___ my sister.","answer":"is","accept":["is"],
              "options":["am","is","are"]（可选，给了就是选词填空）,
              "grammar_point":"be动词 am/is/are"}
    choice : {"text":"___ you have a soccer ball?","options":["Do","Does","Are","Is"],
              "answer":"Do","grammar_point":"一般现在时助动词 do/does"}
"""

from __future__ import annotations

import re
from typing import Any

# 受控知识点表：聚合与推荐都靠它，避免自由文本把画像打散
GRAMMAR_POINTS: list[str] = [
    "be动词 am/is/are",
    "人称代词与物主代词",
    "指示代词 this/that/these/those",
    "名词单复数",
    "冠词 a/an/the",
    "介词 in/on/under",
    "一般现在时 do/does",
    "第三人称单数 -s",
    "情态动词 can",
    "疑问词 what/who/where/how much",
    "序数词与日期",
    "可数与不可数名词",
    "连词 and/but/because",
    "形容词性物主代词",
    "祈使句",
    "there be 句型",
    "现在进行时",
    "一般过去时",
]

RULE_KINDS = ("cloze", "choice")
_BLANKS = ("___", "____", "_____", "…", "...", "［］", "[]")
_WS = re.compile(r"\s+")
_PUNCT = "。，,.!?！？;；:'\"“”‘’"
_CONTRACTIONS = {
    "isn't": "is not",
    "aren't": "are not",
    "don't": "do not",
    "doesn't": "does not",
    "can't": "cannot",
    "it's": "it is",
    "that's": "that is",
    "what's": "what is",
    "where's": "where is",
    "i'm": "i am",
    "he's": "he is",
    "she's": "she is",
    "they're": "they are",
    "we're": "we are",
    "you're": "you are",
    "let's": "let us",
}


def normalize(text: Any) -> str:
    """归一化答案：小写、展开常见缩写、去标点、合并空白。"""
    s = str(text or "").strip().lower()
    s = s.replace("’", "'").replace("‘", "'")
    for short, full in _CONTRACTIONS.items():
        s = re.sub(r"\b" + re.escape(short) + r"\b", full, s)
    s = s.strip(_PUNCT)
    s = re.sub(r"[.!?,;:'\"]+", " ", s)
    return _WS.sub(" ", s).strip()


def accepted_answers(item: dict[str, Any]) -> list[str]:
    """题目允许的答案（answer + accept），已归一化。"""
    out: list[str] = []
    raw: list[Any] = [item.get("answer")]
    raw += list(item.get("accept") or [])
    for cand in raw:
        if cand is None:
            continue
        n = normalize(cand)
        if n and n not in out:
            out.append(n)
    return out


def has_blank(text: str) -> bool:
    return any(b in str(text or "") for b in _BLANKS)


def near_miss(got: str, expected: str) -> bool:
    """内容对但形式不对：复数 -s、大小写粘连、多/少空格之类。"""
    if not got or not expected:
        return False
    if got.replace(" ", "") == expected.replace(" ", ""):
        return True
    g, e = got.strip(), expected.strip()
    if g.rstrip("s") == e or e.rstrip("s") == g:
        return True
    if g.endswith("es") and g[:-2] == e:
        return True
    if e.endswith("es") and e[:-2] == g:
        return True
    return False


def judge(item: dict[str, Any], answer: Any) -> dict[str, Any]:
    """判一道语法题，返回与 summarize() 同构的结果。

    overall 只有三档：100（对）/ 60（形式有误）/ 0（错）。
    """
    kind = str(item.get("kind") or "")
    expected = str(item.get("answer") or "")
    got_raw = str(answer or "").strip()
    got = normalize(got_raw)
    cands = accepted_answers(item)
    ok = got in cands
    near = False
    if not ok and got:
        near = any(near_miss(got, c) for c in cands)
    score = 100.0 if ok else (60.0 if near else 0.0)

    point = str(item.get("grammar_point") or "").strip() or "未标注知识点"
    if ok:
        hint = ""
    elif near:
        hint = f"意思对但形式写错了：正确写法是 “{expected}”。"
    else:
        hint = str(item.get("hint") or "").strip()
        if not hint:
            hint = f"这里应该填 “{expected}”{f'，你写的是 “{got_raw}”' if got_raw else '，你没有作答'}。"

    return {
        "overall": score,
        "dims": {},
        "multi_dim": {},
        "content_valid": True,
        "words": [],
        "errors": [],
        "phones_detail": [],
        "rule": {
            "kind": kind,
            "correct": bool(ok),
            "near_miss": bool(near),
            "expected": expected,
            "actual": got_raw,
            "grammar_point": point,
            "hint": hint,
        },
        "raw": {"item_id": item.get("id"), "answer": got_raw},
    }


def summarize_rule(item: dict[str, Any], answer: Any) -> dict[str, Any]:
    """对外统一入口（名字与 scoring.summarize 对齐）。"""
    return judge(item, answer)


def validate_item(item: dict[str, Any]) -> str:
    """校验一道语法题是否可用，返回空串表示通过。"""
    kind = str(item.get("kind") or "")
    if kind not in RULE_KINDS:
        return f"kind 不是 {RULE_KINDS}"
    text = str(item.get("text") or "").strip()
    if not text:
        return "缺少 text"
    if not str(item.get("answer") or "").strip():
        return "缺少 answer"
    if kind == "cloze":
        if not has_blank(text):
            return "填空题题干里没有空格标记 ___"
        if text.count("___") != 1:
            return "填空题必须有且只有一个 ___"
    if kind == "choice":
        opts = [str(o).strip() for o in (item.get("options") or []) if str(o).strip()]
        if len(opts) < 2:
            return "选择题至少要有 2 个选项"
        if normalize(item.get("answer")) not in [normalize(o) for o in opts]:
            return "选择题的 answer 不在 options 里"
        item["options"] = opts
    if len(text) > 120:
        return "题干过长"
    return ""


def point_of(item: dict[str, Any]) -> str:
    return str(item.get("grammar_point") or "").strip() or "未标注知识点"


def point_display(item: dict[str, Any]) -> str:
    """知识点展示文案（推荐理由与报告用）。"""
    return point_of(item)
