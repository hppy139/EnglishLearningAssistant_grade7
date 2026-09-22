#!/usr/bin/env python3
"""生成语法题（填空 / 选择题）：人教版七年级上册教材 + 大模型仿教材难度生成。

思路：
  1. 用 RAG 从「七上」教材切片里取该单元的原文（优先 Grammar Focus / 对话）作为上下文；
  2. 让大模型**只基于这段原文**出题，词汇与句型不得超出教材；
  3. 本地做三重校验：结构校验（grammar.validate_item）、知识点受控表、词汇超纲检查；
  4. 落盘到 data/parts/grammar_<unit>.json，交给 scripts/build_corpus.py 并进语料。

用法：
  python3 scripts/gen_grammar_items.py --per-unit 3            # 七上全部单元
  python3 scripts/gen_grammar_items.py --units aU2,aU3 --per-unit 2
  python3 scripts/gen_grammar_items.py --units aU1 --dry-run   # 只打印不落盘
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.grammar import GRAMMAR_POINTS, RULE_KINDS, normalize, validate_item  # noqa: E402
from app.llm import LLM, last_error  # noqa: E402
from app.rag import search  # noqa: E402

CORPUS = ROOT / "data" / "seed_grade7.json"
OUT_DIR = ROOT / "data" / "parts"
BOOK = "七上"

SYSTEM = (
    "你是人教版初中英语教研员，要基于给定的【教材原文】编写语法练习题（文本填空 / 选择题）。\n"
    "硬约束：\n"
    "1. 只能用【教材原文】里出现过的单词、短语和句型，严禁引入超纲词汇或更复杂的语法；\n"
    "2. 每道题的 grammar_point 必须从这个列表里选一个：{points}；\n"
    "3. cloze（填空）题干里必须有 ___ 表示空，answer 是唯一正确答案（可给 accept 列出等价写法）；\n"
    "4. choice（选择）必须有 3-4 个 options，且 answer 必须在 options 中，干扰项要来自同学段常见混淆；\n"
    "5. 难度与教材相当：句子不超过 12 个单词，level 按难度填 1/2/3；\n"
    "6. hint 用中文写一句话解析（说明为什么是这个答案），面向 12-13 岁学生。\n"
    "只输出 JSON，不要解释："
    '{{"items":[{{"kind":"cloze","text":"This ___ my sister.","zh":"这是我姐姐。",'
    '"answer":"is","accept":["is"],"options":["am","is","are"],'
    '"grammar_point":"be动词 am/is/are","level":1,"hint":"this 是单数，用 is。"}}]}}'
)

USER = (
    "单元：{name}\n"
    "【教材原文】\n{context}\n\n"
    "【本单元词汇】\n{words}\n\n"
    "请生成 {n} 道题（cloze 与 choice 各占一半左右，覆盖不同 grammar_point）。"
)

_WORD_RE = re.compile(r"[a-z][a-z'\-]+")


def load_units() -> list[dict[str, Any]]:
    data = json.loads(CORPUS.read_text(encoding="utf-8"))
    return [u for u in data.get("units", []) if u.get("book") == BOOK]


def unit_words(unit: dict[str, Any], limit: int = 60) -> list[str]:
    words = [
        str(i.get("text", "")).strip().lower()
        for i in unit.get("items", [])
        if i.get("kind") == "word"
    ]
    return [w for w in words if w][:limit]


def context_of(unit: dict[str, Any], top_k: int = 4) -> str:
    name = str(unit.get("name") or "")
    query = f"{name} Grammar Focus"
    hits = search(query, top_k=top_k, book=BOOK)
    if not hits:
        hits = search(name, top_k=top_k, book=BOOK)
    return "\n".join(f"[p{h['page']}] {h['text'][:420]}" for h in hits)


def book_vocab() -> set[str]:
    """七上教材里的全部英文词（全书切片 + 各单元词表），作为「不超纲」的判定基准。

    只看本单元会误判（如 Starter Unit 1 里出现 fine 就被判超纲），
    教材里出现过就不算超纲。
    """
    words: set[str] = set()
    for chunk in search("English", top_k=400, book=BOOK):
        words |= {w for w in _WORD_RE.findall(str(chunk.get("text", "")).lower()) if len(w) > 1}
    try:
        for unit in load_units():
            words |= set(unit_words(unit, limit=400))
    except OSError:
        pass
    return words


def allowed_vocab(unit: dict[str, Any], context: str, book: set[str] | None = None) -> set[str]:
    """允许出现的英文词：全书词表 + 本单元词表 + 教材上下文里的词。"""
    words = set(book or ())
    words |= set(unit_words(unit, limit=300))
    words |= {w for w in _WORD_RE.findall(context.lower()) if len(w) > 1}
    return words


def off_topic(item: dict[str, Any], vocab: set[str]) -> str:
    """超纲检查：题干里的实词必须大部分出现在教材里。"""
    text = f"{item.get('text','')} {' '.join(str(o) for o in (item.get('options') or []))}"
    words = [w for w in _WORD_RE.findall(text.lower()) if len(w) > 2]
    if not words:
        return ""
    miss = []
    for w in words:
        if w in vocab:
            continue
        # where's / brothers / grandparents' 这类屈折形式不算超纲
        stem = w.rstrip("'").rstrip("s").rstrip("'").rstrip("e")
        if stem in vocab or stem.strip("-") in vocab:
            continue
        miss.append(w)
    if len(miss) / len(words) > 0.3:
        return f"超纲词汇过多：{', '.join(miss[:6])}"
    return ""


def clean_items(raw: list[dict[str, Any]], unit: dict[str, Any], vocab: set[str]) -> tuple[list[dict], list[str]]:
    """校验并规范化生成的题目，返回 (可用题, 被剔除的原因)。"""
    ok: list[dict[str, Any]] = []
    dropped: list[str] = []
    for n, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            dropped.append(f"第{n}题不是对象")
            continue
        kind = str(item.get("kind") or "").strip().lower()
        item["kind"] = kind
        if kind not in RULE_KINDS:
            dropped.append(f"第{n}题 kind={kind} 不支持")
            continue
        reason = validate_item(item)
        if reason:
            dropped.append(f"第{n}题 {reason}")
            continue
        reason = off_topic(item, vocab)
        if reason:
            dropped.append(f"第{n}题 {reason}")
            continue
        point = str(item.get("grammar_point") or "").strip()
        if point and point not in GRAMMAR_POINTS:
            # 不在受控表内就原样保留，但收敛成简短形式，避免画像被打散
            item["grammar_point"] = point[:20]
        item["id"] = f"{unit['id']}_g{n:02d}"
        item["unit_id"] = unit["id"]
        item["unit_name"] = unit.get("name")
        item["book"] = BOOK
        item["origin"] = "大模型生成（教材约束）"
        item["level"] = int(item.get("level") or 1)
        try:
            item["accept"] = [str(a) for a in (item.get("accept") or []) if str(a).strip()]
        except TypeError:
            item["accept"] = []
        ok.append(item)
    return ok, dropped


def gen_unit(unit: dict[str, Any], n: int, llm: LLM, book: set[str] | None = None) -> tuple[list[dict], list[str], str]:
    context = context_of(unit)
    if not context.strip():
        return [], [], "教材里检索不到该单元内容"
    vocab = allowed_vocab(unit, context, book)
    user = USER.format(
        name=unit.get("name"),
        context=context,
        words="、".join(unit_words(unit, limit=40)),
        n=n,
    )
    data = llm.chat_json(SYSTEM.format(points="、".join(GRAMMAR_POINTS)), user, temperature=0.4)
    if not data:
        return [], [], f"大模型没返回可用 JSON（{last_error() or '未知原因'}）"
    raw = data.get("items")
    if not isinstance(raw, list):
        return [], [], "返回里没有 items 数组"
    ok, dropped = clean_items(raw, unit, vocab)
    return ok, dropped, ""


def main() -> int:
    ap = argparse.ArgumentParser(description="基于七上教材 + 大模型生成语法题")
    ap.add_argument("--units", default="", help="只生成这些单元，逗号分隔，如 aU2,aU3；默认七上全部")
    ap.add_argument("--per-unit", type=int, default=3, help="每单元生成几道题（默认 3）")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    args = ap.parse_args()

    units = load_units()
    if args.units:
        want = {u.strip() for u in args.units.split(",") if u.strip()}
        units = [u for u in units if u["id"] in want]
    if not units:
        print("没有匹配的单元", file=sys.stderr)
        return 1

    llm = LLM()
    if not llm.available:
        print("没有配置 LLM_API_KEY，无法生成题目", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    book = book_vocab()
    print(f"七上教材词表：{len(book)} 个英文词")
    total = 0
    for unit in units:
        ok, dropped, err = gen_unit(unit, args.per_unit, llm, book)
        if err:
            print(f"{unit['id']} {unit.get('name')} -> 失败：{err}")
            continue
        print(f"{unit['id']} {unit.get('name')} -> {len(ok)} 题")
        for d in dropped:
            print(f"    剔除：{d}")
        for it in ok:
            opts = " / ".join(str(o) for o in (it.get("options") or []))
            print(f"    [{it['kind']}] {it['text']}  答案={it['answer']}"
                  f"{'  选项=' + opts if opts else ''}  知识点={it.get('grammar_point')}")
        total += len(ok)
        if not args.dry_run and ok:
            out = out_dir / f"grammar_{unit['id']}.json"
            out.write_text(json.dumps(ok, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"共生成 {total} 道题" + ("（dry-run，未落盘）" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
