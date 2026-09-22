#!/usr/bin/env python3
"""用「教材原句」生成填空语法题：让大模型删掉原句里的一个词。

思路（题目内容 100% 出自教材）：
  素材：scripts/extract_items.py 抽出来的教材原句（data/parts/extract_<unit>.json）
  出题：大模型只做一件事——删掉句中一个有语法意义的词，替换成 ___，
       其余单词一个字都不改；再配 3 个教材里出现过的干扰项。

产出：data/parts/cloze_text_<unit>.json，交给 scripts/build_corpus.py 入库。

用法：
  python3 scripts/make_cloze.py --units aU2 --per-unit 4      # 先试一个单元
  python3 scripts/make_cloze.py --per-unit 6                  # 七上全部单元
  python3 scripts/make_cloze.py --per-unit 6 --dry-run        # 只看不落盘
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

from app.grammar import GRAMMAR_POINTS, normalize, validate_item  # noqa: E402
from app.rag import search  # noqa: E402
# app.llm 依赖 httpx，离线模式下不需要，放到 main 里按需导入

PARTS = ROOT / "data" / "parts"
BOOK = "七上"
WORD_RE = re.compile(r"[a-z][a-z'\-]+")

SYSTEM = (
    "你是人教版初中英语教研员，要把教材原句改成「语法填空题」。\n"
    "硬约束：\n"
    "1. 只能【删除】原句中的一个单词并替换成 ___，其余单词、顺序、标点一律不许改动；\n"
    "2. 被删的词必须是有语法意义的词：be动词(am/is/are)、助动词(do/does)、介词(in/on/under/for)、"
    "物主代词(my/your/his/her)、指示代词(this/that/these/those)、疑问词(what/who/where/when/why/how much)、"
    "名词复数词尾、冠词(a/an/the)、连词(and/but/because)；不要删实义名词/动词造成无法作答；\n"
    "3. options 给 3-4 个：正确答案 + 2-3 个干扰项，干扰项必须来自【教材词表】；\n"
    "4. grammar_point 只能从给定列表里选一个；\n"
    "5. zh 给一句中文意思。\n"
    "只输出 JSON："
    '{{"items":[{{"index":0,"text":"This ___ my sister.","answer":"is",'
    '"options":["am","is","are"],"grammar_point":"be动词 am/is/are","zh":"这是我姐姐。"}}]}}'
)

USER = (
    "【教材原句】（编号从 0 开始）\n{sentences}\n\n"
    "【教材词表】（干扰项只能用这里的词）\n{words}\n\n"
    "请逐句改造成填空题，每句一题。"
)


# ---- 离线模式：不用大模型，按「只删一个语法词」的规则挖空 ----
# 每组 = (知识点, 该组可互换的词)。删一个词、用同组的词当干扰项，
# 题干其余部分一字不改，因此 100% 出自教材。
GROUPS: list[tuple[str, list[str]]] = [
    ("be动词 am/is/are", ["am", "is", "are"]),
    ("一般现在时 do/does", ["do", "does"]),
    ("情态动词 can", ["can"]),
    ("介词 in/on/under", ["in", "on", "under", "at", "for", "to", "with", "of", "from"]),
    ("形容词性物主代词", ["my", "your", "his", "her", "its", "our", "their"]),
    ("指示代词 this/that/these/those", ["this", "that", "these", "those"]),
    ("疑问词 what/who/where/how much", ["what", "who", "where", "when", "why", "how"]),
    ("冠词 a/an/the", ["a", "an", "the"]),
    ("连词 and/but/because", ["and", "but", "because", "so"]),
]


def offline_cloze(sentence: str) -> dict[str, Any] | None:
    """把一句教材原句挖空成填空题：删掉第一个「有语法意义」的词。"""
    import random

    tokens = sentence.split()
    words = [re.sub(r"[^A-Za-z']", "", t).lower() for t in tokens]
    for point, group in GROUPS:
        for i, w in enumerate(words):
            if w not in group or words.count(w) != 1:
                continue
            answer = re.sub(r"[^A-Za-z']", "", tokens[i])
            text = " ".join(tokens[:i] + ["___"] + tokens[i + 1 :])
            options = [w] + [o for o in group if o != w][:3]
            random.shuffle(options)
            if len(options) < 3:      # 干扰项不够（如 can 组只有 can / can't）→ 改纯填空，用输入框作答
                options = []
            return {
                "kind": "cloze",
                "text": text,
                "answer": answer,
                "accept": [answer, w],
                "options": options,
                "grammar_point": point,
                "level": 1 if len(words) <= 8 else 2,
                "zh": "",
                "hint": f"这里要用 {answer}（{point}）。",
            }
    return None


def offline_unit(unit: dict[str, Any], n: int, max_per_point: int = 2) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_point: dict[str, int] = {}
    for sent in source_sentences(unit, n * 8):
        item = offline_cloze(sent)
        if not item:
            continue
        point = str(item.get("grammar_point") or "")
        if per_point.get(point, 0) >= max_per_point:   # 同一知识点不扎堆
            continue
        key = normalize(item["text"])
        if key in seen:
            continue
        seen.add(key)
        per_point[point] = per_point.get(point, 0) + 1
        item["id"] = f"{unit['id']}_c{len(out):02d}"
        item["unit_id"] = unit["id"]
        item["unit_name"] = unit.get("name")
        item["book"] = BOOK
        item["source_sentence"] = sent
        item["origin"] = "教材原句挖空（规则）"
        out.append(item)
        if len(out) >= n:
            break
    # 一半做成单选题（选项够 2 个的挖空题天然就是选择题），保证定级卷题型齐全
    for k, it in enumerate(out):
        if (k + 1) % 2 == 0 and len(it.get("options") or []) >= 2:
            it["kind"] = "choice"
    return out


def load_units() -> list[dict[str, Any]]:
    corpus = json.loads((ROOT / "data" / "seed_grade7.json").read_text(encoding="utf-8"))
    return [u for u in corpus.get("units", []) if u.get("book") == BOOK]


def source_sentences(unit: dict[str, Any], limit: int) -> list[str]:
    """取该单元的教材原句：优先严格句，5-14 词，去重。"""
    path = PARTS / f"extract_{unit['id']}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    strict, loose = [], []
    for it in data.get("items") or []:
        if it.get("kind") == "paragraph":
            continue
        text = str(it.get("text") or "").strip()
        n = len(WORD_RE.findall(text.lower()))
        if not (5 <= n <= 14):
            continue
        # 版式拼接的残句（"We go to the same No, I don't."）不能拿来挖空
        if re.search(r"[a-z,]\s+(?:No|Yes|It's|I'm|That's|This is|He's|She's|What's|Where's)\b", text):
            continue
        # 教材自带的选项括号（I like fruit, but I (don't / doesn't) like vegetables.）跳过
        if "(" in text or ")" in text:
            continue
        if it.get("quality") == "strict":
            strict.append(text)
        else:
            loose.append(text)

    def uniq(seq: list[str]) -> list[str]:
        seen, out = set(), []
        for s in seq:
            k = normalize(s)
            if k and k not in seen:
                seen.add(k)
                out.append(s)
        return out

    picked = uniq(strict) + uniq(loose)
    return picked[:limit]


def book_vocab() -> set[str]:
    words: set[str] = set()
    for chunk in search("English", top_k=400, book=BOOK):
        words |= {w for w in WORD_RE.findall(str(chunk.get("text", "")).lower()) if len(w) > 1}
    try:
        for unit in load_units():
            for it in unit.get("items", []):
                if it.get("kind") == "word":
                    w = str(it.get("text", "")).strip().lower()
                    if w:
                        words.add(w)
    except OSError:
        pass
    return words


def unit_words(unit: dict[str, Any], limit: int = 60) -> list[str]:
    out = [
        str(i.get("text", "")).strip().lower()
        for i in unit.get("items", [])
        if i.get("kind") == "word"
    ]
    return [w for w in out if w][:limit]


def validate(item: dict[str, Any], sentence: str, vocab: set[str]) -> str:
    """校验一道填空题：答案必须真在原句里，干扰项必须来自教材。"""
    reason = validate_item(item)
    if reason:
        return reason
    src_words = {normalize(w) for w in WORD_RE.findall(sentence.lower())}
    if normalize(item.get("answer")) not in src_words:
        return "答案不在原句里（可能改写了句子）"
    opts = [normalize(o) for o in (item.get("options") or [])]
    if normalize(item.get("answer")) not in opts:
        return "答案不在选项里"
    wrong = [o for o in opts if o != normalize(item.get("answer"))]
    if len(wrong) < 2:
        return "干扰项少于 2 个"
    if wrong and sum(1 for o in wrong if o in vocab) / len(wrong) < 0.5:
        return "干扰项不在教材词汇里"
    # 题干除 ___ 外应与原句一致（允许大小写/空格差异）
    stem = normalize(item.get("text", "")).replace("___", "")
    if stem and stem not in normalize(sentence):
        return "题干被改写，不是原句"
    return ""


def gen_unit(unit: dict[str, Any], n: int, llm: LLM, vocab: set[str]) -> tuple[list[dict], list[str], str]:
    sentences = source_sentences(unit, n)
    if not sentences:
        return [], [], "该单元没有可用的教材原句"
    user = USER.format(
        sentences="\n".join(f"{i}. {s}" for i, s in enumerate(sentences)),
        words="、".join(unit_words(unit, limit=50)),
    )
    data = llm.chat_json(
        SYSTEM.format(points="、".join(GRAMMAR_POINTS)), user, temperature=0.3
    )
    if not data:
        return [], [], f"大模型没返回可用 JSON（{last_error() or '未知原因'}）"
    raw = data.get("items")
    if not isinstance(raw, list):
        return [], [], "返回里没有 items"
    ok, dropped = [], []
    for k, item in enumerate(raw):
        if not isinstance(item, dict):
            dropped.append(f"第{k}条不是对象")
            continue
        idx = item.get("index")
        src = sentences[idx] if isinstance(idx, int) and 0 <= idx < len(sentences) else ""
        if not src:
            dropped.append(f"第{k}条 index 无效")
            continue
        item["kind"] = "cloze"
        item["level"] = 1 if len(WORD_RE.findall(src)) <= 8 else 2
        reason = validate(item, src, vocab)
        if reason:
            dropped.append(f"「{src[:30]}」{reason}")
            continue
        item["id"] = f"{unit['id']}_c{k:02d}"
        item["unit_id"] = unit["id"]
        item["unit_name"] = unit.get("name")
        item["book"] = BOOK
        item["source_sentence"] = src
        item["origin"] = "教材原句挖空（大模型）"
        item["accept"] = [str(item.get("answer"))]
        ok.append(item)
    return ok, dropped, ""


def main() -> int:
    ap = argparse.ArgumentParser(description="用教材原句生成填空语法题")
    ap.add_argument("--units", default="", help="只处理这些单元，逗号分隔")
    ap.add_argument("--per-unit", type=int, default=6, help="每单元取几句原文出题")
    ap.add_argument("--max-per-point", type=int, default=2, help="每个知识点在同一单元最多出几题")
    ap.add_argument("--offline", action="store_true",
                    help="不调大模型，用规则挖空（余额不足/离线时用，同样 100%% 出自教材）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    units = load_units()
    if args.units:
        want = {u.strip() for u in args.units.split(",") if u.strip()}
        units = [u for u in units if u["id"] in want]

    if args.offline:
        total = 0
        for unit in units:
            ok = offline_unit(unit, args.per_unit, args.max_per_point)
            print(f"{unit['id']} {unit.get('name')} -> {len(ok)} 题")
            for it in ok[:3]:
                print(f"    {it['text']}  答案={it['answer']}  知识点={it.get('grammar_point')}")
            total += len(ok)
            if not args.dry_run and ok:
                out = PARTS / f"cloze_text_{unit['id']}.json"
                out.write_text(json.dumps(ok, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"共生成 {total} 道填空题（规则挖空）" + ("（dry-run）" if args.dry_run else ""))
        return 0

    from app.llm import LLM, last_error as llm_last_error  # noqa: PLC0415

    llm = LLM()
    if not llm.available:
        print("没有配置 LLM_API_KEY", file=sys.stderr)
        return 1
    vocab = book_vocab()
    print(f"教材词表：{len(vocab)} 词")

    total = 0
    for unit in units:
        ok, dropped, err = gen_unit(unit, args.per_unit, llm, vocab)
        if err:
            print(f"{unit['id']} -> 失败：{err}")
            continue
        print(f"{unit['id']} {unit.get('name')} -> {len(ok)} 题")
        for d in dropped[:3]:
            print(f"    剔除：{d}")
        for it in ok[:3]:
            print(f"    {it['text']}  答案={it['answer']}  知识点={it.get('grammar_point')}")
        total += len(ok)
        if not args.dry_run and ok:
            out = PARTS / f"cloze_text_{unit['id']}.json"
            out.write_text(json.dumps(ok, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"共生成 {total} 道填空题" + ("（dry-run）" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
