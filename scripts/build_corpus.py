#!/usr/bin/env python3
"""组装 data/seed_grade7.json（七年级教材语料）。"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PARTS = DATA / "parts"
WORK = DATA / "_work"
OUT = DATA / "seed_grade7.json"
UP_PDF = DATA / "【人教版】七年级上册英语电子课本-社学整理.pdf"
UP_TXT = WORK / "g7a.txt"

UP_UNITS = [
    ("aS1", "Starter Unit 1 Good morning!", "greetings"),
    ("aS2", "Starter Unit 2 What's this in English?", "objects"),
    ("aS3", "Starter Unit 3 What color is it?", "colors"),
    ("aU1", "Unit 1 My name's Gina.", "making_friends"),
    ("aU2", "Unit 2 This is my sister.", "family"),
    ("aU3", "Unit 3 Is this your pencil?", "school_things"),
    ("aU4", "Unit 4 Where's my schoolbag?", "things_around"),
    ("aU5", "Unit 5 Do you have a soccer ball?", "sports"),
    ("aU6", "Unit 6 Do you like bananas?", "food"),
    ("aU7", "Unit 7 How much are these socks?", "shopping"),
    ("aU8", "Unit 8 When is your birthday?", "dates"),
    ("aU9", "Unit 9 My favorite subject is science.", "subjects"),
]

UP_MAP = {
    "Starter Unit 1": "aS1", "Starter Unit 2": "aS2", "Starter Unit 3": "aS3",
    "Unit 1": "aU1", "Unit 2": "aU2", "Unit 3": "aU3", "Unit 4": "aU4",
    "Unit 5": "aU5", "Unit 6": "aU6", "Unit 7": "aU7", "Unit 8": "aU8",
    "Unit 9": "aU9",
}

ENTRY = re.compile(
    r"^([A-Za-z][A-Za-z'\- ]{0,20}?)\s*/[^/]*/\s*(?:[a-zA-Z]+\.\s*)?([一-鿿][^，,]{0,20})"
)


def up_words() -> dict[str, list[tuple[str, str]]]:
    """解析上册词表（PDF 自带文本层）。"""
    if not UP_TXT.exists():
        WORK.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pdftotext", "-layout", str(UP_PDF), str(UP_TXT)], check=True)
    text = UP_TXT.read_text(encoding="utf-8")
    out: dict[str, list[tuple[str, str]]] = {}
    cur = ""
    pos = text.find("Words and Expressions", 20000)
    end = text.find("Vocabulary Index", pos + 100)
    for line in text[pos:end if end > 0 else len(text)].splitlines():
        for seg0 in re.split(r"\s{2,}", line):
            s = seg0.strip()
            hit = re.match(r"^(Starter Unit [123]|Unit [1-9])$", s)
            if hit:
                cur = UP_MAP[hit.group(1)]
                continue
        for seg in re.split(r"\s{2,}", line):
            m = ENTRY.match(seg.strip())
            if not m or not cur:
                continue
            en = m.group(1).strip().lower()
            zh = m.group(2).strip()
            if len(en) < 2 or len(en.split()) > 2:
                continue
            if "（男名）" in zh or "（女名）" in zh or "（姓）" in zh:
                continue
            if (en, zh) not in out.setdefault(cur, []):
                out[cur].append((en, zh))
    return out


def part_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "|" in line:
            en, _, zh = line.rpartition("|")
            if en.strip():
                pairs.append((en.strip(), zh.strip()))
    return pairs


def _bad_text(text: str) -> bool:
    """教材里的括号都是选项 / 填空占位 / 补充成分，含括号的句子不适合直接出题。

    例：I like fruit, but I (don't / doesn't) like vegetables.
        When is Children's Day ( )?
    """
    s = str(text or "")
    return "(" in s or ")" in s


def _norm(text: str) -> str:
    """归一化文本（小写 + 去标点），用于比对「填空后是否还原成原句」。"""
    return re.sub(r"[^a-z0-9 ]", "", str(text or "").lower()).strip()


def level_of(text: str) -> int:
    n = max((len(w) for w in text.split()), default=0)
    return 1 if n <= 5 else (2 if n <= 7 else 3)


def mk(kind: str, uid: str, n: int, en: str, zh: str, **extra) -> dict:
    it = {"id": f"{uid}_{kind}{n:02d}", "kind": kind, "text": en, "zh": zh}
    it.update(extra)
    it["level"] = level_of(en)
    return it


def glob_pairs(pattern: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for path in sorted(PARTS.glob(pattern)):
        pairs.extend(part_pairs(path))
    return pairs


def build_unit(uid: str, name: str, topic: str, book: str, words, bad_sentences=frozenset()) -> dict:
    items = [mk("word", uid, i, w, z) for i, (w, z) in enumerate(words, 1)]
    # 口语选择题改名 oral_choice：choice 留给「语法选择题」（规则判分）
    for kind, pat in (("sentence", "sent"), ("paragraph", "para"),
                      ("semi_open", "qa"), ("oral_choice", "choice")):
        for i, (en, z) in enumerate(glob_pairs(f"{pat}_{uid}.txt"), 1):
            items.append(mk(kind, uid, i, en, z))
    # 语法题（填空 / 选择）：大模型基于教材生成，字段比 txt 多，用 JSON
    gp = PARTS / f"grammar_{uid}.json"
    if gp.exists():
        try:
            raw_items = json.loads(gp.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  ! {gp.name} 解析失败：{exc}", file=sys.stderr)
            raw_items = []
        for i, it in enumerate(raw_items, 1):
            item = dict(it)
            item.setdefault("id", f"{uid}_g{i:02d}")
            item.setdefault("level", level_of(str(item.get("text", ""))))
            item.setdefault("book", book)
            # 大模型生成的题尚未复核（出现过答案与句子语法不符的情况），
            # 标记 verified=False，组卷/推荐时会跳过；跑 scripts/verify_items.py 复核后转正。
            item.setdefault("verified", False)
            items.append(item)
    # 教材原句挖空的填空题（scripts/make_cloze.py 产出）
    cp = PARTS / f"cloze_text_{uid}.json"
    if cp.exists():
        try:
            cloze = json.loads(cp.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  ! {cp.name} 解析失败：{exc}", file=sys.stderr)
            cloze = []
        for i, it in enumerate(cloze, 1):
            item = dict(it)
            item.setdefault("id", f"{uid}_c{i:02d}")
            item.setdefault("level", level_of(str(item.get("text", ""))))
            item.setdefault("book", book)
            # 知识点以「答案是什么」为准，避免标注漂移（答案 is 却标成指示代词）
            ans = str(item.get("answer") or "").strip().lower()
            if ans in ("am", "is", "are"):
                item["grammar_point"] = "be动词 am/is/are"
            elif ans in ("do", "does"):
                item["grammar_point"] = "一般现在时 do/does"
            elif ans in ("a", "an", "the"):
                item["grammar_point"] = "冠词 a/an/the"
            # 原句挖空：把答案填回去必须还原成教材原句；
            # 已被 scripts/verify_items.py 判为 False 的，保持 False（不再被自动置 True）
            src = str(item.get("source_sentence") or "")
            filled = str(item.get("text") or "").replace("___", str(item.get("answer") or ""))
            if item.get("verified") is not False:
                item["verified"] = bool(src) and _norm(filled) == _norm(src)
            items.append(item)
    # 教材正文抽取的句子 / 段落 / 问答题（scripts/extract_items.py 产出）
    ep = PARTS / f"extract_{uid}.json"
    if ep.exists():
        try:
            ex = json.loads(ep.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  ! {ep.name} 解析失败：{exc}", file=sys.stderr)
            ex = {}
        for it in ex.get("items") or []:
            # 被大模型复核判为不通顺的原句，也不再当朗读题
            if _norm(str(it.get("text", ""))) in bad_sentences:
                continue
            item = dict(it)
            item.setdefault("level", level_of(str(item.get("text", ""))))
            item.setdefault("book", book)
            items.append(item)
    # 含括号的（选项 / 填空占位）一律不入库
    items = [it for it in items if not _bad_text(it.get("text"))]
    # 去重：手写的 Target Language 与正文抽取会有重叠，按文本保留首次出现的
    seen: set[str] = set()
    uniq: list[dict] = []
    for it in items:
        key = re.sub(r"[^a-z0-9 ]", "", str(it.get("text", "")).lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(it)
    return {"id": uid, "name": name, "topic": topic, "book": book, "items": uniq}


def main() -> int:
    # 大模型复核判不合格的题，其原句一并拉黑（scripts/verify_items.py 产出）
    bad_sentences: set[str] = set()
    rj = PARTS / "rejected_grammar.json"
    if rj.exists():
        try:
            for row in json.loads(rj.read_text(encoding="utf-8")):
                src = str(row.get("source_sentence") or "").strip()
                if src:
                    bad_sentences.add(_norm(src))
        except json.JSONDecodeError as exc:
            print(f"  ! rejected_grammar.json 解析失败：{exc}", file=sys.stderr)
    if bad_sentences:
        print(f"拉黑不合格原句 {len(bad_sentences)} 条")

    up = up_words()
    units = []
    for uid, name, topic in UP_UNITS:
        units.append(build_unit(uid, name, topic, "七上", up.get(uid, []), bad_sentences))
    corpus = {
        "meta": {"grade": 7, "title": "人教版 Go for it! 七年级上册教材语料",
                 "sources": ["七上 PDF 文本层解析", "目录 Target Language"]},
        "units": units,
    }
    OUT.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {OUT}：{len(units)} 单元 / {sum(len(u['items']) for u in units)} 题")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
