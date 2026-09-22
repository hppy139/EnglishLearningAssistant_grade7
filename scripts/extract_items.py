#!/usr/bin/env python3
"""从七年级上册教材正文里抽取练习题（句子 / 段落 / 问答题）。

与 scripts/build_rag.py 的区别：
  - build_rag.py：全文按 600 字滑窗切片，**只用于检索溯源**，句子常被截断，不能当题；
  - 本脚本：**按页重新提取原文**再分句，产出可直接朗读/作答的完整题目。

流程：
  按页提取 → 逐页判断所属单元（向后继承） → 清中文/音标/行首编号
  → 分句 → 过滤教材指令语（Listen and repeat. 之类） → 去重
  → 分三类：sentence（陈述句）/ qa（疑问句，即 semi_open）/ paragraph（连续短文）

产出：data/parts/extract_<unit>.json，交给 scripts/build_corpus.py 入库。

用法：
  python3 scripts/extract_items.py                 # 抽取并落盘
  python3 scripts/extract_items.py --dry-run       # 只看统计与样本，不落盘
  python3 scripts/extract_items.py --pages 6-89 --sample 20
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PARTS = DATA / "parts"
PDF = DATA / "【人教版】七年级上册英语电子课本-社学整理.pdf"
BOOK = "七上"

# 单元标题：页首常见 "STARTER UNIT 1" / "UNIT 3" / "Unit 3"
UNIT_RE = re.compile(r"STARTER\s+UNIT\s+([123])|UNIT\s+(1[012]|[1-9])", re.I)
# 教材里的中文、音标、特殊符号
NOISE_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\u0250-\u02af\u03b8\u0283\u02a7]"
                     r"|[ˌˈːˑ̩̯‖|—–・]")
# 行首的题号 / 板块号：1a、2b、3c、A1b
LEAD_NOISE = re.compile(r"^\s*(?:\d+\s*[a-d]\b|[A-Z]\d+[a-d]?\b|[一二三四五六七八九十]+[、.])\s*")
# 说话人标记：A: / Bob: / —
SPEAKER_RE = re.compile(r"(?:^|\s)(?:[A-Z][a-z]{1,10}|[A-B])\s*[:：]\s*")

# 教材指令语：句首出现这些动词基本都是练习指令，不是可练句子
INSTR_VERBS = {
    "listen", "match", "circle", "fill", "number", "complete", "practice", "practise",
    "read", "write", "look", "ask", "answer", "check", "repeat", "put", "make", "draw",
    "role-play", "roleplay", "pairwork", "groupwork", "work", "talk", "tell", "say",
    "choose", "underline", "color", "color", "tick", "guess", "act", "play", "sing",
    "copy", "translate", "discuss", "interview", "survey", "report", "use", "compare",
}
# 整句里出现这些就是板块标题/页眉，直接丢
PAGE_NOISE = (
    "grammar focus", "words and expressions", "vocabulary index", "notes on the text",
    "tapescripts", "self check", "pronunciation", "section a", "section b", "section",
    "starter unit", "unit 1", "unit 2", "unit 3", "unit 4", "unit 5", "unit 6", "unit 7",
    "unit 8", "unit 9", "unit 10", "unit 11", "unit 12", "go for it", "page",
    "copyright", "人民教育出版社", "英语", "七年级",
)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def page_texts(first: int, last: int) -> list[str]:
    raw = subprocess.run(
        ["pdftotext", "-layout", "-f", str(first), "-l", str(last), str(PDF), "-"],
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8", "ignore")
    return raw.split("\f")


def unit_of_page(text: str) -> str | None:
    head = text.strip()[:200]
    m = UNIT_RE.search(head)
    if not m:
        return None
    if m.group(1):
        return f"aS{m.group(1)}"
    return f"aU{m.group(2)}"


def clean_line(line: str) -> str:
    line = LEAD_NOISE.sub("", line.strip())
    line = NOISE_RE.sub(" ", line)
    line = SPEAKER_RE.sub(" ", line)
    line = re.sub(r"^[AB]\s+(?=[A-Z])", "", line)     # 对话行首的 A / B（不带冒号）
    line = re.sub(r"\s+", " ", line).strip()
    return line


def split_sentences(text: str) -> list[str]:
    """按 .!? 分句，保留结尾标点。"""
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for p in parts:
        p = p.strip()
        if p and re.search(r"[A-Za-z]", p):
            out.append(p)
    return out


def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


# 整行里出现这些就是教材指令/插图标签/板块信息，整行丢弃（比句首判断更严格）
BAD_PHRASES = (
    "language", "letters", "names", "match", "listen", "look", "circle", "fill",
    "number", "complete", "practice", "practise", "read", "write", "ask", "answer",
    "check", "repeat", "role-play", "pairwork", "groupwork", "talk about", "then ", "add ",
    "conversation", "section", "grammar", "pronunciation", "self check", "make ",
    "draw", "put ", "use ", "compare", "guess", "work in", "survey", "report",
    "words and expressions", "vocabulary", "tapescripts", "notes on",
)


# 句级过滤用：明确的教材指令/板块词（比行级宽松，避免误杀 "I use a pen." 这类真句子）
SENT_BAD = (
    "listen", "match", "circle", "fill in", "number the", "complete", "practice",
    "practise", "role-play", "pairwork", "groupwork", "talk about", "then ", "add ",
    "grammar", "section", "words and expressions", "tapescripts", "vocabulary",
    "notes on", "self check", "pronunciation", "language", "letters", "names",
    "conversation", "greet", "each other", "your partner", "the picture", "bring",
)

# 段落用：只保留明确的指令/板块词（BAD_PHRASES 里的 read/write/ask 等会误杀真短文）
PARA_BAD = (
    "listen", "match", "circle", "fill", "number the", "complete", "practice",
    "role-play", "pairwork", "groupwork", "grammar", "section", "words and expressions",
    "tapescripts", "notes on", "self check",
)


def line_ok(line: str) -> bool:
    """整行就是一句话才保留。

    pdftotext -layout 给出的是版面行，一行里常混着插图标签、题号、教学目标
    （如 "Grace Boys' names Good morning, Helen!"），整行丢弃比事后清洗可靠。
    """
    if not line:
        return False
    if not re.match(r"^[A-Z\"']", line):
        return False
    if not re.search(r"[.!?]\s*$", line):
        return False
    if re.search(r"\s[.!?]\s*$", line):   # 挖空残留："His birthday is on ."
        return False
    # 版式拼接：物主代词后直接接大写字母（"Is that your Are these your books?"）
    if re.search(r"\b(?:your|his|her|its|their|our)\s+[A-Z]", line):
        return False
    low = line.lower()
    if any(p in low for p in BAD_PHRASES):
        return False
    if ";" in line or "[" in line or "]" in line or "/" in line or "’s" in line and line.count("'") > 2:
        return False
    if "(" in line or ")" in line:      # 教材自带的选项 / 填空占位：(don't / doesn't)、( )
        return False
    words = WORD_RE.findall(line)
    if not (3 <= len(words) <= 24):
        return False
    # 孤立的单字母（a / I 除外）：字母表行、选项残留 "on h Match"、"Do you have a b"
    singles = [w for w in words if len(w) == 1 and w.lower() not in ("a", "i")]
    if len(singles) >= 1:
        return False
    # 连续两个大写单字母：字母表行 "Ss Tt Uu"、"Letters A H"
    if re.search(r"\b[A-Z]\s+[A-Z]\b", line):
        return False
    letters = len(re.findall(r"[A-Za-z]", line))
    if letters < len(line) * 0.6:
        return False
    return True


def keep(sentence: str, min_words: int, max_words: int) -> bool:
    s = sentence.strip()
    if not re.match(r"^[A-Z\"']", s):          # 教材句子首字母大写
        return False
    if not s.endswith((".", "!", "?")):
        return False
    if re.search(r"\s[.!?]\s*$", s):          # 挖空残留
        return False
    if re.search(r"\b(?:your|his|her|its|their|our)\s+[A-Z]", s):
        return False
    words = WORD_RE.findall(s)
    if not (min_words <= len(words) <= max_words):
        return False
    if "..." in s or "…" in s or "‛" in s or "ﬁ" in s or "ﬂ" in s:   # 残缺 / 连字
        return False
    if "(" in s or ")" in s:          # 教材自带的选项 / 填空占位
        return False
    if len(words[0]) == 1 and words[0] not in ("A", "I"):    # "G in English?" 这类残句
        return False
    # 版式拼接：句中小写词后直接接另一个句子开头（"What's this It's a map."）
    if re.search(r"[a-z,]\s+(?:It's|I'm|That's|This is|These are|Those are|He's|She's|What's|Where's)\b", s):
        return False
    low = s.lower()
    if any(n in low for n in PAGE_NOISE):
        return False
    if any(n in low for n in SENT_BAD):
        return False
    first = WORD_RE.findall(s)
    if first and first[0].lower().rstrip(".,") in INSTR_VERBS:
        return False
    # 英文占比过低（多为残留中文/数字）
    letters = len(re.findall(r"[A-Za-z]", s))
    if letters < len(s) * 0.5:
        return False
    # 连着三个数字/字母编号的垃圾
    if re.search(r"\b\d[a-d]\b", low):
        return False
    return True


def level_of(text: str) -> int:
    n = max((len(w) for w in WORD_RE.findall(text)), default=0)
    return 1 if n <= 5 else (2 if n <= 7 else 3)


def extract(args) -> dict[str, list[dict[str, Any]]]:
    first, last = args.pages.split("-")[0], args.pages.split("-")[1]
    pages = page_texts(int(first), int(last))
    by_unit: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    dropped = Counter()
    para_buffer: dict[str, list[str]] = {}

    para_rows: list[str] = []

    def flush_para(rows: list[str]) -> None:
        """把一页里通过严格过滤的句子合成一段（教材 2b/3a 短文 / 对话）。"""
        if args.no_paragraph:
            return
        block = " ".join(rows)
        if len(WORD_RE.findall(block)) < args.min_para_words:
            return
        bkey = norm_key(block)[:150]
        if bkey in seen:
            return
        seen.add(bkey)
        by_unit.setdefault(cur_unit, []).append(
            {"kind": "paragraph", "text": block, "page": page_no, "level": 2, "quality": "strict"}
        )

    cur_unit = ""
    page_no = int(first)
    for offset, page in enumerate(pages):
        page_no = int(first) + offset
        u = unit_of_page(page)
        if u:
            cur_unit = u
        if not cur_unit or not page.strip():
            continue
        # 清洗后的行 + 「整行就是一句话」的高质量行
        rows: list[str] = []
        strict: list[str] = []
        for line in page.splitlines():
            if not line.strip():
                continue
            cl = clean_line(line)
            if not cl:
                continue
            rows.append(cl)
            if line_ok(cl):
                strict.append(cl)
            else:
                dropped["行过滤"] += 1

        # ① 素材句子：按原文顺序，所有句子统一过一遍句级过滤
        ordered: list[tuple[str, str]] = []
        for row in rows:
            for s in split_sentences(row):
                if not keep(s, args.min_words, args.max_words):
                    dropped["句级过滤"] += 1
                    continue
                if row in strict:
                    ordered.append((s, "strict"))
                elif not args.strict_only:
                    ordered.append((s, "loose"))
                    dropped["宽松补充"] += 1

        for sent, quality in ordered:
            sent = sent.strip()
            key = norm_key(sent)
            if key in seen:
                dropped["重复"] += 1
                continue
            seen.add(key)
            kind = "semi_open" if sent.endswith("?") else "sentence"
            by_unit.setdefault(cur_unit, []).append(
                {
                    "kind": kind,
                    "text": sent,
                    "page": page_no,
                    "level": level_of(sent),
                    "quality": quality,
                }
            )

        # ② 段落素材：在素材句子上滑动窗口（可以是整段，也可以是其中几句）
        texts = [s for s, _ in ordered]
        if not args.no_paragraph and len(texts) >= args.para_sentences:
            made = 0
            for size in (args.para_sentences, min(args.para_sentences + 2, len(texts))):
                if size < args.para_sentences:
                    continue
                for i in range(0, len(texts) - size + 1, max(args.para_sentences, 1)):
                    if made >= args.max_paras_per_page:
                        break
                    flush_para(texts[i : i + size])
                    made += 1

    for u, items in by_unit.items():
        for i, it in enumerate(items, 1):
            it["id"] = f"{u}_x{i:03d}"
            it["origin"] = "教材正文抽取"
            it["book"] = BOOK
    print(f"抽取统计：保留 {sum(len(v) for v in by_unit.values())} 条"
          f"（行级过滤 {dropped['行过滤']}、句级过滤 {dropped['过滤掉']}、重复 {dropped['重复']}）")
    return by_unit


def main() -> int:
    ap = argparse.ArgumentParser(description="从七上教材正文抽取练习题")
    ap.add_argument("--pages", default="6-89", help="正文页码范围（PDF 物理页），默认 6-89")
    ap.add_argument("--min-words", type=int, default=3, help="句子最少词数（3 可保留 Good morning!）")
    ap.add_argument("--max-words", type=int, default=24, help="句子最多词数")
    ap.add_argument("--para-sentences", type=int, default=3, help="几句话凑一段（段落题）")
    ap.add_argument("--sample", type=int, default=12, help="打印几条样本")
    ap.add_argument("--no-paragraph", action="store_true", help="不抽段落题")
    ap.add_argument("--strict-only", action="store_true", help="只保留「整行就是一句话」的高质量句")
    ap.add_argument("--max-paras-per-page", type=int, default=3, help="每页最多生成几段")
    ap.add_argument("--min-para-words", type=int, default=18, help="一段最少多少词（教材句子短，3 句约 18 词）")
    ap.add_argument("--dry-run", action="store_true", help="只看统计与样本，不落盘")
    args = ap.parse_args()

    if not PDF.exists():
        print(f"找不到教材 PDF：{PDF}", file=sys.stderr)
        return 1

    by_unit = extract(args)
    print("\n各单元题量与题型：")
    for u in sorted(by_unit):
        c = Counter(i["kind"] for i in by_unit[u])
        print(f"  {u}: {len(by_unit[u])} 题  {dict(c)}")

    print(f"\n样本（每单元前几条）：")
    shown = 0
    for u in sorted(by_unit):
        for it in by_unit[u][:2]:
            if shown >= args.sample:
                break
            print(f"  [{u} p{it['page']} {it['kind']}] {it['text'][:88]}")
            shown += 1

    if args.dry_run:
        print("\n（dry-run，未落盘）")
        return 0

    PARTS.mkdir(parents=True, exist_ok=True)
    for u, items in by_unit.items():
        out = PARTS / f"extract_{u}.json"
        out.write_text(
            json.dumps(
                {"unit": u, "book": BOOK, "source": f"七上教材 PDF p{args.pages}", "items": items},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"\n落盘 {len(by_unit)} 个文件到 {PARTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
