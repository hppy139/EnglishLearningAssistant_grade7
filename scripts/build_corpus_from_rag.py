#!/usr/bin/env python3
"""基于 data/rag/chunks.jsonl（大模型解析的教材切片）重建 data/seed_grade7.json。

为什么另起一个脚本（而不是改 scripts/build_corpus.py）：
  原脚本从 data/parts/*.txt（pdftotext 解析产物）组装，正文句子与词表在解析时有出入；
  本脚本改用大模型解析好的教材切片重建，单词与句子更贴近原文。

产出结构与 build_corpus.py 完全一致（id / kind / text / zh / level / grammar_point / answer /
options / sample / verified），所以 app/corpus.py 与整条推荐链路都无需改动。

能从切片直接重建的题型：
  word       ← Words and Expressions in Each Unit（词表页；条目会跨行，先重组再解析）
  sentence   ← 单元正文 + Tapescripts（听力稿的对话原句质量最高）
  semi_open  ← Tapescripts 里的 A:/B: 问答对（问句当题干，答句当参考答案）
  paragraph  ← 正文里连续 ≥3 个完整句子的英文块（版面噪声多，门槛设得较严）

切片里没有、需要迁移旧库的部分：
  cloze / choice（语法题）  ← Grammar 版块是讲解表不是题，原样迁移旧库
  paragraph（可选）         ← 版面难抽，可叠加旧库已有的段落题
  七下单元                 ← chunks 只有七上 138 页，旧库若有则整单元迁移

用法：
  python3 scripts/build_corpus_from_rag.py --dry-run        # 只统计，不写文件
  python3 scripts/build_corpus_from_rag.py                  # 重建（自动备份旧库为 .bak）
  python3 scripts/build_corpus_from_rag.py --sample         # 打印各题型抽样，检查质量
  python3 scripts/build_corpus_from_rag.py --no-grammar     # 不迁移旧语法题
  python3 scripts/build_corpus_from_rag.py --no-para-old    # 不叠加旧库段落题
  python3 scripts/build_corpus_from_rag.py --no-down        # 不迁移七下单元
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CHUNKS = DATA / "rag" / "chunks.jsonl"
OUT = DATA / "seed_grade7.json"

# 七上单元表（与 scripts/build_corpus.py 保持一致）
UP_UNITS: list[tuple[str, str, str]] = [
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

# chunks 的 unit 字段在以下几类里装的是「版块名」而不是单元号
SEC_WORDS = "Words and Expressions in Each Unit"
SEC_TAPES = "Tapescripts"
RE_UNIT_SLOT = re.compile(r"^(S[123]|[1-9])$")     # 正文页：S1 / 1 / 2 ...

CJK = re.compile(r"[\u4e00-\u9fff]")
# 词表条目：good / gʊd / adj. 好的
ENTRY = re.compile(
    r"^([A-Za-z][A-Za-z'\- ]{0,20}?)\s*/[^/]*/\s*(?:[a-zA-Z]+\.\s*)?([一-鿿][^，,]{0,20})"
)
RE_ENTRY_HEAD = re.compile(r"^[A-Za-z][A-Za-z'\- ]{0,20}?\s*/")
RE_UNIT_LINE = re.compile(r"^\s*(STARTER UNIT [123]|UNIT [1-9]|Starter Unit [123]|Unit [1-9])\b")
RE_PAGE_MARK = re.compile(r"^\s*p\.\s*[S]?\d+\s*$", re.I)
PAGE_TAIL = re.compile(r"\s*p\.\s*[S]?\d+\s*$", re.I)
# 说话人前缀：Bob: / Girl 1: / A: / B:
SPEAKER = re.compile(r"^\s*(?:[A-Z][A-Za-z]*(?:\s+\d+)?|[A-Z])\s*:\s*")
# 教材指令句与栏目抬头（不适合当朗读题）
RE_INSTRUCTION = re.compile(
    r"^(Listen|Write|Practice|Practise|Read|Match|Complete|Fill|Look|Ask|Answer|Repeat|"
    r"Number|Circle|Check|Draw|Make|Put|Work|Talk|Role-play|Pairwork|Groupwork|Then|Now|"
    r"Language Goals|语言目标|Self Check)\b",
    re.I,
)
RE_NOISE_HEAD = re.compile(
    r"^(Boys'?\s+names|Girls'?\s+names|Names|Words and Expressions|Vocabulary Index|"
    r"Notes on the Text|Tapescripts|Pronunciation|Grammar|Contents|Check)\b",
    re.I,
)
RE_SLOT = re.compile(r"^\s*\d[a-z]\s*$")                       # 题号：1a / 2b
RE_LETTERS = re.compile(r"^[A-Za-z](?:\s+[A-Za-z]){2,}\s*$")   # 字母表：A H J K
RE_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
RE_ENGLISH = re.compile(r"^[A-Za-z][A-Za-z0-9 ,.'’!?\-:]{6,160}$")


def level_of(text: str) -> int:
    n = max((len(w) for w in text.split()), default=0)
    return 1 if n <= 5 else (2 if n <= 7 else 3)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(text or "").lower()).strip()


def _bad_text(text: str) -> bool:
    """含括号的一律不要：教材里的括号是选项 / 填空占位 / 补充成分。"""
    return "(" in str(text or "") or ")" in str(text or "")


def mk(kind: str, uid: str, n: int, en: str, zh: str = "", **extra) -> dict:
    it = {"id": f"{uid}_{kind}{n:02d}", "kind": kind, "text": en, "zh": zh}
    it.update(extra)
    it["level"] = level_of(en)
    return it


# ---------- 读取与单元归属 ----------
def load_chunks() -> list[dict]:
    if not CHUNKS.exists():
        print(f"找不到 {CHUNKS}，请先准备教材切片", file=sys.stderr)
        return []
    rows: list[dict] = []
    for line in CHUNKS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def split_by_unit_title(text: str) -> list[tuple[str, str]]:
    """按行首单元标题把文本切成 [(标题, 该段文本)]。

    词表页与听力稿页一页可能含多个单元，必须逐段归属，否则整页都算给第一个单元。
    """
    blocks: list[tuple[str, str]] = []
    cur = ""
    buf: list[str] = []
    for line in text.splitlines():
        m = RE_UNIT_LINE.match(line)
        if m:
            if buf and cur:
                blocks.append((cur, "\n".join(buf)))
            cur = " ".join(m.group(1).split())
            buf = []
            continue
        buf.append(line)
    if buf and cur:
        blocks.append((cur, "\n".join(buf)))
    return blocks


def canon_unit(title: str) -> str:
    """单元标题 → uid（Starter Unit 1 / Unit 1 → aS1 / aU1）。"""
    t = " ".join(str(title or "").split())
    for k, v in UP_MAP.items():
        if t.lower() == k.lower():
            return v
    m = re.search(r"(Starter Unit [123]|Unit [1-9])", t, re.I)
    if m:
        key = " ".join(m.group(1).split())
        for k, v in UP_MAP.items():
            if k.lower() == key.lower():
                return v
    return ""


# ---------- 词表：条目会跨行，先重组再解析 ----------
def word_entries(text: str) -> list[str]:
    """把词表页文本重组成一条条「条目」（音标/释义换行时合并到同一条）。"""
    out: list[str] = []
    cur = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or RE_PAGE_MARK.match(line):
            continue
        if RE_UNIT_LINE.match(line):
            if cur:
                out.append(cur)
                cur = ""
            continue
        if RE_ENTRY_HEAD.match(line):        # 新条目：英文 + 音标
            if cur:
                out.append(cur)
            cur = line
            continue
        # 续行：仅当当前条目还没有中文释义时才合并（音标后半 / 词性释义换行）
        if cur and not CJK.search(cur):
            cur = cur + " " + line
        elif cur:
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


def parse_words(text: str) -> list[tuple[str, str]]:
    """词表页 → [(英文, 中文)]。"""
    out: list[tuple[str, str]] = []
    for entry in word_entries(text):
        e = PAGE_TAIL.sub("", entry).strip()
        e = e.replace("（男名）", "").replace("（女名）", "").replace("（姓）", "")
        m = ENTRY.match(e)
        if not m:
            continue
        en = m.group(1).strip().lower()
        zh = re.sub(r"\s*p\.\s*[S]?\d+$", "", m.group(2).strip(), flags=re.I).strip()
        if len(en) < 2 or len(en.split()) > 2 or not zh:
            continue
        if (en, zh) not in out:
            out.append((en, zh))
    return out


# ---------- 正文：先按行去噪，再合并硬换行，最后按句切 ----------
def _line_ok(line: str) -> bool:
    """这一行能不能进语料（丢弃题号 / 栏目 / 字母表 / 孤立单词 / 指令 / 中文）。"""
    if not line or CJK.search(line):
        return False
    if RE_SLOT.match(line) or RE_UNIT_LINE.match(line) or RE_NOISE_HEAD.match(line):
        return False
    if RE_LETTERS.match(line) or RE_PAGE_MARK.match(line):
        return False
    if _bad_text(line) or "___" in line:
        return False
    if RE_INSTRUCTION.match(line):
        return False
    # 孤立单词：正文里成片的 Bob / Dale / quilt 是人名词或单词框，会粘进句子
    if len(line.split()) == 1 and not re.search(r"[.!?]$", line):
        return False
    return True


def blocks_of(text: str) -> list[str]:
    """把一页文本整理成若干「连续英文块」（空行/噪声行处分段）。"""
    blocks: list[str] = []
    cur: list[str] = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line:
            if cur:
                blocks.append(" ".join(cur))
                cur = []
            continue
        if not _line_ok(line):
            if cur:
                blocks.append(" ".join(cur))
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append(" ".join(cur))
    return [re.sub(r"\s+", " ", b).strip() for b in blocks if b.strip()]


def _sent_ok(s: str) -> bool:
    if not s or CJK.search(s) or _bad_text(s) or "___" in s:
        return False
    if RE_INSTRUCTION.match(s) or RE_NOISE_HEAD.match(s) or RE_UNIT_LINE.match(s):
        return False
    if not RE_ENGLISH.match(s):
        return False
    words = s.split()
    if len(words) < 3 or len(words) > 25:
        return False
    if not re.search(r"[.!?]$", s):
        return False
    if len(re.findall(r"[A-Za-z]", s)) < len(s) * 0.6:
        return False
    return True


def strip_speaker(s: str) -> str:
    return SPEAKER.sub("", str(s or "")).strip()


def extract_sentences(text: str) -> list[str]:
    """正文/听力稿 → 句子列表（去掉说话人前缀）。"""
    out: list[str] = []
    for block in blocks_of(text):
        for piece in RE_SENT_SPLIT.split(block):
            s = strip_speaker(piece.strip())
            if _sent_ok(s):
                out.append(s)
    return out


def extract_paragraphs(text: str, min_sent: int = 3, min_chars: int = 90) -> list[str]:
    """连续块里含 ≥min_sent 个完整句子的，作为段落朗读题。"""
    paras: list[str] = []
    for block in blocks_of(text):
        sents = [strip_speaker(p.strip()) for p in RE_SENT_SPLIT.split(block)]
        sents = [s for s in sents if _sent_ok(s)]
        if len(sents) < min_sent:
            continue
        joined = " ".join(sents).strip()
        if len(joined) < min_chars or _bad_text(joined):
            continue
        paras.append(joined)
    return paras


def extract_qa(text: str) -> list[tuple[str, str]]:
    """听力稿里的问答对 → [(问题, 参考答案)]。"""
    items: list[tuple[str, str]] = []
    lines: list[str] = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if not line or CJK.search(line) or RE_PAGE_MARK.match(line):
            lines.append("")
            continue
        lines.append(strip_speaker(line))
    lines = [l for l in lines if l and not CJK.search(l)]
    for i in range(len(lines) - 1):
        q, a = lines[i], lines[i + 1]
        if "?" not in q or not _sent_ok(q):
            continue
        if CJK.search(a) or _bad_text(a) or not re.search(r"[.!?]$", a):
            continue
        if len(a.split()) < 2 or len(a.split()) > 25:
            continue
        pair = (q, a)
        if pair not in items:
            items.append(pair)
    return items


# ---------- 组装 ----------
def build_up_units(chunks: list[dict]) -> tuple[list[dict], dict]:
    """从切片重建七上 12 个单元，返回 (units, 统计)。"""
    words: dict[str, list[tuple[str, str]]] = {uid: [] for uid, _, _ in UP_UNITS}
    sents: dict[str, list[str]] = {uid: [] for uid, _, _ in UP_UNITS}
    paras: dict[str, list[str]] = {uid: [] for uid, _, _ in UP_UNITS}
    qas: dict[str, list[tuple[str, str]]] = {uid: [] for uid, _, _ in UP_UNITS}
    stat = Counter()

    for ch in chunks:
        slot = str(ch.get("unit") or "").strip()
        text = str(ch.get("text") or "")
        if not text.strip():
            continue

        if RE_UNIT_SLOT.match(slot):                       # ① 单元正文页
            uid = "aS" + slot[1] if slot.startswith("S") else "aU" + slot
            if uid not in words:
                continue
            stat["正文页"] += 1
            sents[uid].extend(extract_sentences(text))
            paras[uid].extend(extract_paragraphs(text))
            qas[uid].extend(extract_qa(text))
        elif slot == SEC_WORDS:                            # ② 词表页
            stat["词表页"] += 1
            for title, block in split_by_unit_title(text):
                uid = canon_unit(title)
                if uid:
                    words[uid].extend(parse_words(block))
        elif slot == SEC_TAPES:                            # ③ 听力稿页（句子质量最高）
            stat["听力稿页"] += 1
            for title, block in split_by_unit_title(text):
                uid = canon_unit(title)
                if uid:
                    sents[uid].extend(extract_sentences(block))
                    qas[uid].extend(extract_qa(block))
        else:                                              # ④ 其余版块不出题
            stat["skip"] += 1

    units: list[dict] = []
    for uid, name, topic in UP_UNITS:
        items: list[dict] = []
        for i, (w, z) in enumerate(words[uid], 1):
            items.append(mk("word", uid, i, w, z))
        for i, s in enumerate(sents[uid], 1):
            items.append(mk("sentence", uid, i, s))
        for i, p in enumerate(paras[uid], 1):
            items.append(mk("paragraph", uid, i, p))
        for i, (q, a) in enumerate(qas[uid], 1):
            items.append(mk("semi_open", uid, i, q, sample=a))
        units.append(
            {"id": uid, "name": name, "topic": topic, "book": "七上", "items": dedup(items)}
        )
    return units, dict(stat)


def dedup(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for it in items:
        text = str(it.get("text") or "")
        if _bad_text(text):
            continue
        key = _norm(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def migrate(
    units: list[dict], old: dict | None, keep_words: bool, keep_para: bool, keep_down: bool
) -> tuple[list[dict], dict]:
    """迁移旧库：语法题、段落题、应答/选择题；单词（可选）；七下单元（可选）。"""
    stat = Counter()
    if not old:
        return units, dict(stat)
    old_units = old.get("units") or []
    old_by_id = {str(u.get("id")): u for u in old_units}

    for u in units:
        ou = old_by_id.get(u["id"])
        if not ou:
            continue
        known = {_norm(i.get("text")) for i in u["items"]}
        # sentence 只从切片重建（它最依赖教材原文的准确性）；
        # 其余题型在切片里抽不全（Grammar 版块只有讲解、词表页只有 9 页），从旧库补充。
        want = {"cloze", "choice", "semi_open", "oral_choice"}
        if keep_para:
            want.add("paragraph")
        if keep_words:
            want.add("word")
        for it in ou.get("items") or []:
            if str(it.get("kind")) not in want:
                continue
            key = _norm(it.get("text"))
            if key in known:
                continue
            u["items"].append(dict(it))          # 原样保留 answer / options / verified
            known.add(key)
            stat["迁移" + str(it.get("kind"))] += 1

    if keep_down:
        for ou in old_units:
            uid = str(ou.get("id") or "")
            if uid.startswith("b"):              # 七下单元 id 以 b 开头
                units.append(ou)
                stat["迁移七下单元"] += 1
    return units, dict(stat)


def report(units: list[dict], title: str) -> None:
    kinds: Counter = Counter()
    for u in units:
        for it in u.get("items", []):
            kinds[str(it.get("kind"))] += 1
    print(f"\n--- {title} ---")
    print(f"单元 {len(units)} 个 / 题目 {sum(kinds.values())} 道")
    for k, n in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"   {k:<12} {n}")
    print("\n各单元题量：")
    for u in units:
        print(f"   {u['id']:<5} {str(u.get('name'))[:34]:<36} {len(u.get('items', []))}")


def sample(units: list[dict], per: int = 6) -> None:
    for kind in ("word", "sentence", "paragraph", "semi_open", "cloze", "choice"):
        print("=" * 16, kind)
        n = 0
        for u in units:
            for it in u["items"]:
                if it.get("kind") != kind:
                    continue
                extra = " | zh=" + str(it.get("zh"))[:22] if it.get("zh") else ""
                if it.get("sample"):
                    extra += " | sample=" + str(it.get("sample"))[:30]
                if it.get("answer"):
                    extra += " | ans=" + str(it.get("answer"))
                print("  ", u["id"], "|", str(it.get("text"))[:70] + extra)
                n += 1
                if n >= per:
                    break
            if n >= per:
                break
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description="用教材切片重建 data/seed_grade7.json")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    ap.add_argument("--sample", action="store_true", help="打印各题型抽样")
    ap.add_argument("--with-old-words", action="store_true",
                    help="切片词表页只有 9 页（覆盖不全），加这个参数用旧库单词补齐")
    ap.add_argument("--no-para-old", action="store_true", help="不叠加旧库段落题")
    ap.add_argument("--no-down", action="store_true", help="不迁移七下单元")
    ap.add_argument("--no-backup", action="store_true", help="覆盖前不备份旧库")
    args = ap.parse_args()

    chunks = load_chunks()
    if not chunks:
        return 1
    print(f"读取切片 {len(chunks)} 条：{CHUNKS.name}")

    units, stat = build_up_units(chunks)
    print("切片归类：", stat)

    old = None
    if OUT.exists():
        try:
            old = json.loads(OUT.read_text(encoding="utf-8"))
            print(f"读取旧库：{len(old.get('units') or [])} 单元")
        except json.JSONDecodeError as exc:
            print(f"  ! 旧库解析失败（{exc}），跳过迁移", file=sys.stderr)

    units, mstat = migrate(
        units, old,
        keep_words=args.with_old_words,
        keep_para=not args.no_para_old,
        keep_down=not args.no_down,
    )
    if mstat:
        print("迁移：", mstat)

    report(units, "重建结果")
    if args.sample:
        sample(units)

    if args.dry_run:
        print("[dry-run] 未写入任何文件")
        return 0

    if OUT.exists() and not args.no_backup:
        bak = OUT.with_name(OUT.name + ".bak")
        shutil.copy2(OUT, bak)
        print(f"\n已备份旧库 -> {bak.name}")

    corpus = {
        "meta": {
            "grade": 7,
            "title": "人教版 Go for it! 七年级教材语料",
            "sources": ["data/rag/chunks.jsonl（大模型解析的教材切片）",
                        "旧库迁移：语法题 / 段落题 / 七下单元"],
        },
        "units": units,
    }
    OUT.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写入 {OUT}：{len(units)} 单元 / {sum(len(u['items']) for u in units)} 题")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
