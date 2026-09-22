#!/usr/bin/env python3
"""把两本七年级教材 PDF 切片入库，供 RAG 检索。
   # 目前实际是用chatgpt解析pdf文档所得

两本书的秉性不同：
  七上：PDF 自带文本层 → pdftotext 直接取文本

产出：
  data/rag/chunks.jsonl   切片（每页按窗口切，带 book/page/unit 元数据）
  data/rag/meta.json      统计信息
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAG = DATA / "rag"
PARTS = DATA / "parts"

UP_PDF = DATA / "【人教版】七年级上册英语电子课本-社学整理.pdf"

CHUNK = 600
OVERLAP = 120
UNIT_RE = re.compile(r"(Starter Unit [123]|Unit [1-9]|Unit 1[012])")


def slice_text(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    out = []
    i = 0
    while i < len(text):
        out.append(text[i : i + CHUNK])
        i += max(CHUNK - OVERLAP, 1)
    return out


def add(out: list, book: str, page: int, unit: str, text: str) -> None:
    for n, chunk in enumerate(slice_text(text), 1):
        m = UNIT_RE.search(chunk)
        out.append(
            {
                "id": f"{book}-p{page:03d}-{n}",
                "book": book,
                "page": page,
                "unit": m.group(1) if m else unit,
                "text": chunk,
            }
        )


def from_text_layer(pdf: Path, book: str) -> list[dict]:
    """pdftotext 按页取文本（自带文本层的 PDF）。"""
    raw = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"], capture_output=True, check=True
    ).stdout.decode("utf-8", "ignore")
    pages = raw.split("\f")
    out: list[dict] = []
    unit = ""
    for i, page in enumerate(pages, 1):
        if not page.strip():
            continue
        m = UNIT_RE.search(page)
        if m:
            unit = m.group(1)
        add(out, book, i, unit, page)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="七年级教材 PDF 切片入库（RAG）")
    ap.add_argument("--no-up", action="store_true", help="跳过上册")
    ap.add_argument("--no-parts", action="store_true", help="跳过已抽取词表")
    args = ap.parse_args()

    RAG.mkdir(parents=True, exist_ok=True)
    chunks: list[dict] = []
    if not args.no_up and UP_PDF.exists():
        chunks += from_text_layer(UP_PDF, "七上")
        print(f"七上文本层：{len(chunks)} 条")
    out_file = RAG / "chunks.jsonl"
    with out_file.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    (RAG / "meta.json").write_text(
        json.dumps({"count": len(chunks), "books": sorted({c["book"] for c in chunks}),
                    "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"写入 {out_file}：{len(chunks)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
