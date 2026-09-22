#!/usr/bin/env python3
"""按知识点补语法题（离线批处理）。

与 scripts/gen_grammar_items.py 的分工：
  - 那个按「单元」出题，一批覆盖多个知识点；
  - 这个按「知识点」出题，专门补题量不足的点——
    否则学生的薄弱点恰好落在小题量知识点上时，「再来一组」会无题可出、只能跳点。

生成的题做三重校验（结构 / 知识点受控表 / 超纲词汇），落盘到 data/gen_cache/，
app/corpus.iter_items() 会自动并入，运行时即可被推荐、被判分。

用法：
  python3 scripts/topup_grammar.py --dry-run            # 只看要补哪些点，不调用大模型
  python3 scripts/topup_grammar.py --min 6 --need 4     # 低于 6 道的点各补 4 道
  python3 scripts/topup_grammar.py --points "第三人称单数 -s" --need 5
  python3 scripts/topup_grammar.py --need 3 --no-save   # 试生成但不落盘
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import grammar_topup  # noqa: E402
from app.llm import LLM  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="按知识点补齐语法题（大模型生成 + 教材约束校验）")
    ap.add_argument("--min", type=int, default=6, help="题量低于该值的知识点才算不足（默认 6）")
    ap.add_argument("--need", type=int, default=4, help="每个知识点补几道（默认 4）")
    ap.add_argument("--points", default="", help="只补这些知识点，逗号分隔；默认全部不足的")
    ap.add_argument("--limit", type=int, default=8, help="最多补几个知识点（按题量从少到多，默认 8）")
    ap.add_argument("--dry-run", action="store_true", help="只统计与打印计划，不调用大模型")
    ap.add_argument("--no-save", action="store_true", help="生成但不落盘（试效果）")
    args = ap.parse_args()

    counts = grammar_topup.point_counts()
    print(f"题库现有语法题 {sum(counts.values())} 道，按知识点：")
    for point, n in sorted(counts.items(), key=lambda x: (x[1], x[0])):
        flag = "  ← 不足" if n < args.min else ""
        print(f"   {point:<30} {n:>3} 道{flag}")

    if args.points:
        want = [p.strip() for p in args.points.split(",") if p.strip()]
        targets = [(p, counts.get(p, 0)) for p in want]
    else:
        targets = grammar_topup.short_points(min_count=args.min)[: args.limit]

    if not targets:
        print(f"\n没有低于 {args.min} 道的知识点，无需补题")
        return 0
    print(f"\n待补知识点：{', '.join(f'{p}({n})' for p, n in targets)}")

    if args.dry_run:
        print("dry-run：未调用大模型、未落盘")
        return 0

    llm = LLM()
    if not llm.available:
        print("没有配置 LLM_API_KEY，无法生成题目", file=sys.stderr)
        return 1

    total = 0
    for point, have in targets:
        ok, dropped, err = grammar_topup.generate_for_point(
            point, args.need, save=not args.no_save
        )
        if err:
            print(f"\n{point}（原有 {have} 道）-> 失败：{err}")
            continue
        print(
            f"\n{point}（原有 {have} 道）-> 新增 {len(ok)} 道"
            + ("（未落盘）" if args.no_save else f"，已写入 {grammar_topup.cache_file(point).name}")
        )
        for d in dropped:
            print(f"    剔除：{d}")
        for it in ok:
            opts = " / ".join(str(o) for o in (it.get("options") or []))
            print(
                f"    [{it['kind']}] {it['text']}  答案={it['answer']}"
                f"{'  选项=' + opts if opts else ''}  {it.get('hint', '')}"
            )
        total += len(ok)
    print(f"\n共补 {total} 道题；缓存目录：{grammar_topup.CACHE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
