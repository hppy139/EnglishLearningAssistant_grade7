#!/usr/bin/env python3
"""用大模型复核语法题（填空 / 选择）是否准确。

背景：早期由大模型直接生成的题出现过「答案与句子语法不符」的情况
（例：Li Jingjing music ___ fun. 标答 it's，实际应为 is）。
因此规则规定：
  - 教材原句挖空的题 → 填回答案必须还原成原句（build_corpus 里已强校验，verified=True）
  - 大模型生成的题 → 默认 verified=False，必须经本脚本复核通过才会进入定级卷与推荐

复核内容（逐题）：
  1. 把 answer 填回题干（或选中该选项）后，句子是否语法正确、意思通顺；
  2. 在 options 里 answer 是否唯一正确；
  3. 若标答有误，给出正确答案 fix（在选项内则自动修正，否则剔除该题）。

用法：
  python3 scripts/verify_items.py            # 复核并写回 data/parts
  python3 scripts/verify_items.py --dry-run   # 只打印判定，不落盘
  python3 scripts/verify_items.py --batch 10  # 每批发给大模型几道
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.grammar import RULE_KINDS, normalize  # noqa: E402

PARTS = ROOT / "data" / "parts"
CORPUS = ROOT / "data" / "seed_grade7.json"

SYSTEM = (
    "你是人教版初中英语教研员，要逐道复核语法题是否正确。\n"
    "对每道题判断：\n"
    "1. 把 answer 填回题干的空格（或选中该选项）后，句子是否语法正确、意思通顺；\n"
    "2. 在给定的 options 里（若有），answer 是否唯一正确，其它选项是否确实错误；\n"
    "3. 若 answer 不正确，给出正确答案 fix（必须是 options 中的一个；没有 options 就给正确单词）。\n"
    "只输出 JSON：{\"results\":[{\"id\":\"题号\",\"ok\":true,\"reason\":\"一句话理由\","
    "\"fix\":null}]}\n"
    "注意：ok=false 且没给可用 fix 的题会被剔除。"
)

USER = "【待复核题目】\n{items}\n\n请逐题复核。"


def load_targets(only_pending: bool = False) -> list[dict[str, Any]]:
    """取待复核的语法题。默认复核全部（含已通过规则校验的原句挖空题）。

    原因：PDF 会把教材表格压成一行（如 "Li Jingjing music It's fun."），
    「填回答案 == 原句」的规则校验拦不住这种本身就不通顺的素材。
    """
    data = json.loads(CORPUS.read_text(encoding="utf-8"))
    out = []
    for unit in data.get("units", []):
        for item in unit.get("items", []):
            if str(item.get("kind")) not in RULE_KINDS:
                continue
            if only_pending and item.get("verified") is not False:
                continue
            item["unit_id"] = unit.get("id")
            out.append(item)
    return out


def apply_fix(item: dict[str, Any], fix: Any) -> bool:
    """按大模型给的正确答案修正；无法修正返回 False（该题剔除）。"""
    if not fix:
        return False
    opts = [str(o) for o in (item.get("options") or [])]
    fix_n = normalize(fix)
    if opts:
        hit = next((o for o in opts if normalize(o) == fix_n), None)
        if hit is None:
            return False
        item["answer"] = hit
    else:
        item["answer"] = str(fix).strip()
    return True


def save_back(fixed: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> None:
    """把复核结果写回 data/parts（下一轮 build_corpus 会带进去）。"""
    by_unit: dict[str, list[dict[str, Any]]] = {}
    for it in fixed + rejected:
        uid = str(it.get("unit_id") or str(it.get("id", ""))[:4])
        by_unit.setdefault(uid, []).append(it)
    for uid, items in by_unit.items():
        for name in (f"grammar_{uid}.json", f"cloze_text_{uid}.json"):
            path = PARTS / name
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            by_id = {str(x.get("id")): x for x in data}
            for it in items:
                target = by_id.get(str(it.get("id")))
                if target is None:
                    continue
                target["verified"] = bool(it.get("verified"))
                for key in ("answer", "options", "grammar_point", "verify_reason"):
                    if key in it:
                        target[key] = it[key]
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    if rejected:
        out = PARTS / "rejected_grammar.json"
        out.write_text(
            json.dumps(
                [
                    {
                        "id": it.get("id"),
                        "text": it.get("text"),
                        "answer": it.get("answer"),
                        "reason": it.get("verify_reason"),
                        "source_sentence": it.get("source_sentence"),
                    }
                    for it in rejected
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"被剔除的题已记录到 {out.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description="用大模型复核语法题")
    ap.add_argument("--batch", type=int, default=10, help="每批发给大模型几道")
    ap.add_argument("--only-pending", action="store_true", help="只复核尚未校验的题")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pending = load_targets(only_pending=args.only_pending)
    if not pending:
        print("没有待复核的语法题（全部已校验）")
        return 0
    print(f"待复核 {len(pending)} 道语法题")

    from app.llm import LLM, last_error  # noqa: PLC0415

    llm = LLM()
    if not llm.available:
        print("没有配置 LLM_API_KEY", file=sys.stderr)
        return 1

    fixed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    def _ask(chunk: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        payload = [
            {
                "id": it.get("id"),
                "kind": it.get("kind"),
                "text": it.get("text"),
                "answer": it.get("answer"),
                "options": it.get("options"),
                "grammar_point": it.get("grammar_point"),
            }
            for it in chunk
        ]
        # 批量复核输出长，必须给足 max_tokens，否则 JSON 被截断导致整批失败
        data = llm.chat_json(
            SYSTEM, USER.format(items=json.dumps(payload, ensure_ascii=False))
        )
        if not data or not isinstance(data.get("results"), list):
            return {}
        return {str(r.get("id")): r for r in data["results"] if isinstance(r, dict)}

    def _handle(chunk: list[dict[str, Any]], verdicts: dict[str, dict[str, Any]]) -> None:
        for it in chunk:
            verdict = verdicts.get(str(it.get("id")))
            if not verdict:
                continue
            ok = bool(verdict.get("ok"))
            reason = str(verdict.get("reason") or "")
            if ok:
                it["verified"] = True
                it["verify_reason"] = reason
                fixed.append(it)
                print(f"  ✅ {str(it.get('text'))[:44]} | {reason[:40]}")
                continue
            if apply_fix(it, verdict.get("fix")):
                it["verified"] = True
                it["verify_reason"] = f"已按复核修正为 {it.get('answer')}；{reason}"
                fixed.append(it)
                print(f"  🔧 {str(it.get('text'))[:44]} | 修正答案 → {it.get('answer')}")
            else:
                it["verified"] = False
                it["verify_reason"] = reason or "复核判定不可用"
                rejected.append(it)
                print(f"  ❌ {str(it.get('text'))[:44]} | {it['verify_reason'][:40]}")

    for start in range(0, len(pending), args.batch):
        chunk = pending[start : start + args.batch]
        verdicts = _ask(chunk)
        if verdicts:
            _handle(chunk, verdicts)
            continue
        if len(chunk) > 1:
            # 整批失败（多半是输出被截断）→ 逐题重试，别把整批丢掉
            print(f"  批次 {start}-{start + len(chunk)} 失败，改为逐题重试")
            for one in chunk:
                _handle([one], _ask([one]))
            continue
        print(f"  {chunk[0].get('id')} 复核失败（{last_error() or '未知'}），跳过")

    print(f"\n复核结果：通过 {len(fixed)} 道，剔除 {len(rejected)} 道")
    if args.dry_run:
        print("（dry-run，未落盘）")
        return 0
    save_back(fixed, rejected)
    print("已写回 data/parts，请重新执行：python3 scripts/build_corpus.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
