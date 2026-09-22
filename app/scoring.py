"""把驰声返回的评测结果（不同工具字段不统一）归一成统一结构。"""

from __future__ import annotations

from typing import Any

from .phonemes import label_for, tip_for

OVERALL_KEYS = ("overall", "overall_score", "total_score", "totalScore", "score", "total")
DIM_KEYS = {
    "accuracy": "准确度",
    "fluency": "流利度",
    "integrity": "完整度",
    "pronunciation": "发音",
    "integrality": "完整度",
    "completeness": "完整度",
    "rhythm": "节奏感",
}
GOOD_DP = {"", "correct", "right", "ok", "okay", "good", "normal", "true", "0"}
BAD_DP = {
    "mispron", "mispronunciation", "missing", "miss", "addition", "insert",
    "substitution", "wrong", "error", "err", "bad", "false",
}


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


WORD_KEYS = {"word", "word_text", "content", "text", "char"}
PHONE_LIST_KEYS = {"phone", "phones"}


def _walk(node: Any, out: dict[str, list[Any]], parent: str = "") -> None:
    """递归收集 词级 / 音素级 / 自然拼读级 记录（兼容驰声各 core_type 的字段差异）。"""
    if isinstance(node, dict):
        keys = {str(k).lower() for k in node.keys()}
        if parent in ("stress", "phoneme"):
            return  # 重音/音标辅助信息，不重复计入
        if parent in PHONE_LIST_KEYS:
            out["phones"].append(node)
        elif parent == "phonics":
            out["phonics"].append(node)
        elif keys & PHONE_LIST_KEYS:
            out["words"].append(node)
        elif keys & {"phoneme", "ipa"} and "score" not in keys:
            out["phones"].append(node)
        elif keys & WORD_KEYS and _num(node.get("score")) is not None:
            out["words"].append(node)
        for k, v in node.items():
            _walk(v, out, str(k).lower())
    elif isinstance(node, list):
        for value in node:
            _walk(value, out, parent)


def _collect(payload: Any) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {"phones": [], "words": [], "phonics": []}
    _walk(payload, out)
    return out


def _unwrap(payload: Any) -> Any:
    if isinstance(payload, dict):
        for key in ("result", "data", "Result", "Data"):
            inner = payload.get(key)
            if isinstance(inner, (dict, list)):
                return inner
    return payload


def extract_overall(payload: Any) -> float | None:
    node = payload
    for _ in range(3):
        if not isinstance(node, dict):
            break
        for key in OVERALL_KEYS:
            value = _num(node.get(key))
            if value is not None:
                return value
        node = _unwrap(node)
    return None


def extract_dims(payload: Any) -> dict[str, float]:
    dims: dict[str, float] = {}
    node = payload
    for _ in range(3):
        if not isinstance(node, dict):
            break
        for key, label in DIM_KEYS.items():
            raw_value = node.get(key)
            value = _num(raw_value)
            if value is None and isinstance(raw_value, dict):
                # 驰声的 fluency 是对象：{overall, pause, speed}
                value = _num(raw_value.get("overall") if raw_value.get("overall") is not None else raw_value.get("score"))
                for sub_key, sub_label in (("pause", "停顿"), ("speed", "语速")):
                    sub = _num(raw_value.get(sub_key))
                    if sub is not None:
                        dims.setdefault(sub_label, sub)
            if value is not None and label not in dims:
                dims[label] = value
        node = node.get("result") if isinstance(node, dict) else None
    return dims


# ---- 纠音对齐表（仅 en_word_correction / en_sentence_correction 返回）----
# 结构：details[].phone[] = {"lab": 期望音素, "rec": 识别音素, "score": n}
# lab/rec 里的 "#" 表示空：lab 有值 + rec 空 → 漏读；lab 空 + rec 有值 → 多读；
# 两者都有值但不相等 → 错读；相等但分低 → 发音不到位。
KIND_LABELS = {
    "mispron": "错读",
    "missing": "漏读",
    "addition": "多读",
    "weak": "发音不到位",
    "correct": "读对",
}
_STRESS_MARKS = str.maketrans({c: "" for c in "ˈˌ.‿'’|"})
_EMPTY_MARKS = {"#", "-", "—", "–", "none", "null"}


def norm_phone(value: Any) -> str:
    """归一化音素：去掉重音/音节点并统一小写，避免 ˈæ 与 æ 被误判成错读。"""
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text in _EMPTY_MARKS:
        return ""
    return text.translate(_STRESS_MARKS).strip()


def phone_kind(lab: str, rec: str, score: float | None) -> str:
    """比对 lab / rec，判定读对、错读、漏读、多读、发音不到位。"""
    if not lab and not rec:
        return "skip"
    if not lab:
        return "addition"
    if not rec:
        return "missing"
    if lab == rec:
        return "weak" if (score is not None and score < 60) else "correct"
    return "mispron"


def _collect_alignment(node: Any, out: list[dict[str, Any]], word: str = "") -> None:
    """按顺序收集 {word, lab, rec, score} 对齐记录（word 用最近的词级字段）。"""
    if isinstance(node, dict):
        keys = {str(k).lower() for k in node.keys()}
        if "lab" in keys or "rec" in keys:
            out.append(
                {
                    "word": word,
                    "lab": norm_phone(node.get("lab")),
                    "rec": norm_phone(node.get("rec")),
                    "score": _num(node.get("score")),
                }
            )
            return
        for key in ("word", "word_text", "content", "text"):
            raw = node.get(key)
            if isinstance(raw, str) and raw.strip():
                word = raw.strip()
                break
        for k, v in node.items():
            if str(k).lower() in ("stress", "phoneme"):
                continue
            _collect_alignment(v, out, word)
    elif isinstance(node, list):
        for value in node:
            _collect_alignment(value, out, word)


def extract_phones_detail(payload: Any) -> list[dict[str, Any]]:
    """逐词音素对齐明细（其它工具无 lab/rec，自然返回空列表）。"""
    records: list[dict[str, Any]] = []
    _collect_alignment(payload, records)
    groups: list[dict[str, Any]] = []
    for row in records:
        kind = phone_kind(row["lab"], row["rec"], row["score"])
        if kind == "skip":
            continue
        if not groups or groups[-1]["word"] != row["word"]:
            groups.append({"word": row["word"], "phones": [], "issues": {}})
        group = groups[-1]
        group["phones"].append(
            {"lab": row["lab"], "rec": row["rec"], "kind": kind, "score": row["score"]}
        )
        if kind != "correct":
            group["issues"][kind] = group["issues"].get(kind, 0) + 1
    for group in groups:
        group["has_issue"] = bool(group["issues"])
    return [g for g in groups if g["phones"]]


def extract_errors(payload: Any) -> list[dict[str, Any]]:
    out = _collect(payload)
    errors: list[dict[str, Any]] = []

    # 教材/驰声的自然拼读级：phoneme 是 IPA（如 eɪ），overall 是得分，spell 是拼写
    for ph in out["phonics"]:
        score = _num(ph.get("overall"))
        ipas = ph.get("phoneme")
        if score is None or score >= 60:
            continue
        if isinstance(ipas, str):
            ipas = [ipas]
        for ipa in ipas or []:
            errors.append(
                {
                    "phoneme": str(ipa),
                    "expected": str(ipa),
                    "actual": "",
                    "score": score,
                    "dp_type": "low_score",
                    "spell": str(ph.get("spell") or ""),
                }
            )

    for phone in out["phones"]:
        # 驰声 phone 数组只给 char+score，真正的 IPA 已在 phonics 里处理过
        if phone.get("char") and not phone.get("phoneme"):
            continue
        score = _num(phone.get("score"))
        dp = str(phone.get("dp_type") or phone.get("dpType") or "").strip().lower()
        is_bad = dp in BAD_DP or (dp not in GOOD_DP and score is not None and score < 60)
        if not is_bad:
            continue
        perr = phone.get("phoneme_error") or phone.get("phonemeError") or {}
        errors.append(
            {
                "phoneme": str(phone.get("phoneme") or phone.get("phone") or perr.get("expected") or ""),
                "expected": str(perr.get("expected") or phone.get("phoneme") or phone.get("phone") or ""),
                "actual": str(perr.get("actual") or perr.get("recognized") or ""),
                "score": score,
                "dp_type": dp or ("low_score" if score is not None else "unknown"),
            }
        )
    # 纠音对齐表：把错读/漏读/多读也归一成 errors，供报告与推荐使用
    for detail in extract_phones_detail(payload):
        for phone in detail["phones"]:
            if phone["kind"] == "correct":
                continue
            errors.append(
                {
                    "phoneme": phone["lab"],
                    "expected": phone["lab"],
                    "actual": phone["rec"],
                    "score": phone["score"],
                    "dp_type": phone["kind"],
                    "kind": phone["kind"],
                    "word": detail["word"],
                }
            )
    return errors


def extract_words(payload: Any) -> list[dict[str, Any]]:
    out = _collect(payload)
    words: list[dict[str, Any]] = []
    for item in out["words"]:
        score = _num(item.get("score"))
        text = str(
            item.get("word") or item.get("word_text") or item.get("content")
            or item.get("text") or item.get("char") or ""
        )
        if not text or score is None:
            continue
        words.append({"text": text, "score": score})
    # 去重，保留最低分
    merged: dict[str, float] = {}
    for w in words:
        if w["text"] not in merged or w["score"] < merged[w["text"]]:
            merged[w["text"]] = w["score"]
    return [{"text": k, "score": v} for k, v in merged.items()]


MULTI_DIM_MAX = {"cnt": 10.0, "flu": 3.0, "grammar": 3.0, "pron": 4.0}


def extract_multi_dim(payload: Any) -> dict[str, float]:
    """半开放题（en.scne.exam）的分项分，按各自满分归一化到 0~100。

    实测满分：cnt=10、flu=3、grammar=3、pron=4（各量程不同，必须先归一）。
    """
    out: dict[str, float] = {}
    node = payload
    for _ in range(4):
        if not isinstance(node, dict):
            break
        md = node.get("multi_dim") or node.get("multiDim")
        if isinstance(md, dict):
            for key, full in MULTI_DIM_MAX.items():
                val = _num(md.get(key))
                if val is not None:
                    out[key] = round(val / full * 100, 1)
            break
        nxt = node.get("details") if isinstance(node.get("details"), dict) else node.get("result")
        if not isinstance(nxt, dict):
            break
        node = nxt
    return out


def summarize(payload: Any) -> dict[str, Any]:
    """单题归一化结果。"""
    overall = extract_overall(payload)
    multi = extract_multi_dim(payload)
    # 内容未命中（cnt=0）→ 该题的发音/流利/完整度不可信，聚合时不计入
    content_valid = None if "cnt" not in multi else multi["cnt"] > 0
    return {
        "overall": overall,
        "dims": extract_dims(payload),
        "multi_dim": multi,
        "content_valid": content_valid,
        "words": extract_words(payload),
        "errors": extract_errors(payload),
        "phones_detail": extract_phones_detail(payload),
        "raw": payload,
    }


def level_label(overall: float | None) -> tuple[str, str]:
    if overall is None:
        return "未知", "这次没拿到分数，换个安静的地方再录一次吧。"
    if overall >= 88:
        return "优秀 A2+", "发音到位、语流连贯，可以挑战更难的话题表达。"
    if overall >= 75:
        return "良好 A2", "整体清楚流利，再抠几个难音和重音就更棒了。"
    if overall >= 60:
        return "合格 A1+", "能完整表达，问题集中在个别音、重音和流利度。"
    if overall >= 45:
        return "基础 A1", "还需要多跟读，先把单词读准再练句子。"
    return "起步 Pre-A1", "先一个音一个音模仿，每天 10 分钟跟读课文。"


DIM_LABELS = {
    "pron": "发音准确",
    "flu": "流利度",
    "integrity": "完整度",
    "content": "内容命中",
    "grammar": "语法",
}


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总成 4 维体检表：发音准确 / 流利度 / 完整度 / 内容命中。

    - 半开放题的分项分按各自满分归一后再统计（cnt/10、flu/3、pron/4）；
    - 内容未命中（cnt=0）的题不进发音/流利/完整统计，避免误诊；
    - 每个维度缺失就不计入（不是 0 分）。
    """
    # overall = 口语总分，语法题不计入（两者量纲不同：语法是 0/60/100 三档）
    spoken = [
        float(r["summary"]["overall"])
        for r in results
        if not r["summary"].get("rule") and r["summary"].get("overall") is not None
    ]
    overall = round(sum(spoken) / len(spoken), 1) if spoken else None
    grammar_scores = [
        float(r["summary"]["overall"])
        for r in results
        if r["summary"].get("rule") and r["summary"].get("overall") is not None
    ]

    buckets: dict[str, list[float]] = {k: [] for k in DIM_LABELS}
    extras: dict[str, list[float]] = {"pause": [], "speed": []}
    invalid = 0
    for r in results:
        s = r["summary"]
        if s.get("rule"):                       # 语法题只进语法统计
            if s.get("overall") is not None:
                buckets["grammar"].append(float(s["overall"]))
            continue
        md = s.get("multi_dim") or {}
        d = s.get("dims") or {}
        if "cnt" in md:
            buckets["content"].append(md["cnt"])
        if s.get("content_valid") is False:
            invalid += 1          # 答非所问：该题的发音/流利/完整不可信
            continue
        pron = md.get("pron") if md.get("pron") is not None else (d.get("发音") or d.get("准确度"))
        if pron is not None:
            buckets["pron"].append(float(pron))
        flu = md.get("flu") if md.get("flu") is not None else d.get("流利度")
        if flu is not None:
            buckets["flu"].append(float(flu))
        if isinstance(d.get("停顿"), (int, float)):
            extras["pause"].append(float(d["停顿"]))
        if isinstance(d.get("语速"), (int, float)):
            extras["speed"].append(float(d["语速"]))
        if isinstance(d.get("完整度"), (int, float)):
            buckets["integrity"].append(float(d["完整度"]))

    dim_avg = {
        DIM_LABELS[k]: (round(sum(v) / len(v), 1) if v else None) for k, v in buckets.items()
    }
    valid_dims = {k: v for k, v in dim_avg.items() if v is not None}
    weakest = min(valid_dims, key=lambda k: valid_dims[k]) if valid_dims else None

    phoneme_hits: dict[str, dict[str, Any]] = {}

    def hit_of(ph: str) -> dict[str, Any]:
        return phoneme_hits.setdefault(
            ph,
            {
                "phoneme": ph, "count": 0, "worst": 100.0, "words": set(),
                "kinds": {}, "misread_as": {},
            },
        )

    weak_words: list[dict[str, Any]] = []
    profile = {label: 0 for label in ("错读", "漏读", "多读", "发音不到位")}
    confusions: dict[tuple[str, str], int] = {}
    word_issues: dict[str, dict[str, int]] = {}

    for r in results:
        summary = r["summary"]
        for err in summary.get("errors") or []:
            if err.get("kind"):
                continue  # 来自对齐表，统一由下面的 phones_detail 统计，避免重复计数
            ph = err.get("expected") or err.get("phoneme")
            if not ph:
                continue
            bucket = hit_of(ph)
            bucket["count"] += 1
            score = err.get("score")
            if isinstance(score, (int, float)):
                bucket["worst"] = min(bucket["worst"], float(score))
            for w in summary.get("words") or []:
                if float(w["score"]) < 70:
                    bucket["words"].add(w["text"])
        for w in summary.get("words") or []:
            if float(w["score"]) < 70:
                weak_words.append({"text": w["text"], "score": float(w["score"])})
        # 纠音对齐表：错读 / 漏读 / 多读 / 发音不到位
        for detail in summary.get("phones_detail") or []:
            word = str(detail.get("word") or "").strip()
            for phone in detail.get("phones") or []:
                kind = str(phone.get("kind") or "")
                label = KIND_LABELS.get(kind)
                if label in profile:
                    profile[label] += 1
                if kind == "mispron" and phone.get("lab") and phone.get("rec"):
                    key = (str(phone["lab"]), str(phone["rec"]))
                    confusions[key] = confusions.get(key, 0) + 1
                if kind not in ("mispron", "missing", "weak"):
                    continue
                ph = str(phone.get("lab") or "")
                if not ph:
                    continue
                bucket = hit_of(ph)
                bucket["count"] += 1
                if isinstance(phone.get("score"), (int, float)):
                    bucket["worst"] = min(bucket["worst"], float(phone["score"]))
                bucket["kinds"][kind] = bucket["kinds"].get(kind, 0) + 1
                if kind == "mispron" and phone.get("rec"):
                    mis = bucket["misread_as"]
                    mis[str(phone["rec"])] = mis.get(str(phone["rec"]), 0) + 1
                if word:
                    bucket["words"].add(word)
            if word and detail.get("issues"):
                acc = word_issues.setdefault(word.lower(), {})
                for kind, num in detail["issues"].items():
                    acc[kind] = acc.get(kind, 0) + int(num)

    weak_phonemes = sorted(
        phoneme_hits.values(), key=lambda x: (-x["count"], x["worst"])
    )
    for item in weak_phonemes:
        item["words"] = sorted(item["words"])[:5]
        item["label"] = label_for(item["phoneme"])
        item["tip"] = tip_for(item["phoneme"])
        kinds = item.get("kinds") or {}
        item["kinds"] = kinds
        item["kinds_text"] = "、".join(
            f"{KIND_LABELS.get(k, k)}×{v}" for k, v in sorted(kinds.items(), key=lambda x: -x[1])
        )
        misread = sorted((item.get("misread_as") or {}).items(), key=lambda x: -x[1])[:2]
        item["misread_as"] = [{"phoneme": k, "count": v} for k, v in misread]
        item["misread_text"] = "、".join(f"/{k}/" for k, _ in misread)

    # 去重弱词（带上该词具体错在哪几类）
    seen: dict[str, dict[str, Any]] = {}
    for w in weak_words:
        key = w["text"].lower()
        cur = seen.get(key)
        if cur is None or w["score"] < cur["score"]:
            seen[key] = {"text": w["text"], "score": w["score"], "issues": word_issues.get(key, {})}
    weak_words = sorted(seen.values(), key=lambda x: x["score"])[:10]

    # ---- 语法画像：按知识点统计正确率 ----
    # results 是「定级 + 每轮练习」的时间序累积，最后一条即最近一次作答，
    # 所以 last_correct 天然是「这个点现在还会不会错」。
    gp: dict[str, dict[str, Any]] = {}
    for r in results:
        rule = r["summary"].get("rule") or {}
        if not rule:
            continue
        point = str(rule.get("grammar_point") or "未标注知识点")
        bucket = gp.setdefault(
            point,
            {"point": point, "total": 0, "correct": 0, "wrong_as": [], "last_correct": None},
        )
        bucket["total"] += 1
        if rule.get("correct"):
            bucket["correct"] += 1
        else:
            expected = str(rule.get("expected") or "").strip()
            actual = str(rule.get("actual") or "").strip()
            bucket["wrong_as"].append(f"{expected} → {actual or '空'}")
        bucket["last_correct"] = bool(rule.get("correct"))   # 时间序：后写覆盖前写
    grammar_profile = {
        point: {
            "point": point,
            "total": b["total"],
            "correct": b["correct"],
            "wrong": b["total"] - b["correct"],
            "rate": round(b["correct"] / b["total"] * 100, 1),
            "wrong_as": list(dict.fromkeys(b["wrong_as"]))[:3],
            "last_correct": b["last_correct"],
            "mastered": bool(b["last_correct"]),
        }
        for point, b in gp.items()
    }
    # 薄弱知识点只看「最近一次仍做错」。
    # 聚合是累计的，若只判 wrong>0，学生答对后这个点会永久留在薄弱清单里，
    # 结果就是「本轮完成，再来一组」反复推同一个知识点。
    weak_grammar = sorted(
        [g for g in grammar_profile.values() if g["wrong"] and not g["last_correct"]],
        key=lambda x: (x["rate"], -x["wrong"]),
    )
    # 已过关的知识点（最近一次答对）：推荐时不再重复推
    mastered_grammar = sorted(
        g["point"] for g in grammar_profile.values() if g["total"] and g["last_correct"]
    )

    confusions_top = sorted(confusions.items(), key=lambda x: -x[1])[:5]

    label, desc = level_label(overall)
    return {
        "overall": overall,
        "level": label,
        "level_desc": desc,
        "dims": dim_avg,
        "weakest_dim": weakest,
        "extras": {k: (round(sum(v) / len(v), 1) if v else None) for k, v in extras.items()},
        "invalid_count": invalid,
        "weak_phonemes": weak_phonemes[:6],
        "weak_words": weak_words,
        "error_profile": {k: v for k, v in profile.items() if v},
        "error_total": sum(profile.values()),
        "sound_confusions": [
            {"expected": a, "actual": b, "count": n, "label": f"/{a}/ → /{b}/"}
            for (a, b), n in confusions_top
        ],
        "grammar_score": round(sum(grammar_scores) / len(grammar_scores), 1) if grammar_scores else None,
        "grammar_profile": grammar_profile,
        "weak_grammar": weak_grammar,
        "mastered_grammar": mastered_grammar,
        "spoken_count": len(spoken),
        "answered": len(results),
    }
