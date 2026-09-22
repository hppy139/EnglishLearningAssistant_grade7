"""七年级教材语料库：加载、组卷（定级测评）、同类题检索。"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from . import DATA_DIR
from .phonemes import guess_phones, matches_phoneme

CORPUS_FILE = DATA_DIR / "grade7_corpus.json"
SEED_FILE = DATA_DIR / "seed_grade7.json"

_CACHE: dict[str, Any] | None = None

_TEXT_NOISE = re.compile(r"[^a-z0-9\u4e00-\u9fff]+")


def norm_text(text: Any) -> str:
    """归一化文本：小写、去标点与多余空格。用于判断「同一句话」，与题库去重。"""
    return _TEXT_NOISE.sub(" ", str(text or "").lower()).strip()


def load_corpus(force: bool = False) -> dict[str, Any]:
    """优先用爬虫产出的 grade1_corpus.json，回退到内置种子语料。"""
    global _CACHE
    if _CACHE is not None and not force:
        return _CACHE
    path = CORPUS_FILE if CORPUS_FILE.exists() else SEED_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = {"meta": {}, "units": []}
    _CACHE = data
    return data


def corpus_source() -> str:
    return "grade7_corpus.json(含爬取)" if CORPUS_FILE.exists() else "seed_grade7.json(教材抽取)"


def iter_items(data: dict[str, Any] | None = None) -> Iterable[dict[str, Any]]:
    data = data or load_corpus()
    items: list[dict[str, Any]] = []
    for unit in data.get("units", []):
        for item in unit.get("items", []):
            items.append(
                {
                    **item,
                    "unit_id": unit.get("id"),
                    "unit_name": unit.get("name"),
                    "topic": unit.get("topic"),
                }
            )
    # 并入大模型按知识点补的语法题（data/gen_cache/）：同 id / 同题干不重复
    from .grammar_topup import merge_generated

    return iter(merge_generated(items))


def all_items() -> list[dict[str, Any]]:
    return list(iter_items())


def find_item(item_id: str) -> dict[str, Any]:
    for item in iter_items():
        if item.get("id") == item_id:
            return item
    raise KeyError(f"未找到题目 {item_id}")


def item_phones(item: dict[str, Any]) -> list[str]:
    """题目涉及的音素：优先用语料里人工标注的 phones，否则按拼读近似推断。

    句子/段落必须逐词推断：整串拼接会丢掉词边界，this 会被当成 /θ/。
    """
    phones = item.get("phones")
    if isinstance(phones, list) and phones:
        return list(phones)
    text = str(item.get("text", ""))
    out: list[str] = []
    for word in re.findall(r"[A-Za-z']+", text):
        out.extend(guess_phones(word))
    return out


# 定级卷按三类组织（choice 题型不在工具白名单内，全部排除）
WORD_LIMIT = 3          # 单词栏最多推荐几题
GRAMMAR_LIMIT = 3       # 语法栏最多推荐几题
WORD_KINDS = ("word", "phonics")
SENTENCE_KINDS = ("sentence",)
ANSWER_KINDS = ("semi_open",)
PARAGRAPH_KINDS = ("paragraph",)

# 语料题型 → 工具链类型（见 app/tools.py）
# rule = 语法题（填空 / 选择），不调驰声，本地规则判分
CHAIN_KIND = {
    "word": "word",
    "phonics": "word",
    "sentence": "sentence",
    "semi_open": "sentence_answer",
    "paragraph": "paragraph",
    "oral_choice": "choice",
    "cloze": "rule",
    "choice": "rule",
}

# 规则判分的题型
RULE_KINDS = ("cloze", "choice")


def chain_kind_of(item: dict[str, Any]) -> str:
    return CHAIN_KIND.get(str(item.get("kind")), "word")


# 高价值易错音素（定级卷优先覆盖）
FOCUS_PHONEMES = {
    "θ", "ð", "r", "l", "v", "w", "ʃ", "tʃ", "dʒ", "ŋ",
    "æ", "iː", "ɪ", "ʊ", "uː", "eɪ", "aɪ", "ɔɪ", "əʊ",
}


def _seed_pool(kind: str) -> list[dict[str, Any]]:
    """教材题优先（抓取语料质量不齐，定级卷不用）；未通过校验的题不出。"""
    pool = [
        i
        for i in all_items()
        if str(i.get("kind")) == kind and i.get("verified") is not False
    ]
    return [i for i in pool if i.get("topic") != "crawled"] or pool


def _pick_by_length(
    pool: list[dict[str, Any]],
    count: int,
    exclude_texts: set[str] | None = None,
    weak_phonemes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """按词数从短到长挑（句子/段落的难度阶梯）；若有弱音素则含弱音素的题优先。"""
    blocked = {norm_text(t) for t in (exclude_texts or set())}
    pool = [x for x in pool if norm_text(x.get("text")) not in blocked]
    ordered = sorted(
        pool, key=lambda x: (len(str(x.get("text", "")).split()), int(x.get("level", 1)))
    )
    if weak_phonemes:
        weak = set(weak_phonemes)
        focus = [x for x in ordered if set(item_phones(x)) & weak]
        rest = [x for x in ordered if x not in focus]
        ordered = focus + rest
    return ordered[:count]


def _pick_spread(
    pool: list[dict[str, Any]], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    """按难度阶梯随机取题：把候选切成 count 段（短→长），每段随机取一个。

    _pick_by_length 是确定性取最短的前 N 个，会导致句子前两题、段落题每次刷新都一样。
    """
    if not pool or count <= 0:
        return []
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for x in pool:                      # 同一段话可能对应多个题号，先去重再抽
        key = norm_text(x.get("text"))
        if key and key not in seen:
            seen.add(key)
            uniq.append(x)
    ordered = sorted(
        uniq, key=lambda x: (len(str(x.get("text", "")).split()), int(x.get("level", 1)))
    )
    picked: list[dict[str, Any]] = []
    n = len(ordered)
    for i in range(count):
        lo = i * n // count
        hi = max(lo + 1, (i + 1) * n // count)
        seg = [x for x in ordered[lo:hi] if x not in picked]
        if not seg:
            seg = [x for x in ordered if x not in picked]
        if not seg:
            break
        picked.append(rng.choice(seg))
    return picked


def build_placement(
    size: int = 3,
    seed: int | None = None,
    skip_paragraph: bool = False,
) -> list[dict[str, Any]]:
    """三类定级卷：单词 size 题 + 句子（跟读 2 + 应答 1）+ 段落 1。"""
    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []

    # ① 单词：优先含"高价值易错音素"的词，并按难度阶梯从易到难取
    focus = [w for w in _seed_pool("word") if set(item_phones(w)) & FOCUS_PHONEMES]
    words = focus or _seed_pool("word")
    picked.extend(_pick_spread(words, size, rng))
    phonics = _seed_pool("phonics")
    if phonics:
        picked.append(rng.choice(phonics))

    # ② 句子：跟读 2 句（短→长，每次换题）+ 应答 1 题（semi_open 出四维）
    picked.extend(_pick_spread(_seed_pool("sentence"), 2, rng))
    answers = _seed_pool("semi_open")
    if answers:
        picked.append(rng.choice(answers))

    # ③ 段落 1 段（核心不可用时由调用方跳过）
    if not skip_paragraph:
        picked.extend(_pick_spread(_seed_pool("paragraph"), 1, rng))

    # ④ 语法题：填空 2 + 选择 1（规则判分，不调驰声）
    picked.extend(_pick_spread(_seed_pool("cloze"), 2, rng))
    picked.extend(_pick_spread(_seed_pool("choice"), 1, rng))

    for item in picked:
        chain = chain_kind_of(item)
        item["chain"] = chain
        if chain == "rule":
            item["group"] = "grammar"
        elif chain.startswith("word"):
            item["group"] = "word"
        elif chain.startswith("sentence"):
            item["group"] = "sentence"
        else:
            item["group"] = "paragraph"
    return picked


def find_similar(
    weak_phonemes: list[str],
    exclude_ids: set[str] | None = None,
    kinds: tuple[str, ...] = ("word", "phonics", "sentence"),
    limit: int = 30,
    exclude_texts: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """按弱音素在语料里找同类练习：命中越多、难度越低越靠前。"""
    exclude_ids = exclude_ids or set()
    blocked = {norm_text(t) for t in (exclude_texts or [])}
    scored: list[tuple[int, int, int, dict[str, Any]]] = []
    for item in iter_items():
        if item.get("id") in exclude_ids:
            continue
        if item.get("kind") not in kinds:
            continue
        text = str(item.get("text", ""))
        if norm_text(text) in blocked:      # 不能与定级测评考过的文本重复
            continue
        phones = item_phones(item)
        hit = 0
        for p in weak_phonemes:
            if p in phones or matches_phoneme(text, p):
                hit += 2
        if hit == 0:
            continue
        level = int(item.get("level", 1))
        crawled = 1 if item.get("topic") == "crawled" else 0
        scored.append((hit, level, crawled, item))
    # 命中多 → 难度低 → 内置教材优先
    scored.sort(key=lambda x: (-x[0], x[1], x[2], str(x[3].get("id"))))
    return [item for _, _, _, item in scored[:limit]]


# 易混音对立表（用于词表内配对，不生成新词）
CONTRAST: dict[str, list[str]] = {
    "θ": ["s", "f"], "ð": ["d", "z"], "ʃ": ["s"], "tʃ": ["ʃ", "tr"],
    "v": ["w", "f"], "r": ["l"], "l": ["r"], "w": ["v"],
    "iː": ["ɪ"], "ɪ": ["iː"], "æ": ["e"], "e": ["æ"], "uː": ["ʊ"], "ʊ": ["uː"],
    "ŋ": ["n"], "dʒ": ["tʃ"],
}


def find_pair(
    target: str,
    exclude_ids: set[str] | None = None,
    exclude_texts: Iterable[str] | None = None,
    actual: str | None = None,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    """词表内配对：一个含目标音的词 + 一个含其易混音、其余音相近的词。

    两个词都来自教材词表（不生成新词），用于对比纠音。
    actual 是测评里把这个音实际读成的音（来自纠音对齐表），优先用它做对比。
    """
    exclude_ids = exclude_ids or set()
    blocked = {norm_text(t) for t in (exclude_texts or [])}
    rng = rng or random.Random()
    words = [
        w
        for w in _seed_pool("word")
        if str(w.get("id")) not in exclude_ids and norm_text(w.get("text")) not in blocked
    ]
    partners = list(CONTRAST.get(target, []))
    if actual and actual not in partners:
        partners.insert(0, actual)
    # 实测读成的音必须对上：θ→s 就配含 s 的词，别再配 f 这种「可能混」的音
    primary = [actual] if actual else []
    pool_a = [w for w in words if target in item_phones(w)]
    if not pool_a or not partners:
        return []
    # 对比练习要短：低难度短词优先，再在头部随机，避免每次都推同一个词
    pool_a.sort(key=lambda w: (int(w.get("level", 1)), len(str(w.get("text", "")))))
    a = rng.choice(pool_a[:5]) if len(pool_a) > 1 else pool_a[0]

    def _rank(w: dict[str, Any]) -> tuple[int, int]:
        return (int(w.get("level", 1)), len(str(w.get("text", ""))))

    pa = set(item_phones(a))

    def _sim(pw: set[str]) -> tuple:
        """越接近最小对立对越好：音素数相同 > 长度接近 > 差异小 > Jaccard 高。

        只用 Jaccard 会偏爱长词（thin 会配到 cousin），对比辨音要的是「只差一个音」。
        """
        inter = len(pa & pw)
        return (
            1 if len(pa) == len(pw) else 0,
            -abs(len(pa) - len(pw)),
            -len(pa ^ pw),
            inter / (len(pa | pw) or 1),
        )

    best: dict[str, Any] | None = None
    best_key: tuple | None = None
    for w in words:
        if str(w.get("id")) == str(a.get("id")):
            continue
        pw = set(item_phones(w))
        if not pw or target in pw or not (pw & set(partners)):
            continue
        if primary and not (pw & set(primary)):
            continue
        key = _sim(pw)
        if best_key is None or key > best_key:
            best, best_key = w, key
    if best is None or best_key is None or best_key[3] < 0.25:
        # 音素结构差太远时，退化为「含实测错误音（或易混音）且最短」的词
        want = set(primary) or set(partners)
        cand = [
            w for w in words
            if (set(item_phones(w)) & want) and target not in item_phones(w)
        ]
        cand.sort(key=_rank)
        if cand:
            best = cand[0]
    return [a, best] if best else []


def fallback_pool(
    exclude_ids: set[str] | None = None,
    kinds: tuple[str, ...] = ("word", "sentence"),
    limit: int = 20,
) -> list[dict[str, Any]]:
    exclude_ids = exclude_ids or set()
    pool = [
        i
        for i in iter_items()
        if i.get("kind") in kinds and i.get("id") not in exclude_ids
    ]
    pool.sort(key=lambda x: (int(x.get("level", 1)), str(x.get("id"))))
    return pool[:limit]


def _word_reason(phoneme: str, kinds: dict[str, int] | None) -> str:
    """按具体问题类型写练习理由（错读 / 漏读 / 多读 / 不到位）。"""
    kinds = kinds or {}
    if not phoneme:
        return "同难度巩固练习"
    if kinds.get("mispron"):
        return f"纠音：/{phoneme}/ 上次读错了，慢速跟读标准音"
    if kinds.get("missing"):
        return f"补漏：/{phoneme}/ 上次没读出来，先把音读准"
    if kinds.get("weak"):
        return f"打磨：/{phoneme}/ 能认出来但不够到位"
    if kinds.get("addition"):
        return f"控音：/{phoneme}/ 附近多读了音，注意别加音"
    return f"巩固 /{phoneme}/"


def _weak_rank(hit: dict[str, Any]) -> int:
    """弱音素优先级：错读 > 漏读 > 不到位 > 其它。"""
    kinds = hit.get("kinds") or {}
    if kinds.get("mispron"):
        return 0
    if kinds.get("missing"):
        return 1
    if kinds.get("weak"):
        return 2
    return 3


def find_by_grammar(
    points: list[str],
    exclude_ids: set[str] | None = None,
    exclude_texts: Iterable[str] | None = None,
    exclude_points: set[str] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """按语法知识点找练习题。

    排序策略（避免"一直推已经答对的知识点"）：
      1. 先给「本次做错的知识点」的题（错得多的点优先）；
      2. 不够时用「本次没测过的知识点」补（已答对的点不再重复）；
      3. 都不够再按难度兜底。
    """
    exclude_ids = exclude_ids or set()
    blocked = {norm_text(t) for t in (exclude_texts or [])}
    skip_points = {str(p) for p in (exclude_points or set())}
    rank = {str(p): i for i, p in enumerate(points or [])}

    pool: list[tuple[str, int, str, dict[str, Any]]] = []
    for item in iter_items():
        if str(item.get("kind")) not in RULE_KINDS:
            continue
        if item.get("verified") is False:        # 未经大模型校验的题不出
            continue
        if str(item.get("id")) in exclude_ids:
            continue
        if norm_text(item.get("text")) in blocked:
            continue
        pool.append(
            (
                str(item.get("grammar_point") or ""),
                int(item.get("level", 1)),
                str(item.get("id")),
                item,
            )
        )

    hit = [x for x in pool if rank and x[0] in rank]
    hit.sort(key=lambda x: (rank[x[0]], x[1], x[2]))
    fresh = [x for x in pool if x[0] and x[0] not in skip_points and x not in hit]
    fresh.sort(key=lambda x: (x[1], x[2]))
    rest = [x for x in pool if x not in hit and x not in fresh]
    rest.sort(key=lambda x: (x[1], x[2]))
    return [x[3] for x in (hit + fresh + rest)[:limit]]


def _grammar_weight(g: dict[str, Any] | None) -> float:
    """知识点的复习权重。

    关键：答对过 ≠ 永久掌握，所以这里是「降频」而不是「归零」——
    否则学生会发现某个知识点一旦答对就再也不出现，巩固无从谈起。
    """
    if not g or not g.get("total"):
        return 3.0                     # 没测过：中等（用来补新知识点）
    if not g.get("last_correct"):
        return 10.0                    # 最近仍做错：最高
    if g.get("wrong"):
        return 2.5                     # 曾经错过、最近答对：要复习，降频
    return 1.0                         # 一直答对：低频巩固


def _pick_grammar(
    items: list[dict[str, Any]],
    weak_points: list[str],
    profile: dict[str, Any],
    exclude_ids: set[str] | None,
    exclude_texts: Iterable[str] | None,
    limit: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """语法题选点：薄弱点先占名额，剩下的名额按掌握程度加权随机。

    - 薄弱点（最近仍做错）最多拿 limit-1 个名额，保证每轮至少留 1 题做复习/巩固；
    - 其余名额加权随机：没测过(3.0) > 半掌握(2.5) > 一直对(1.0)，
      权重低的点被抽中概率小但不为 0，对应「降低频次，但不要不出现」。
    """
    blocked = {norm_text(t) for t in (exclude_texts or [])}
    used = set(exclude_ids or set())
    by_point: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        if str(it.get("id")) in used or norm_text(it.get("text")) in blocked:
            continue
        by_point.setdefault(str(it.get("grammar_point") or ""), []).append(it)
    for group in by_point.values():
        group.sort(
            key=lambda x: (int(x.get("level", 1)), len(str(x.get("text", ""))), str(x.get("id")))
        )

    picked: list[dict[str, Any]] = []
    # ① 薄弱点优先（最多 limit-1 题，给复习/巩固留位置）
    quota = max(1, limit - 1)
    for point in weak_points:
        if len(picked) >= quota:
            break
        group = by_point.get(point) or []
        if not group:
            continue
        for _ in range(min(len(group), quota - len(picked))):
            it = group.pop(0)
            picked.append(it)
            used.add(str(it.get("id")))
    # ② 剩余名额：加权随机（降频但不缺席）
    while len(picked) < limit:
        pool = [p for p, g in by_point.items() if g]
        if not pool:
            break
        weights = [max(_grammar_weight(profile.get(p)), 0.01) for p in pool]
        point = rng.choices(pool, weights=weights)[0]
        it = by_point[point].pop(0)
        if str(it.get("id")) in used:
            continue
        picked.append(it)
        used.add(str(it.get("id")))
    return picked[:limit]


def recommend_pool(
    agg: dict[str, Any],
    exclude_ids: set[str] | None = None,
    exclude_texts: Iterable[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """按测评薄弱环节分派三类练习：word / sentence / paragraph。

    针对性规则：
    1. 错读音素 → 先用「最小对立对」对着实测错误配（把 /θ/ 读成 /s/ 就配 θ 词 + s 词）；
    2. 漏读音素 → 给含该音素的短词，先把音读出来；
    3. 句子/段落 → 含弱音素的题优先，其余按长度递进；
    4. 所有推荐都不与定级测评考过的文本重复。
    """
    exclude = set(exclude_ids or set())
    blocked = list(exclude_texts or [])
    blocked_set = {norm_text(t) for t in blocked}
    dims = {k: v for k, v in (agg.get("dims") or {}).items() if v is not None}
    order = sorted(dims, key=lambda k: dims[k])          # 最弱在前
    weak_hits = list(agg.get("weak_phonemes") or [])
    confusions = list(agg.get("sound_confusions") or [])
    weak_sorted = sorted(weak_hits, key=lambda w: (_weak_rank(w), -int(w.get("count") or 0)))
    weak_ph = [str(w.get("phoneme")) for w in weak_sorted if w.get("phoneme")]
    rng = random.Random()
    out: dict[str, list[dict[str, Any]]] = {"word": [], "sentence": [], "paragraph": [], "grammar": []}

    # ---- 单词栏 ①：错读 → 最小对立对（θ/s 这种） ----
    for conf in confusions[:2]:
        expected = str(conf.get("expected") or "")
        actual = str(conf.get("actual") or "")
        if not expected or not actual or len(out["word"]) >= WORD_LIMIT:
            continue
        used = exclude | {str(w.get("id")) for w in out["word"]}
        for w in find_pair(expected, used, exclude_texts=blocked, actual=actual, rng=rng):
            if len(out["word"]) >= WORD_LIMIT:
                break
            it = dict(w)
            it.update(
                {
                    "dimension": "发音准确",
                    "origin": "词表配对",
                    "contrast": f"{expected}/{actual}",
                    "reason": f"对比纠音：先分清 /{expected}/ 和 /{actual}/，再读这个词",
                }
            )
            out["word"].append(it)

    # ---- 单词栏 ②：弱音素命中，按错读/漏读给不同理由 ----
    if len(out["word"]) < WORD_LIMIT and weak_ph:
        used = exclude | {str(w.get("id")) for w in out["word"]}
        hits = list(
            find_similar(
                weak_ph, exclude_ids=used, kinds=WORD_KINDS, limit=15, exclude_texts=blocked
            )
        )
        # 从易到难：按 level 分层再取，避免一栏里全是同难度的短词
        by_level: dict[int, list[dict[str, Any]]] = {}
        for h in hits:
            by_level.setdefault(int(h.get("level", 1)), []).append(h)
        for h in [x for lv in sorted(by_level) for x in by_level[lv]]:
            if len(out["word"]) >= WORD_LIMIT:
                break
            phones = item_phones(h)
            ph = next((p for p in phones if p in weak_ph), weak_ph[0])
            kinds = next((w.get("kinds") or {} for w in weak_sorted if w.get("phoneme") == ph), {})
            it = dict(h)
            it.update(
                {"dimension": "发音准确", "origin": "教材原题", "reason": _word_reason(ph, kinds)}
            )
            out["word"].append(it)

    if not out["word"]:      # 没有弱音素时，用高价值音素词兜底（同样从易到难）
        cand = [
            x
            for x in _seed_pool("word")
            if set(item_phones(x)) & FOCUS_PHONEMES
            and norm_text(x.get("text")) not in blocked_set
        ]
        for w in _pick_spread(cand, WORD_LIMIT, rng):
            it = dict(w)
            it.update({"dimension": "发音准确", "origin": "教材原题", "reason": "高价值音素巩固"})
            out["word"].append(it)

    # ---- 句子栏：含弱音素的短句优先，其余按长度递进 ----
    need_flu = any(k in order[:2] for k in ("流利度", "完整度"))
    sent_pool = [
        x for x in _seed_pool("sentence") if norm_text(x.get("text")) not in blocked_set
    ]
    if weak_ph:      # 含弱音素的句子优先，但仍按长度阶梯取（短 → 长）
        focus_sents = [x for x in sent_pool if set(item_phones(x)) & set(weak_ph)]
        sent_pool = focus_sents + [x for x in sent_pool if x not in focus_sents]
    for s in _pick_spread(sent_pool, 2, rng):
        it = dict(s)
        on_focus = bool(set(item_phones(it)) & set(weak_ph))
        it["dimension"] = "流利度" if need_flu else "完整度"
        it["origin"] = "教材原题"
        if on_focus and weak_ph:
            ph = next((p for p in item_phones(it) if p in weak_ph), weak_ph[0])
            it["reason"] = f"在整句里练 /{ph}/，把词连起来读"
        else:
            it["reason"] = f"针对「{it['dimension']}」"
        out["sentence"].append(it)
    if "内容命中" in order[:2]:
        for a in _pick_by_length(_seed_pool("semi_open"), 1, exclude_texts=blocked):
            it = dict(a)
            it.update(
                {
                    "dimension": "内容命中",
                    "origin": "教材原题",
                    "reason": "练「把意思说全」，不要只答一两个词",
                }
            )
            out["sentence"].append(it)

    # ---- 段落栏：固定 1 段连贯朗读 ----
    for pg in _pick_by_length(
        _seed_pool("paragraph"), 1, exclude_texts=blocked, weak_phonemes=weak_ph
    ):
        it = dict(pg)
        it["dimension"] = "流利度" if need_flu else "完整度"
        it["origin"] = "教材原题"
        it["reason"] = f"连贯朗读练「{it['dimension']}」，一口气读完"
        out["paragraph"].append(it)

    # ---- 语法栏：仍做错的知识点优先；已答对的降频复习，但不会完全不出现 ----
    profile = agg.get("grammar_profile") or {}
    weak_grammar = list(agg.get("weak_grammar") or [])
    weak_points = [str(g.get("point")) for g in weak_grammar if g.get("point")]

    def _grammar_candidates() -> list[dict[str, Any]]:
        return [
            i
            for i in iter_items()
            if str(i.get("kind")) in RULE_KINDS and i.get("verified") is not False
        ]

    picked = _pick_grammar(
        _grammar_candidates(),
        weak_points=weak_points,
        profile=profile,
        exclude_ids=exclude,
        exclude_texts=blocked,
        limit=GRAMMAR_LIMIT,
        rng=rng,
    )
    # 薄弱点一道题都排不出来（小题量知识点，且练过的题都被排除）时，让大模型补题；
    # 结果落盘到 data/gen_cache/，本轮与以后都能直接用。
    if weak_points and not any(str(i.get("grammar_point")) in set(weak_points) for i in picked):
        from . import grammar_topup

        grammar_topup.ensure_topup(weak_points[0], need=GRAMMAR_LIMIT)
        picked = _pick_grammar(
            _grammar_candidates(),
            weak_points=weak_points,
            profile=profile,
            exclude_ids=exclude,
            exclude_texts=blocked,
            limit=GRAMMAR_LIMIT,
            rng=rng,
        )
    for it in picked:
        point = str(it.get("grammar_point") or "")
        hit = next((g for g in weak_grammar if str(g.get("point")) == point), None)
        if hit:
            detail = f"（{hit['wrong_as'][0]}）" if hit.get("wrong_as") else ""
            reason = f"「{point}」上次 {hit['total']} 题错 {hit['wrong']} 题{detail}，这次再练"
        elif point in profile:
            reason = f"「{point}」已经能答对，隔几轮再巩固一下"
        else:
            reason = f"新知识点「{point}」，先试试"
        item = dict(it)
        item.update(
            {"dimension": "语法", "origin": "教材知识点", "point": point, "reason": reason}
        )
        out["grammar"].append(item)

    for group in out:
        for it in out[group]:
            it["chain"] = chain_kind_of(it)
            it.setdefault("group", group)
            it.setdefault("point", str(it.get("grammar_point") or ""))
    return out


def stats() -> dict[str, Any]:
    items = all_items()
    kinds: dict[str, int] = {}
    for item in items:
        kinds[item.get("kind", "?")] = kinds.get(item.get("kind", "?"), 0) + 1
    return {
        "source": corpus_source(),
        "units": len(load_corpus().get("units", [])),
        "items": len(items),
        "by_kind": kinds,
    }
