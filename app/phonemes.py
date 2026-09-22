"""极简「拼读 → 音素」近似工具。

用途只有一个：把评测报告里的弱音素（如 /θ/）映射回孩子学过的单词，
用于「同类习题推荐」。不追求 G2P 精度，命中率够用即可。
"""

from __future__ import annotations

import re

# 多音素规则（越靠前优先级越高）
_RULES: list[tuple[str, str]] = [
    (r"tch", "tʃ"),
    (r"dge", "dʒ"),
    (r"igh", "aɪ"),
    (r"air", "eə"),
    (r"are", "eə"),
    (r"ear", "ɪə"),
    (r"eer", "ɪə"),
    (r"ere", "ɪə"),
    (r"ure", "ʊə"),
    (r"our", "aʊ"),
    (r"ough", "ʌ"),
    (r"ch", "tʃ"),
    (r"sh", "ʃ"),
    (r"th", "θ"),
    (r"ph", "f"),
    (r"wh", "w"),
    (r"ck", "k"),
    (r"ng", "ŋ"),
    (r"qu", "kw"),
    (r"ai", "eɪ"),
    (r"ay", "eɪ"),
    (r"ea", "iː"),
    (r"ee", "iː"),
    (r"ie", "aɪ"),
    (r"oa", "əʊ"),
    (r"oe", "əʊ"),
    (r"oo", "uː"),
    (r"ou", "aʊ"),
    (r"ow", "əʊ"),
    (r"oi", "ɔɪ"),
    (r"oy", "ɔɪ"),
    (r"ar", "ɑː"),
    (r"er", "ɜː"),
    (r"ir", "ɜː"),
    (r"ur", "ɜː"),
    (r"or", "ɔː"),
    (r"aw", "ɔː"),
    (r"au", "ɔː"),
]

_LETTERS: dict[str, str] = {
    "a": "æ", "b": "b", "c": "k", "d": "d", "e": "e", "f": "f", "g": "g", "h": "h",
    "i": "ɪ", "j": "dʒ", "k": "k", "l": "l", "m": "m", "n": "n", "o": "ɒ", "p": "p",
    "q": "kw", "r": "r", "s": "s", "t": "t", "u": "ʌ", "v": "v", "w": "w", "x": "ks",
    "y": "j", "z": "z",
}

# 弱音素 → 常见拼写（用于从语料里挑同类练习词）
PHONEME_TO_GRAPHEME: dict[str, list[str]] = {
    "θ": ["th"],
    "ð": ["th"],
    "ʃ": ["sh", "ti", "ci", "ssi"],
    "tʃ": ["ch", "tch"],
    "dʒ": ["j", "ge", "dge", "gi"],
    "ŋ": ["ng", "nk"],
    "r": ["r", "wr"],
    "l": ["l", "ll"],
    "v": ["v"],
    "w": ["w", "wh"],
    "j": ["y"],
    "kw": ["qu"],
    "ks": ["x"],
    "æ": ["a"],
    "e": ["e", "ea"],
    "ɪ": ["i", "y"],
    "iː": ["ee", "ea", "e", "ie"],
    "ɒ": ["o", "a"],
    "ʌ": ["u", "o", "ou"],
    "ɑː": ["ar", "a"],
    "ɔː": ["or", "aw", "au", "al"],
    "ɜː": ["er", "ir", "ur"],
    "ə": ["a", "e", "er", "o"],
    "uː": ["oo", "u", "ue", "ew", "ui"],
    "ʊ": ["oo", "u", "ou"],
    "eɪ": ["a", "ai", "ay", "ei"],
    "aɪ": ["i", "igh", "y", "ie"],
    "ɔɪ": ["oi", "oy"],
    "əʊ": ["o", "oa", "ow", "oe"],
    "aʊ": ["ou", "ow"],
    "ɪə": ["ear", "eer", "ere"],
    "eə": ["air", "are", "ear"],
    "ʊə": ["ure", "oor"],
    "s": ["s", "ss", "ce", "ci"],
    "z": ["z", "s", "se"],
    "f": ["f", "ph", "gh"],
    "h": ["h"],
    "m": ["m"],
    "n": ["n", "kn"],
    "p": ["p"],
    "b": ["b"],
    "t": ["t", "tt"],
    "d": ["d"],
    "k": ["k", "c", "ck"],
    "g": ["g", "gu"],
}

# 弱音素 → 给七年级孩子的中文纠正提示
PHONEME_TIPS: dict[str, str] = {
    "θ": "舌尖轻轻放在上下牙齿中间，往外吹小风，别用「s」代替（three、thank）。",
    "ð": "舌尖轻碰上牙齿，嗓子里要出声、震动（this、mother）。",
    "ʃ": "嘴唇往前噘成小圆圈，像让别人安静「嘘——」（fish、ship）。",
    "tʃ": "先轻轻「t」再「嘘」，连成一个音（chair、teacher）。",
    "dʒ": "舌头抵上牙床，嗓子里出声（jump、orange）。",
    "ŋ": "鼻子里出声，舌头后部抬起来（sing、morning）。",
    "r": "舌尖卷起来，不碰上颚，像小老虎「r」（red、rabbit）。",
    "l": "舌尖顶住上牙齿后面，气流从两边走（like、look）。",
    "v": "上牙轻轻咬住下嘴唇，嗓子震动（five、very）。",
    "w": "双唇收成小圆口再放开（we、water）。",
    "j": "像说「耶」的开头（yellow、yes）。",
    "æ": "嘴巴张得大大的，下巴往下（cat、apple）。",
    "e": "嘴张一小指宽，短促（red、pen）。",
    "ɪ": "短短的「衣」，不要拖长（sit、fish）。",
    "iː": "拉长的「衣」，嘴角往两边咧（see、green）。",
    "ɒ": "嘴巴圆圆的，短音（dog、box）。",
    "ʌ": "放松的「啊」，短促（duck、cup）。",
    "ɑː": "嘴巴张大往后，长音（car、father）。",
    "ɔː": "嘴唇收圆往前，长音（ball、four）。",
    "ɜː": "舌头放平不动，长音（bird、girl）。",
    "uː": "撅嘴拉长的「乌」（food、blue）。",
    "ʊ": "短短的「乌」（book、look）。",
    "eɪ": "从「ei」滑到「i」，嘴角拉开（name、cake）。",
    "aɪ": "从「a」滑到「i」（like、nine）。",
    "ɔɪ": "从「o」滑到「i」（boy、toy）。",
    "əʊ": "从「e」滑到「乌」，收成圆嘴（nose、hello）。",
    "aʊ": "从「a」滑到「乌」（mouse、cow）。",
    "s": "牙齿轻轻合上，气流细而长（six、see）。",
    "z": "和 s 同位置，但嗓子要震动（zoo、nose）。",
    "f": "上牙轻咬下唇，只出气不出声（fish、four）。",
    "h": "像哈一口气（hello、head）。",
    "k": "舌根抬起再放开（cat、book）。",
    "g": "和 k 同位置，嗓子出声（dog、bag）。",
    "p": "双唇闭上再爆破打开（pen、pig）。",
    "b": "和 p 同位置，嗓子出声（bag、ball）。",
    "t": "舌尖顶上牙床再弹开（ten、cat）。",
    "d": "和 t 同位置，嗓子出声（dog、duck）。",
    "m": "双唇闭上，鼻子里出声（mother、milk）。",
    "n": "舌尖顶上牙床，鼻子里出声（nose、name）。",
}

# 音素 → 大白话拟音（给家长/孩子看）
PHONEME_LABEL: dict[str, str] = {
    "θ": "th（清）", "ð": "th（浊）", "ʃ": "sh", "tʃ": "ch", "dʒ": "j",
    "ŋ": "后鼻音 ng", "æ": "大口 æ", "ʌ": "短音 ʌ", "ɒ": "短音 ɒ",
    "ɜː": "长音 er", "ɑː": "长音 ar", "ɔː": "长音 or", "iː": "长音 ee",
    "ɪ": "短音 i", "uː": "长音 oo", "ʊ": "短音 oo", "eɪ": "双元音 ei",
    "aɪ": "双元音 ai", "ɔɪ": "双元音 oi", "əʊ": "双元音 ou", "aʊ": "双元音 au",
}


# 拼写是 th、实际读浊音 /ð/ 的高频词（七年级课本里反复出现，闭集可枚举）
# 不区分的话，this / that / the 会被当成 /θ/ 词推荐给 /θ/ 读不准的学生，方向完全错。
D_TH_WORDS: set[str] = {
    "the", "this", "that", "these", "those", "they", "them", "their", "theirs",
    "there", "then", "than", "though", "thus", "mother", "father", "brother",
    "other", "others", "another", "weather", "whether", "either", "neither",
    "with", "without", "together", "rather", "feather", "leather", "bother",
    "gather", "breathe", "bathe", "clothe", "clothes", "smooth", "southern",
    "northern", "worthy", "rhythm", "grandmother", "grandfather", "themselves",
}


def guess_phones(word: str) -> list[str]:
    """把单词近似拆成音素列表（只用于同类题推荐，不用于判分）。"""
    text = re.sub(r"[^a-z]", "", (word or "").lower())
    if not text:
        return []
    soft_th = text in D_TH_WORDS
    phones: list[str] = []
    i = 0
    while i < len(text):
        matched = False
        for pattern, phone in _RULES:
            if text.startswith(pattern, i):
                phones.append("ð" if (pattern == "th" and soft_th) else phone)
                i += len(pattern)
                matched = True
                break
        if matched:
            continue
        ch = text[i]
        if ch in _LETTERS:
            # 词尾不发音的 e
            if ch == "e" and i == len(text) - 1 and len(phones) > 0:
                i += 1
                continue
            phones.append(_LETTERS[ch])
        i += 1
    return phones


def graphemes_for(phoneme: str) -> list[str]:
    return PHONEME_TO_GRAPHEME.get(phoneme, [phoneme])


def tip_for(phoneme: str) -> str:
    return PHONEME_TIPS.get(phoneme, "跟着老师把这个音拉长、夸张地再读三遍。")


def label_for(phoneme: str) -> str:
    return PHONEME_LABEL.get(phoneme, phoneme)


def matches_phoneme(text: str, phoneme: str) -> bool:
    """单词拼写里是否含有该音素对应的常见字母组合。"""
    low = (text or "").lower()
    if phoneme in ("θ", "ð"):
        # th 拼写要按浊化例外表分流，否则 this/that/the 会被当成 /θ/ 词
        soft = low.strip() in D_TH_WORDS
        return soft if phoneme == "ð" else ("th" in low and not soft)
    for g in graphemes_for(phoneme):
        if g in low:
            return True
    return False


# ---- 纠音对齐表（错读 / 漏读 / 多读）----
KIND_TEXT: dict[str, str] = {
    "mispron": "错读",
    "missing": "漏读",
    "addition": "多读",
    "weak": "发音不到位",
    "correct": "读对",
}

# 常见「把 A 读成 B」的成因与一个动作可改的练法
SUBSTITUTION_TIPS: dict[tuple[str, str], str] = {
    ("θ", "s"): "舌尖没伸出来、用 /s/ 顶替：对着镜子把舌尖轻放在上下齿之间再送气。",
    ("θ", "f"): "用下唇碰上门牙了：舌尖要伸到齿间，不是咬下唇。",
    ("s", "θ"): "舌尖伸太出来了：/s/ 舌尖停在齿后即可，别伸到齿间。",
    ("ð", "d"): "舌尖没伸到齿间、还挡住了气：舌尖轻触上齿边缘，嗓子里出声。",
    ("ð", "z"): "位置对了但舌尖太靠后：舌尖轻碰上齿边缘再让声带振动。",
    ("z", "s"): "声带没振动：/z/ 与 /s/ 同位置，但必须出声音。",
    ("r", "l"): "舌尖碰到上颚了：/r/ 要卷舌尖、但不接触任何地方。",
    ("l", "r"): "/l/ 不要卷舌：舌尖顶住上齿龈，气流从舌两侧走。",
    ("n", "l"): "舌尖位置不对：/n/ 舌尖顶上齿龈、气流从鼻子出。",
    ("ŋ", "n"): "舌根没抬起来：/ŋ/ 用舌根抵住软腭，鼻音出。",
    ("v", "w"): "上齿没咬下唇：/v/ 要上齿轻咬下唇并振动声带。",
    ("w", "v"): "/w/ 是双唇收圆，不用牙齿参与。",
    ("æ", "e"): "嘴巴张得不够大：/æ/ 要下巴明显往下。",
    ("e", "æ"): "嘴张太大了：/e/ 只需一小指宽。",
    ("iː", "ɪ"): "拖得太短：/iː/ 要拉长、嘴角往两边咧。",
    ("ɪ", "iː"): "拖太长了：/ɪ/ 要短促，不要拉长。",
    ("uː", "ʊ"): "/uː/ 要撅嘴拉长。",
    ("eɪ", "e"): "/eɪ/ 是滑动音：从 /e/ 滑到 /ɪ/，别读成单个静态音。",
}


def kind_text(kind: str) -> str:
    return KIND_TEXT.get(kind, kind)


def issue_tip(expected: str, actual: str = "", kind: str = "") -> str:
    """针对一处具体问题给练法（错读优先给成因提示）。"""
    if kind == "missing" or not actual:
        return f"/{expected}/ 漏读了：放慢速度逐音跟读，确保每个音都读到位。"
    if kind == "addition":
        return f"多读出了 /{actual}/：跟读时注意不要加音，读准节奏。"
    if kind == "weak":
        return f"/{expected}/ 能听出来但不到位：对照标准音慢速夸张读 3 遍。"
    return SUBSTITUTION_TIPS.get(
        (expected, actual),
        f"你把 /{expected}/ 读成了 /{actual}/：先听标准音，再慢速对比录一遍。",
    )
