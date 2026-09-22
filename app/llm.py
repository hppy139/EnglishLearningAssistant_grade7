"""大模型（DeepSeek / OpenAI 兼容接口）封装。

大模型在本项目里干三件事：
1. 看一眼题目，决定用驰声的哪个 MCP 工具评（plan_tool）；
2. 看完定级卷成绩，写「水平定级 + 弱点分析 + 鼓励语」（analyze）；
3. 根据弱点从候选题库里挑同类练习（recommend）。

没有 LLM_API_KEY 时全部返回 None，系统自动降级为规则引擎。
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

import httpx

from . import config

# 最近一次调用失败原因。之前异常被静默吞掉，模型名写错也只会「悄悄降级成规则」，
# 这里留痕并打进服务日志，health 接口也会带出来。
LAST_ERROR: dict[str, str] = {}


def last_error() -> str:
    return LAST_ERROR.get("chat", "")

# 驰声 MCP 里的英文口语评测工具（交给大模型挑）
TOOL_CATALOG: list[dict[str, str]] = [
    {"name": "en_word_eval", "desc": "英文单词评测：单个单词发音，返回总分和每个音标得分", "use": "单词跟读题（cat / three / apple）"},
    {"name": "en_word_correction", "desc": "英文单词纠音：单词发音 + 纠正建议", "use": "孩子同一个词反复读错时"},
    {"name": "en_phonics_eval", "desc": "英文自然拼读评测：字母/字母组合的发音", "use": "字母音、自然拼读题（a / sh / th）"},
    {"name": "en_sentence_eval", "desc": "英文句子评测：返回总分、流利度、准确度、完整度及逐词得分", "use": "短句跟读题（I like red.）"},
    {"name": "en_sentence_correction", "desc": "英文句子纠音：句子发音 + 纠正建议", "use": "句子读得磕巴、需要逐词纠正时"},
    {"name": "en_vocab_eval", "desc": "英文词语评测：一次评测多个词语", "use": "一次读好几个词的词汇题"},
    {"name": "en_paragraph_eval", "desc": "英文段落评测：段落朗读，逐句逐词得分", "use": "小短文朗读题"},
    {"name": "en_choice_eval", "desc": "英文口语选择评测：听/说后从预设选项中判断", "use": "选择题（cat|dog|duck）"},
    {"name": "en_semi_open_eval", "desc": "英文半开放题评测：看图说话、回答问题等开放表达", "use": "问答题（What's your name?）"},
    {"name": "en_realtime_eval", "desc": "英文实时朗读评测：边读边评", "use": "需要实时反馈的跟读"},
    {"name": "create_stream_session", "desc": "创建流式评测会话，返回 session_id，客户端用 WebSocket 推实时音频", "use": "浏览器麦克风实时流式评测"},
]

ALLOWED_TOOLS = {t["name"] for t in TOOL_CATALOG}

# 没大模型时的兜底映射
RULE_TOOL_MAP: dict[str, str] = {
    "word": "en_word_eval",
    "phonics": "en_phonics_eval",
    "sentence": "en_sentence_eval",
    "vocab": "en_vocab_eval",
    "paragraph": "en_paragraph_eval",
    "choice": "en_choice_eval",
    "semi_open": "en_semi_open_eval",
    "realtime": "en_realtime_eval",
}


class LLM:
    def __init__(self) -> None:
        self.api_key = config.llm_api_key()
        self.base_url = config.llm_base_url()
        self.model = config.llm_model()
        self.timeout = config.llm_timeout()

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _endpoint(self) -> str:
        base = self.base_url
        if base.endswith("/chat/completions"):
            return base
        if not re.search(r"/v\d+$", base):
            base = base + "/v1"
        return base + "/chat/completions"

    def chat(
        self, system: str, user: str, temperature: float = 0.3, max_tokens: int | None = None
    ) -> str | None:
        if not self.available:
            return None
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "stream": False,
        }
        # 注意：deepseek 是推理模型，「思考」也消耗 max_tokens。
        # 限制过小会让正文为空（表现为静默降级），所以默认不限；
        # 要提速请用 reasoning_effort=low，而不是砍 max_tokens。
        if max_tokens:
            payload["max_tokens"] = max_tokens
        effort = config.llm_reasoning_effort()
        if effort:
            payload["reasoning_effort"] = effort      # 少想一点，快一倍
        try:
            resp = httpx.post(
                self._endpoint(),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
                trust_env=config.http_trust_env(),
            )
            if resp.status_code == 400 and effort:
                # 该模型 / 中转不支持 reasoning_effort：去掉参数重试一次
                payload.pop("reasoning_effort", None)
                resp = httpx.post(
                    self._endpoint(),
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                    trust_env=config.http_trust_env(),
                )
            if resp.status_code >= 400:
                LAST_ERROR["chat"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
                print(f"[LLM] 调用失败：{LAST_ERROR['chat']}", file=sys.stderr)
                return None
            LAST_ERROR.pop("chat", None)
            data = resp.json()
            try:
                msg = data["choices"][0].get("message") or {}
            except (KeyError, IndexError, TypeError):
                msg = {}
            content = str(msg.get("content") or "").strip()
            if not content:
                # deepseek 是推理模型：思考（reasoning_content）会占用 max_tokens，
                # 被占满时 content 会是空的 —— 退一步从思考内容里抠 JSON
                content = str(msg.get("reasoning_content") or "").strip()
                if content:
                    print("[LLM] content 为空，改用 reasoning_content 兜底", file=sys.stderr)
            if not content:
                LAST_ERROR["chat"] = f"空响应：{str(data)[:160]}"
                print(f"[LLM] {LAST_ERROR['chat']}", file=sys.stderr)
                return None
            return content
        except Exception as exc:
            LAST_ERROR["chat"] = f"{type(exc).__name__}: {exc}"
            print(f"[LLM] 调用异常：{LAST_ERROR['chat']}", file=sys.stderr)
            return None

    def chat_json(
        self, system: str, user: str, temperature: float = 0.2, max_tokens: int | None = None
    ) -> dict[str, Any] | None:
        text = self.chat(system, user, temperature, max_tokens)
        if not text:
            return None
        return _first_json(text)

    # ---------- 1. 工具选择 ----------
    def plan_tool(self, item: dict[str, Any], history: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        catalog = "\n".join(f"- {t['name']}：{t['desc']}（适合：{t['use']}）" for t in TOOL_CATALOG)
        system = (
            "你是初中七年级（人教版 Go for it!）英语口语评测的教研专家，需要为每道题挑选最合适的驰声（Chivox）语音评测工具。\n"
            "可选工具：\n" + catalog + "\n"
            "规则：\n"
            "1. 只能从上面的工具名里选一个，不要自造名字；\n"
            "2. 能短不长：单词题不要升格成句子评测，除非题目本身是段落；\n"
            "3. ref_text 必须是英文参考文本：单词题给单词本身，句子题给完整句子，半开放题给参考答案；\n"
            "4. 只输出 JSON，不要解释。\n"
            "输出格式：{\"tool\": \"en_word_eval\", \"ref_text\": \"cat\", \"note\": \"一句话说明为什么选它\"}"
        )
        user = json.dumps(
            {
                "item": {
                    "id": item.get("id"),
                    "kind": item.get("kind"),
                    "text": item.get("text"),
                    "zh": item.get("zh"),
                    "sample": item.get("sample"),
                    "answer": item.get("answer"),
                    "level": item.get("level"),
                    "unit": item.get("unit_name"),
                },
                "history": history or [],
            },
            ensure_ascii=False,
        )
        data = self.chat_json(system, user)
        if not data:
            return None
        tool = str(data.get("tool") or "").strip()
        if tool not in ALLOWED_TOOLS:
            return None
        return {
            "tool": tool,
            "ref_text": str(data.get("ref_text") or "").strip(),
            "note": str(data.get("note") or "").strip(),
            "source": "llm",
        }

    # ---------- 2. 定级 + 弱点分析 ----------
    def analyze(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        system = (
            "给 12-13 岁学生写三句话口语小结，直接给结果。\n"
            "① conclusion：一句话结论（分数 + 最强/最弱维度）；\n"
            "② evidence：1-3 条，引用给定分数或音素；错读就写「把 /θ/ 读成 /s/」这种具体形式，"
            "语法低就点名 weak_grammar 里的知识点；\n"
            "③ action：一个可执行动作（含时长与验收）；\n"
            "口语化、不指责，不出现「差/落后」等词。\n"
            "只输出 JSON：{\"conclusion\":\"...\",\"evidence\":[\"...\"],\"action\":\"...\"}"
        )
        # 注意：max_tokens 要留足「思考 + 正文」的余量（太小 content 会为空），
        # 但也不能给太大：deepseek 是推理模型，给得多它就多想，整体更慢。
        data = self.chat_json(system, json.dumps(payload, ensure_ascii=False))
        if not data or not data.get("conclusion"):
            return None
        data["source"] = "llm"
        if not isinstance(data.get("evidence"), list):
            data["evidence"] = [str(data["evidence"])] if data.get("evidence") else []
        return data

    # ---------- 3. 同类题推荐 ----------
    def recommend(self, payload: dict[str, Any]) -> list[dict[str, Any]] | None:
        system = (
            "你是初中英语教研员。从 candidates 里挑出最该练的题，直接给结果。\n"
            "规则：\n"
            "1. 只能用 candidates 里的 id（都是课本内容，不要自造）；\n"
            "2. 语法先补 weak_grammar 里的知识点，再补没测过的；发音错读配对比词、漏读配音素短词；\n"
            "3. level 小的优先；\n"
            "4. 最多 8 题：word ≤3、sentence ≤3、grammar ≤3、paragraph ≤1；\n"
            "5. reason 一句话，点名音素或知识点。\n"
            "只输出 JSON：{\"picks\":[{\"id\":\"...\",\"reason\":\"...\",\"focus\":\"发音准确\"}]}"
        )
        data = self.chat_json(system, json.dumps(payload, ensure_ascii=False))
        if not data or not isinstance(data.get("picks"), list):
            return None
        return [p for p in data["picks"] if isinstance(p, dict) and p.get("id")]


def _first_json(text: str) -> dict[str, Any] | None:
    """从大模型输出里抠出第一个 JSON 对象（容忍 ```json 代码块）。"""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    candidates = [cleaned]
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        candidates.append(match.group(0))
    for raw in candidates:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None
