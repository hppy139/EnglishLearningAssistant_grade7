"""全局配置：全部从环境变量 / .env 读取，Key 不下发到浏览器。"""

from __future__ import annotations

import os

from . import ROOT


def _load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(path, override=False)
    except Exception:
        # 没装 python-dotenv 时手搓解析，够用即可
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def env(name: str, default: str = "") -> str:
    raw = (os.getenv(name) or default).strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        raw = raw[1:-1]
    return raw.strip()


def env_int(name: str, default: int) -> int:
    raw = env(name)
    try:
        return int(raw)
    except ValueError:
        return default


# ---------- 驰声 ----------
def chivox_api_key() -> str:
    return env("CHIVOX_API_KEY")


def chivox_mcp_url() -> str:
    return env("CHIVOX_MCP_URL", "https://mcp.cloud.chivox.com").rstrip("/")


def chivox_accent() -> int:
    return env_int("CHIVOX_ACCENT", 3)


def chivox_rank() -> int:
    return env_int("CHIVOX_RANK", 100)


def chivox_skip_tools() -> set[str]:
    """暂时不可用的评测工具（逗号分隔），组卷时跳过对应题型。

    例：CHIVOX_SKIP_TOOLS=en_choice_eval,en_semi_open_eval
    用 scripts/check_chivox.py 探测出哪些 core 报 51000 后填这里。
    """
    raw = env("CHIVOX_SKIP_TOOLS")
    return {part.strip() for part in raw.split(",") if part.strip()}


def chivox_timeout() -> float:
    """MCP 请求超时（秒）。驰声评测偶尔较慢，默认 120 秒。"""
    try:
        return float(env("CHIVOX_TIMEOUT", "120"))
    except ValueError:
        return 120.0


def http_trust_env() -> bool:
    """是否沿用 http_proxy / all_proxy 等环境代理。

    默认 False：容器/集群里常见 all_proxy=socks://... 但没装 socksio，
    或代理已停，会让请求在发出前直接失败。需要走代理时设 HTTP_TRUST_ENV=1。
    """
    return env("HTTP_TRUST_ENV", "0") not in ("0", "", "false", "False", "no")


def http_timeout() -> float:
    try:
        return float(env("HTTP_TIMEOUT", "60"))
    except ValueError:
        return 60.0


# ---------- 大模型 ----------
def llm_api_key() -> str:
    return env("LLM_API_KEY")


def llm_base_url() -> str:
    return env("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")


def llm_model() -> str:
    return env("LLM_MODEL", "deepseek-chat")


def llm_enabled() -> bool:
    """总开关：LLM_DISABLE=1 时全部走规则（报告/推荐秒回，但文案是模板）。

    deepseek-flash 是推理模型，一次调用要等 10 秒上下；
    想要快就设 LLM_DISABLE=1，想要质量就保持默认。
    """
    return env("LLM_DISABLE", "0").lower() not in ("1", "true", "yes", "on")


def recommend_use_llm() -> bool:
    """推荐题是否交给大模型挑（RECOMMEND_USE_LLM=0 时纯规则、秒回）。

    推理模型挑题质量更好但要等十几秒，用户可按需切换。
    """
    return env("RECOMMEND_USE_LLM", "1").lower() not in ("0", "false", "no", "off", "")


def grammar_topup_online() -> bool:
    """题量不足的知识点，是否允许在线让大模型补题（默认开）。

    补出来的题会落盘到 data/gen_cache/，之后直接复用，不会重复调用大模型；
    演示时不想多等，可以设 GRAMMAR_TOPUP_ONLINE=0。
    """
    return env("GRAMMAR_TOPUP_ONLINE", "1").lower() not in ("0", "false", "no", "off")


def llm_reasoning_effort() -> str:
    """推理强度（deepseek 等推理模型支持）：low / medium / high。

    这些任务要的就是一小段 JSON，让模型少想能快一倍；
    设成空字符串则不传该参数（兼容不支持它的接口）。
    """
    return env("LLM_REASONING_EFFORT", "low")


def llm_timeout() -> float:
    return float(env("LLM_TIMEOUT", "60"))
