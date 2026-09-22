#!/usr/bin/env python3
"""驰声 MCP 一键体检：连接 / 工具列表 / 各评测内核是否真的可用。

用法：
    python scripts/check_chivox.py
    PROBE_TIMEOUT=20 python scripts/check_chivox.py     # 单个 core 等待上限（默认 40 秒）

判读：
    只有 tools/list 通过、评测都报 51000/53000/504 → 驰声服务端的评测内核不可用，
    不是本地代码或音频的问题，请把输出发给驰声技术支持。
"""

from __future__ import annotations

import base64
import io
import math
import os
import struct
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audio import to_eval_audio  # noqa: E402
from app.chivox import ChivoxMCP  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "samples" / "sample1.wav"


def sample_b64() -> str:
    """用一段**真实语音**做探测（data/samples/sample1.wav），并转成 mp3 再 base64。

    注意两个坑：
    1. 合成正弦音等非语音音频会被驰声内核拒绝（51000 / 53000），不能用来判断服务可用性；
    2. audio_base64 通道对 wav(PCM) 会卡死超时，必须用 mp3。
    """
    if not SAMPLE.exists():
        raise SystemExit(
            "缺少探测音频：请先准备 data/samples/sample1.wav（2~5 秒真实英语语音）\n"
            "  curl -s -o /tmp/s.mp3 '<可访问的英语音频URL>' && \\\n"
            "  ffmpeg -y -i /tmp/s.mp3 -t 30 -ac 1 -ar 16000 -sample_fmt s16 data/samples/sample1.wav"
        )
    data, kind = to_eval_audio(SAMPLE.read_bytes(), ".wav")
    print("       探测音频：%s（转码为 %s，base64 %d 字节）" % (SAMPLE.name, kind, len(base64.b64encode(data))))
    return base64.b64encode(data).decode("ascii")


CASES = [
    ("en_word_eval", "hello"),
    ("en_sentence_eval", "I like red."),
    ("en_choice_eval", "cat|dog|duck"),
    ("en_paragraph_eval", "This is my book. I like it."),
    ("en_semi_open_eval", "My name is Tom."),
    ("cn_sentence_eval", "你好"),
]

BAD_WORDS = ("失败", "timeout", "unsuccessfully", "Gateway Time-out")


def main() -> int:
    timeout = float(os.getenv("PROBE_TIMEOUT") or 10)
    mcp = ChivoxMCP(timeout=timeout)
    t0 = time.time()
    try:
        tools = mcp.list_tools()
        print("[OK]   tools/list: %d 个工具，%.2fs（连接与鉴权正常）" % (len(tools), time.time() - t0))
    except Exception as exc:
        print("[FAIL] tools/list: %s" % str(exc).splitlines()[0][:120])
        return 2
    audio = sample_b64()
    bad = 0
    print("       单个 core 等待上限 %.0f 秒，用真实语音逐项探测：" % timeout)
    for tool, ref in CASES:
        t = time.time()
        try:
            result = mcp.evaluate(tool=tool, ref_text=ref, audio_base64=audio)
            dt = time.time() - t
            text = str(result)
            if isinstance(result, str) and any(w in text for w in BAD_WORDS):
                print("[FAIL] %-20s %5.1fs %s" % (tool, dt, text[:80]))
                bad += 1
            else:
                overall = (result.get("result") or {}).get("overall") if isinstance(result, dict) else result
                print("[OK]   %-20s %5.1fs overall=%s" % (tool, dt, overall))
        except Exception as exc:
            print("[FAIL] %-20s %5.1fs %s" % (tool, time.time() - t, str(exc).splitlines()[0][:80]))
            bad += 1
    mcp.close()
    if bad == 0:
        print("结论：全部可用 ✅")
        return 0
    print("结论：%d/%d 个评测内核不可用 —— 属驰声服务端问题（常见错误码 51000 内核启动失败 / 53000 内核超时 / HTTP 504）。" % (bad, len(CASES)))
    print("      请把本输出、报障时间、以及门户上的 applicationId 发给驰声技术支持。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
