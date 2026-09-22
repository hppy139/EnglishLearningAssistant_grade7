#!/usr/bin/env python3
"""本地服务冒烟测试：合成一段 wav，走完 组卷 → 上传评测 → 报告。

用法（默认连 8099，可用 BASE 环境变量覆盖）：
    python scripts/_smoke_http.py
    BASE=http://127.0.0.1:8080 python scripts/_smoke_http.py
"""
import io
import math
import os
import struct
import sys
import wave

import httpx

BASE = os.getenv("BASE", "http://127.0.0.1:8099").rstrip("/")
buf = io.BytesIO()
w = wave.open(buf, "wb")
w.setnchannels(1)
w.setsampwidth(2)
w.setframerate(16000)
w.writeframes(
    b"".join(struct.pack("<h", int(2500 * math.sin(2 * math.pi * 200 * t / 16000))) for t in range(16000 * 3))
)
w.close()
wav = buf.getvalue()

c = httpx.Client(base_url=BASE, timeout=180)
h = c.get("/api/health").json()
print(
    "health:", h["corpus"]["items"], "题 /", h["rag"]["chunks"], "切片 / llm:", h["llm_model"],
    "| llm_last_error:", h.get("llm_last_error") or "无",
)
s = c.post("/api/session", json={"student": "小明", "size": 3}).json()
# 工具按题型固定链路（app/tools.py: CHAINS），chain="rule" 表示语法题走本地规则判分
print("plan:", [(i["group"], i["kind"], i["chain"], i["tool"]) for i in s["items"]])
r = c.post(f"/api/session/{s['session_id']}/item/0", files={"audio": ("t.wav", wav, "audio/wav")})
print("item0:", r.status_code, str(r.json())[:220] if r.status_code == 200 else r.text[:220])
rep = c.post(f"/api/session/{s['session_id']}/report")
if rep.status_code != 200:
    print("report:", rep.status_code, rep.text[:300])
    sys.exit(1)
d = rep.json()
a = d.get("analysis") or {}
practice = d.get("practice") or {}
print(
    "report: ok | analysis.source =", a.get("source", "?"),
    "| 推荐题数:", {k: len(v) for k, v in practice.items()},
)
print("  conclusion:", str(a.get("conclusion") or "")[:120])
print(
    "  画像: error_profile =", d.get("error_profile") or {},
    "| grammar_score =", d.get("grammar_score"),
)
