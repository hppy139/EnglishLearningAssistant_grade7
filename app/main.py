"""七年级英语口语定级助手：Web API。

链路：教材语料组卷 → 大模型挑驰声工具 → 录音/上传/指定文件评测 → 大模型分析弱点 → 推同类练习。
"""

from __future__ import annotations

import base64
import os
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import ROOT, config, tools
from .audio import (
    MAX_SECONDS,
    has_arecord,
    has_ffmpeg,
    probe_wav,
    read_local_file,
    record_devices,
    record_wav,
    to_eval_audio,
)
from .chivox import ChivoxMCP, ChivoxMCPError
from .corpus import all_items, corpus_source, stats
from .llm import LLM, last_error as llm_last_error
from .rag import search as rag_search, stats as rag_stats
from .session import (
    STORE,
    SessionError,
    build_report,
    evaluate_entry,
    evaluate_single,
    next_round,
    read_history,
    record_practice,
)

STATIC_DIR = ROOT / "static"
app = FastAPI(title="七年级英语口语定级助手", version="2.1.0")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class SessionBody(BaseModel):
    student: str = "小朋友"
    size: int = 6


class StreamBody(BaseModel):
    core_type: str
    ref_text: str
    sample_rate: Optional[int] = 16000
    audio_type: Optional[str] = "wav"
    channel: Optional[int] = 1


class AnswerBody(BaseModel):
    answer: str = ""


class PracticeAnswerBody(BaseModel):
    answer: str = ""
    item_id: Optional[str] = None
    kind: Optional[str] = None
    text: Optional[str] = None
    session_id: Optional[str] = None


@app.get("/favicon.ico")
def favicon() -> Response:
    """浏览器会自动请求站点图标。没有图标文件时返回 204，避免在日志里刷 404。"""
    icon = STATIC_DIR / "favicon.ico"
    if icon.exists():
        return FileResponse(icon, media_type="image/x-icon")
    return Response(status_code=204)


@app.get("/")
def index() -> FileResponse:
    page = STATIC_DIR / "index.html"
    if not page.exists():
        raise HTTPException(404, "缺少 static/index.html")
    # 强制不缓存，避免浏览器拿着旧页面调试半天
    return FileResponse(page, headers={"Cache-Control": "no-store, must-revalidate"})


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "chivox_mcp_url": config.chivox_mcp_url(),
        "has_chivox_key": bool(config.chivox_api_key()),
        "llm_model": config.llm_model() if config.llm_api_key() else "",
        "has_llm_key": bool(config.llm_api_key()),
        "llm_last_error": llm_last_error(),
        "corpus": stats(),
        "rag": rag_stats(),
        "ffmpeg": has_ffmpeg(),
        "arecord": has_arecord(),
        "cores": tools.core_status(),
    }


@app.get("/api/items")
def list_items(kind: str | None = None, unit: str | None = None) -> dict[str, Any]:
    items = all_items()
    if kind:
        items = [i for i in items if i.get("kind") == kind]
    if unit:
        items = [i for i in items if i.get("unit_id") == unit]
    return {"count": len(items), "source": corpus_source(), "items": items}


@app.get("/api/mcp/tools")
def mcp_tools() -> dict[str, Any]:
    try:
        with ChivoxMCP() as mcp:
            tools = mcp.list_tools()
    except ChivoxMCPError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {
        "count": len(tools),
        "tools": [
            {"name": t.get("name"), "description": (t.get("description") or "")[:200]}
            for t in tools
            if isinstance(t, dict)
        ],
    }


@app.post("/api/mcp/stream-session")
def stream_session(body: StreamBody) -> dict[str, Any]:
    """实时录音：先建流式会话，拿到 session_id 后客户端用 WebSocket 推音频。"""
    try:
        with ChivoxMCP() as mcp:
            result = mcp.create_stream_session(
                core_type=body.core_type,
                ref_text=body.ref_text,
                sample_rate=body.sample_rate,
                audio_type=body.audio_type,
                channel=body.channel,
            )
    except ChivoxMCPError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {"ok": True, "result": result}


@app.post("/api/session")
def create_session(body: SessionBody) -> dict[str, Any]:
    session = STORE.create(student=body.student, size=body.size)
    return {
        "session_id": session["id"],
        "student": session["student"],
        "items": [
            {
                "index": e["index"],
                "id": e["item"].get("id"),
                "group": e.get("group", "word"),
                "kind": e["item"].get("kind"),
                "text": e["item"].get("text"),
                "zh": e["item"].get("zh"),
                "sample": e["item"].get("sample"),
                "unit": e["item"].get("unit_name"),
                "tool": e["tool"],
                "chain": e.get("chain", "word"),
                # 语法题给前端渲染用（answer 不下发，避免直接看到答案）
                "options": e["item"].get("options"),
                "grammar_point": e["item"].get("grammar_point"),
            }
            for e in session["items"]
        ],
    }


def _encode_with_check(raw: bytes, suffix: str) -> str:
    """统一转成评测用音频（16k 单声道 mp3）再 base64。

    实测结论：驰声 MCP 的 audio_base64 通道 **对 wav(PCM) 会卡死超时，对 mp3 正常**
    （同一段真实语音：wav base64 → 90s 超时；mp3 base64 → 0.7s 出分）。
    所以这里一律走 mp3。
    """
    info = probe_wav(raw)
    seconds = float(info.get("seconds") or 0)
    if seconds > MAX_SECONDS * 3:
        raise HTTPException(
            400,
            f"音频过长（{seconds:.0f} 秒）：单题请控制在 {int(MAX_SECONDS)} 秒以内（超出部分会被截断）",
        )
    data, kind = to_eval_audio(raw, suffix)
    if not data:
        raise HTTPException(400, "音频为空或无法解析")
    if kind in ("wav-py", "raw") and not has_ffmpeg():
        raise HTTPException(
            400,
            "本机没有 ffmpeg，无法把音频转成 mp3；请先安装（sudo apt-get install -y ffmpeg）"
            "或直接上传 mp3 文件（驰声 base64 通道对 wav 会超时）",
        )
    return base64.b64encode(data).decode("ascii")


def _audio_source(
    audio: Optional[UploadFile],
    audio_path: Optional[str],
    audio_url: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """三种音频入口 → (base64, url)。"""
    if audio is not None:
        raw = audio.file.read()
        if not raw:
            raise HTTPException(400, "上传的音频是空的")
        suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
        return _encode_with_check(raw, suffix), None
    if audio_path:
        try:
            raw = read_local_file(audio_path)
        except FileNotFoundError as exc:
            raise HTTPException(400, str(exc)) from exc
        suffix = os.path.splitext(audio_path)[1] or ".wav"
        return _encode_with_check(raw, suffix), None
    if audio_url:
        return None, audio_url.strip()
    raise HTTPException(400, "请提供 audio 文件、audio_path 本地路径或 audio_url 之一")


@app.post("/api/session/{session_id}/item/{index}")
async def evaluate_session_item(
    session_id: str,
    index: int,
    audio: Optional[UploadFile] = File(default=None),
    audio_path: Optional[str] = Form(default=None),
    audio_url: Optional[str] = Form(default=None),
) -> dict[str, Any]:
    session = STORE.get(session_id)
    entry = next((e for e in session["items"] if e["index"] == index), None)
    if entry is None:
        raise HTTPException(404, f"会话里没有第 {index} 题")
    b64, url = _audio_source(audio, audio_path, audio_url)
    try:
        result = evaluate_entry(entry, audio_base64=b64, audio_url=url)
    except ChivoxMCPError as exc:
        raise HTTPException(502, str(exc)) from exc
    session["results"] = [r for r in session["results"] if r["index"] != index]
    session["results"].append(result)
    session["results"].sort(key=lambda r: r["index"])
    return {
        "index": index,
        "group": result.get("group", "word"),
        "tool": result["tool"],
        "ref_text": result["ref_text"],
        "overall": result["summary"].get("overall"),
        "dims": result["summary"].get("dims"),
        "multi_dim": result["summary"].get("multi_dim"),
        "words": result["summary"].get("words"),
        "errors": result["summary"].get("errors"),
        "answered": len(session["results"]),
        "total": len(session["items"]),
    }


@app.post("/api/session/{session_id}/item/{index}/answer")
def answer_session_item(session_id: str, index: int, body: AnswerBody) -> dict[str, Any]:
    """语法题（填空 / 选择）作答：本地规则判分，不需要音频。"""
    session = STORE.get(session_id)
    entry = next((e for e in session["items"] if e["index"] == index), None)
    if entry is None:
        raise HTTPException(404, f"会话里没有第 {index} 题")
    if str(entry.get("chain")) != "rule":
        raise HTTPException(400, "这是口语题，请用录音或上传音频提交")
    result = evaluate_entry(entry, answer=body.answer)
    session["results"] = [r for r in session["results"] if r["index"] != index]
    session["results"].append(result)
    session["results"].sort(key=lambda r: r["index"])
    s = result["summary"]
    return {
        "index": index,
        "group": "grammar",
        "tool": "rule",
        "overall": s.get("overall"),
        "rule": s.get("rule"),
        "answered": len(session["results"]),
        "total": len(session["items"]),
    }


@app.post("/api/practice/answer")
def practice_answer(body: PracticeAnswerBody) -> dict[str, Any]:
    """练习页的语法题作答。"""
    try:
        result = evaluate_single(
            item_id=body.item_id or None,
            kind=body.kind,
            text=body.text,
            answer=body.answer,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if body.session_id:
        try:
            record_practice(body.session_id, result)
        except SessionError:
            pass
    s = result["summary"]
    return {
        "tool": result["tool"],
        "group": result.get("group", "grammar"),
        "overall": s.get("overall"),
        "rule": s.get("rule"),
        "recorded": bool(body.session_id),
    }


@app.post("/api/session/{session_id}/next-round")
def session_next_round(session_id: str) -> dict[str, Any]:
    """闭环一期③：用「定级 + 已练」结果重新画像，出下一组练习题。"""
    try:
        return next_round(session_id)
    except SessionError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/session/{session_id}/report")
def session_report(session_id: str) -> dict[str, Any]:
    session = STORE.get(session_id)
    try:
        return build_report(session)
    except SessionError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/evaluate/{item_id}")
async def evaluate_item(
    item_id: str,
    audio: Optional[UploadFile] = File(default=None),
    audio_path: Optional[str] = Form(default=None),
    audio_url: Optional[str] = Form(default=None),
) -> dict[str, Any]:
    """单题练习（推荐列表里的题直接练）。"""
    b64, url = _audio_source(audio, audio_path, audio_url)
    try:
        result = evaluate_single(item_id, audio_base64=b64, audio_url=url)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ChivoxMCPError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {
        "item_id": item_id,
        "tool": result["tool"],
        "ref_text": result["ref_text"],
        "overall": result["summary"].get("overall"),
        "dims": result["summary"].get("dims"),
        "words": result["summary"].get("words"),
        "errors": result["summary"].get("errors"),
    }


@app.get("/api/rag/stats")
def rag_status() -> dict[str, Any]:
    return rag_stats()


@app.get("/api/rag/search")
def rag_query(q: str, top_k: int = 5, book: Optional[str] = None) -> dict[str, Any]:
    """在七年级教材切片里检索：用于找原句、例句、单元出处。"""
    hits = rag_search(q, top_k=top_k, book=book)
    return {"query": q, "count": len(hits), "hits": hits}


@app.post("/api/evaluate/{item_id}/server-record")
def server_record_single(item_id: str, seconds: int = Form(5)) -> dict[str, Any]:
    """单题练习的兜底通道：服务器本机录音。"""
    try:
        raw = record_wav(seconds)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    try:
        result = evaluate_single(item_id, audio_base64=_encode_with_check(raw, ".wav"))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ChivoxMCPError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {
        "item_id": item_id,
        "tool": result["tool"],
        "ref_text": result["ref_text"],
        "overall": result["summary"].get("overall"),
        "dims": result["summary"].get("dims"),
        "words": result["summary"].get("words"),
        "errors": result["summary"].get("errors"),
        "recorded_seconds": seconds,
    }


@app.post("/api/practice/eval")
async def practice_eval(
    kind: str = Form(...),
    text: str = Form(...),
    item_id: Optional[str] = Form(default=None),
    audio: Optional[UploadFile] = File(default=None),
    audio_path: Optional[str] = Form(default=None),
    audio_url: Optional[str] = Form(default=None),
    session_id: Optional[str] = Form(default=None),
) -> dict[str, Any]:
    """练习评测：教材题给 item_id，配对/临时题给 kind + text。

    带 session_id 时结果会回写到会话（闭环一期①），用于后续出下一轮。
    """
    b64, url = _audio_source(audio, audio_path, audio_url)
    try:
        result = evaluate_single(
            item_id=item_id or None, kind=kind, text=text, audio_base64=b64, audio_url=url
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ChivoxMCPError as exc:
        raise HTTPException(502, str(exc)) from exc
    if session_id:
        try:
            record_practice(session_id, result)
        except SessionError:
            pass            # 会话不存在/已过期：只影响闭环，不阻断本次练习
    s = result["summary"]
    return {
        "tool": result["tool"],
        "group": result.get("group", "word"),
        "ref_text": result["ref_text"],
        "overall": s.get("overall"),
        "dims": s.get("dims"),
        "multi_dim": s.get("multi_dim"),
        "words": s.get("words"),
        "errors": s.get("errors"),
        "recorded": bool(session_id),
    }


@app.get("/api/audio/devices")
def audio_devices() -> dict[str, Any]:
    """服务器本机录音设备（arecord -l），浏览器枚举不到麦克风时用来判断。"""
    return {"arecord": has_arecord(), "devices": record_devices()}


@app.post("/api/session/{session_id}/item/{index}/server-record")
def server_record_item(session_id: str, index: int, seconds: int = Form(5)) -> dict[str, Any]:
    """兜底通道：用服务器本机声卡录一段音频再评测。"""
    session = STORE.get(session_id)
    entry = next((e for e in session["items"] if e["index"] == index), None)
    if entry is None:
        raise HTTPException(404, f"会话里没有第 {index} 题")
    try:
        raw = record_wav(seconds)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    try:
        result = evaluate_entry(entry, audio_base64=_encode_with_check(raw, ".wav"))
    except ChivoxMCPError as exc:
        raise HTTPException(502, str(exc)) from exc
    session["results"] = [r for r in session["results"] if r["index"] != index]
    session["results"].append(result)
    session["results"].sort(key=lambda r: r["index"])
    return {
        "index": index,
        "tool": result["tool"],
        "ref_text": result["ref_text"],
        "overall": result["summary"].get("overall"),
        "dims": result["summary"].get("dims"),
        "words": result["summary"].get("words"),
        "errors": result["summary"].get("errors"),
        "answered": len(session["results"]),
        "total": len(session["items"]),
        "recorded_seconds": seconds,
    }


@app.get("/api/history")
def history(limit: int = 50) -> dict[str, Any]:
    return {"rows": read_history(limit)}


@app.get("/api/debug/convert-check")
def convert_check() -> dict[str, Any]:
    """看看本地有没有 ffmpeg（没有也能跑，只是浏览器录音不转码）。"""
    return {"ffmpeg": has_ffmpeg(), "note": "无 ffmpeg 时浏览器 webm 录音会原样上传"}
