"""音频输入处理：浏览器录音（webm/ogg）、上传文件、本地路径，统一成 16k 单声道 wav。"""

from __future__ import annotations

import array
import io
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

# 单题音频上限（秒）：超过就截断，避免驰声内核 core is timeout
# 七年级单题朗读一般 5-15 秒；长录音（含大段静音）最容易把评测内核拖超时
try:
    MAX_SECONDS = float(os.getenv("AUDIO_MAX_SECONDS") or 30)
except ValueError:
    MAX_SECONDS = 30.0


def _ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def probe_wav(raw: bytes) -> dict:
    """读 wav 头：采样率 / 声道 / 位深 / 时长。非 wav 返回 {}。"""
    try:
        with wave.open(io.BytesIO(raw), "rb") as w:
            rate = w.getframerate()
            frames = w.getnframes()
            return {
                "rate": rate,
                "channels": w.getnchannels(),
                "sample_width": w.getsampwidth(),
                "seconds": round(frames / rate, 2) if rate else 0.0,
            }
    except Exception:
        return {}


def normalize_wav_py(raw: bytes, target_rate: int = 16000) -> bytes | None:
    """纯 Python：16bit wav → 16k 单声道（没有 ffmpeg 时的兜底）。"""
    try:
        with wave.open(io.BytesIO(raw), "rb") as w:
            ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            data = w.readframes(n)
    except Exception:
        return None
    if sw != 2 or ch not in (1, 2) or sr <= 0:
        return None
    samples = array.array("h")
    samples.frombytes(data)
    if sys.byteorder == "big":
        samples.byteswap()
    if ch == 2:
        samples = array.array(
            "h", ((samples[i] + samples[i + 1]) >> 1 for i in range(0, len(samples) - 1, 2))
        )
    limit = int(sr * MAX_SECONDS)
    if len(samples) > limit:
        samples = samples[:limit]
    if sr != target_rate and samples:
        ratio = target_rate / sr
        out_len = int(len(samples) * ratio)
        samples = array.array(
            "h", (samples[min(int(i / ratio), len(samples) - 1)] for i in range(out_len))
        )
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(target_rate)
        out.writeframes(samples.tobytes())
    return buf.getvalue()


def to_wav16k(raw: bytes, suffix: str = ".webm") -> tuple[bytes, str]:
    """统一成 16kHz / 单声道 / 16bit wav。"""
    exe = _ffmpeg()
    if not exe:
        fixed = normalize_wav_py(raw)
        if fixed:
            return fixed, "wav16k-py"
        return raw, "raw"
    if suffix.lower() not in (".webm", ".mp3", ".m4a", ".ogg", ".mp4", ".wav", ".flac", ".aac"):
        suffix = ".bin"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"in{suffix}"
        dst = Path(tmp) / "out.wav"
        src.write_bytes(raw)
        try:
            subprocess.run(
                [
                    exe, "-y", "-i", str(src),
                    "-t", str(MAX_SECONDS),          # 超长音频直接截断，防驰声内核超时
                    "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", "-f", "wav", str(dst),
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return raw, "raw"
        if not dst.exists():
            return raw, "raw"
        return dst.read_bytes(), "wav16k"


SUPPORTED_IN = (".webm", ".mp3", ".m4a", ".ogg", ".mp4", ".wav", ".flac", ".aac", ".opus")


def _ffmpeg_convert(raw: bytes, suffix: str, fmt: str = "mp3") -> bytes | None:
    """用 ffmpeg 统一成 16k 单声道；fmt=mp3|wav。失败返回 None。"""
    exe = _ffmpeg()
    if not exe:
        return None
    suffix = suffix if suffix.lower() in SUPPORTED_IN else ".bin"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"in{suffix}"
        dst = Path(tmp) / f"out.{fmt}"
        src.write_bytes(raw)
        cmd = [exe, "-y", "-i", str(src), "-t", str(MAX_SECONDS), "-ac", "1", "-ar", "16000"]
        if fmt == "mp3":
            cmd += ["-b:a", "32k", "-f", "mp3", str(dst)]
        else:
            cmd += ["-sample_fmt", "s16", "-f", "wav", str(dst)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=int(MAX_SECONDS) + 30)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None
        if not dst.exists():
            return None
        return dst.read_bytes()


def to_eval_audio(raw: bytes, suffix: str = ".webm") -> tuple[bytes, str]:
    """评测用的音频字节。

    重要：驰声 MCP 的 `audio_base64` 通道对 **wav(PCM) 会卡死超时**，对 **mp3 正常**
    （本机实测：同一段真实语音，wav base64 → 90s 超时；mp3 base64 → 0.7s 出分）。
    所以一律优先转 mp3。

    返回 (bytes, kind)，kind ∈ {mp3, mp3-raw, wav-ffmpeg, wav-py, raw}
    """
    suffix = (suffix or "").lower()
    if has_ffmpeg():
        data = _ffmpeg_convert(raw, suffix, "mp3")
        if data:
            return data, "mp3"
        data = _ffmpeg_convert(raw, suffix, "wav")
        if data:
            return data, "wav-ffmpeg"
        return raw, "raw"
    if suffix in (".mp3", ".m4a", ".aac"):
        return raw, "mp3-raw"          # 本身就是压缩格式，可原样上传
    fixed = normalize_wav_py(raw)
    if fixed:
        return fixed, "wav-py"
    return raw, "raw"


def read_local_file(path: str) -> bytes:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"音频文件不存在: {path}")
    return file_path.read_bytes()


def has_ffmpeg() -> bool:
    return _ffmpeg() is not None


def _arecord() -> str | None:
    return shutil.which("arecord")


def has_arecord() -> bool:
    """服务器本机能否直接录音（浏览器拿不到麦克风时的兜底通道）。"""
    return _arecord() is not None


def record_devices() -> str:
    exe = _arecord()
    if not exe:
        return ""
    try:
        out = subprocess.run([exe, "-l"], capture_output=True, timeout=10)
        return (out.stdout or out.stderr).decode("utf-8", "ignore").strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def record_wav(seconds: int = 5, sample_rate: int = 16000) -> bytes:
    """用服务器本机的声卡录一段 16k 单声道 wav。"""
    exe = _arecord()
    if not exe:
        raise RuntimeError("服务器上没有 arecord，无法本机录音")
    seconds = max(1, min(int(seconds), 30))
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "rec.wav"
        cmd = [
            exe, "-q", "-d", str(seconds), "-f", "S16_LE",
            "-r", str(sample_rate), "-c", "1", "-t", "wav", str(dst),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=seconds + 15)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("录音超时") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or b"").decode("utf-8", "ignore").strip()
            raise RuntimeError(f"录音失败：{detail or '声卡不可用'}") from exc
        if not dst.exists():
            raise RuntimeError("录音失败：没有生成音频")
        return dst.read_bytes()
