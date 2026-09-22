"""demo2：七年级英语口语定级评测助手。"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

__all__ = ["ROOT", "DATA_DIR"]
