"""Runs on: M4 (CPU only). Makes src/ importable without installing the package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
