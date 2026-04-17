"""Test path setup for local sibling country packages."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MICROPLEX_US_SRC = ROOT_DIR.parent / "microplex-us" / "src"

if MICROPLEX_US_SRC.exists() and str(MICROPLEX_US_SRC) not in sys.path:
    sys.path.insert(0, str(MICROPLEX_US_SRC))
