"""推上去之前掃一遍有沒有把金鑰／token 寫進去。

    python tools/scan_secrets.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PATTERNS = {
    "github token": r"gho_[A-Za-z0-9]{20,}",
    "openai key": r"sk-[A-Za-z0-9]{20,}",
    "google key": r"AIza[0-9A-Za-z_-]{25,}",
    "aws key": r"AKIA[0-9A-Z]{16}",
    "assigned secret": (
        r"(?i)(api[_-]?key|api_key|token|secret|password)"
        r"\s*[=:]\s*[\"'][^\"'\s]{12,}[\"']"
    ),
}

TEXT_SUFFIXES = {".py", ".md", ".ipynb", ".txt", ".json", ".cfg", ".toml", ".yaml", ".yml"}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    files = subprocess.run(
        ["git", "-C", str(root), "ls-files"], capture_output=True, text=True
    ).stdout.split()
    hits: list[tuple[str, str, str]] = []
    for name in files:
        path = root / name
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in PATTERNS.items():
            for match in re.finditer(pattern, text):
                hits.append((name, label, match.group(0)[:60]))

    print(f"scanned {len(files)} tracked files")
    if not hits:
        print("no secrets found")
        return 0
    print("POTENTIAL SECRETS:")
    for name, label, snippet in hits:
        print(f"  {name}: [{label}] {snippet}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
