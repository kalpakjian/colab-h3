"""驗證產出的 notebook：結構合法 + 內嵌模組與原始檔一致 + Python 語法可過。

    python tools/verify_notebook.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "ComfyUI_H3_Colab.ipynb"
PKG = ROOT / "h3colab"

problems: list[str] = []


def fail(message: str) -> None:
    problems.append(message)
    print(f"  FAIL {message}")


def ok(message: str) -> None:
    print(f"  OK   {message}")


def main() -> int:
    if not NB.exists():
        fail(f"找不到 {NB}")
        return 1
    nb = json.loads(NB.read_text(encoding="utf-8"))

    print("[notebook]")
    if nb.get("nbformat") != 4:
        fail("nbformat 應為 4")
    else:
        ok("nbformat = 4")
    cells = nb.get("cells") or []
    ok(f"{len(cells)} 格")
    for index, cell in enumerate(cells):
        if cell.get("cell_type") not in ("code", "markdown"):
            fail(f"cell {index} 型別不對：{cell.get('cell_type')}")
        source = cell.get("source")
        if not isinstance(source, list) or not all(isinstance(x, str) for x in source):
            fail(f"cell {index} 的 source 必須是字串陣列")

    print("\n[內嵌模組]")
    writefiles = [c for c in cells if "".join(c.get("source", [])).startswith("%%writefile")]
    if len(writefiles) != 8:
        fail(f"應該有 8 個 %%writefile 格，實際 {len(writefiles)}")
    for cell in writefiles:
        text = "".join(cell["source"])
        header, _, body = text.partition("\n")
        name = header.replace("%%writefile h3colab/", "").strip()
        lines = body.splitlines()
        if lines and lines[-1] == "":
            lines = lines[:-1]
        extracted = "\n".join(line[1:] if line.startswith("+") else line for line in lines)
        original = (PKG / name).read_text(encoding="utf-8")
        if extracted.rstrip("\n") != original.rstrip("\n"):
            fail(f"{name} 內嵌內容與原檔不一致")
            continue
        try:
            compile(extracted, name, "exec")
        except SyntaxError as exc:
            fail(f"{name} 語法錯誤：{exc}")
            continue
        ok(f"{name} 內容一致且語法可過（{len(lines)} 行）")

    print("\n[關鍵內容]")
    all_code = "\n".join(
        "".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code"
    )
    for needle in ("detect_gpu", "ensure_comfyui", "download_profile", "start_comfyui",
                   "synth_script", "retime_beats", "MiniMaxH3", "ref_audios",
                   "voice_driven", "upload_audio"):
        if needle in all_code:
            ok(f"含有 {needle}")
        else:
            fail(f"找不到 {needle}")

    print()
    if problems:
        print(f"失敗 {len(problems)} 項")
        return 1
    print("全部通過")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
