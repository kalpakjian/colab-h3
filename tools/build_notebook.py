"""由 h3colab/ 產生一份自帶程式碼的 Colab notebook。

    python tools/build_notebook.py

產物：ComfyUI_H3_Colab.ipynb（每個模組以 %%writefile 內嵌，Colab 開箱即跑，
不需要先把檔案推上 GitHub）。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "h3colab"
OUT = ROOT / "ComfyUI_H3_Colab.ipynb"

MODULES = [
    "config.py",
    "prompt_compiler.py",
    "payloads.py",
    "client.py",
    "tts_voice.py",
    "setup_colab.py",
    "cli.py",
    "__init__.py",
]


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def writefile_cells() -> list[dict]:
    cells: list[dict] = []
    for name in MODULES:
        source = (PKG / name).read_text(encoding="utf-8")
        body = "\n".join(f"+{line}" for line in source.splitlines())
        cells.append(code(f"%%writefile h3colab/{name}\n{body}\n"))
    return cells


def build() -> dict:
    cells: list[dict] = []

    cells.append(md("""
# ComfyUI × MiniMax H3 × Colab

一條龍：裝 ComfyUI → 備齊權重 → 啟動 → **把一句話編譯成正確的 H3 提示組** → TTS → 聲音驅動生成。

| 這份 notebook 解決什麼 | 怎麼解決 |
|---|---|
| 不知道該下哪些模型 | `h3colab config` 內建實查過的檔名與大小（官方 H3 / Turbo LoRA / PRISM） |
| 24GB 卡塞不下 | 依 GPU 自動選 profile：L4 用 `w6a8` UNET + 4-bit CLIP；A100 用 `int8` |
| 提示寫不好，AI 亂講話 | `prompt_compiler` 產出官方格式：風格句 + 時間碼 Storyboard + Audio 行 + 禁止項 |
| 沒有 TTS，不算聲音驅動 | Edge TTS 產出 32kHz wav，並用**真實音軌長度**回填時間碼，嘴型才對得上 |
| 掛著掛著就燒光算力 | 用完中斷 runtime；權重放 Drive 當倉庫 |

**執行順序**：下面每格依序按 Shift+Enter。權重下載最慢（30GB 上下），建議用 Google AI Pro 的額度跑。
"""))

    cells.append(code("""
# == Cell 1：環境檢查 + 掛載 Drive ==
import os, sys, subprocess, json

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROFILE = "l4"          # l4 / l4_ref / a100 / a100_ref / t4
STORE = "drive"         # drive = 雲端硬碟當倉庫（可重複用）；local = 直落本機（單次最快）
VOICE_DRIVEN = True     # 要做聲音驅動數字人 → 額外下載 ref2va 權重（PRISM / ref2va）

try:
    from google.colab import drive
    drive.mount("/content/drive")
    print("Drive 已掛載")
except Exception as exc:
    print("掛不到 Drive（不在 Colab？）：", exc)

!pip -q install -U huggingface_hub requests
!apt-get -qq update && apt-get -qq install -y ffmpeg
!pip -q install edge-tts

import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU:", props.name, round(props.total_memory / 1024**3, 1), "GB",
          "SM" + f"{props.major}{props.minor}")
"""))

    cells.append(md("""
## Cell 2：寫入 h3colab 套件

下面幾格會把 `h3colab/` 整包寫進 runtime（提示編譯器、TTS、payload 建構、ComfyUI 客戶端、佈署腳本）。
"""))

    cells.extend(writefile_cells())

    cells.append(code("""
# == Cell 3：安裝 ComfyUI + 節點 + 權重 ==
import sys
sys.path.insert(0, "/content")
from h3colab import config
from h3colab.setup_colab import (
    detect_gpu, ensure_comfyui, ensure_nodes, download_profile,
)

gpu = detect_gpu()
print("GPU:", gpu)
profile = config.pick_profile(gpu["vram_gb"], needs_ref=(PROFILE.endswith("_ref")))
if profile.key != PROFILE:
    print(f"提醒：偵測到 {gpu['vram_gb']}GB，建議用 profile={profile.key}（{profile.label}）")
print("採用：", profile.label)
if profile.warn:
    print("注意：", profile.warn)

ensure_comfyui(config.COMFY_ROOT)
ensure_nodes(config.COMFY_ROOT)
download_profile(profile, store=STORE, materialize=True)

# 聲音驅動走的是 ref2va（PRISM / ref2va），跟 t2v/i2v 的 fl2va 是不同權重，
# 所以要另外抓一份，不然 Cell 7 會找不到模型。
if VOICE_DRIVEN:
    ref_profile = config.pick_profile(gpu["vram_gb"], needs_ref=True)
    print("另外抓聲音驅動用的 ref2va 權重：", ref_profile.label)
    download_profile(ref_profile, store=STORE, materialize=True)
"""))

    cells.append(code("""
# == Cell 4：啟動 ComfyUI ==
import sys, time
sys.path.insert(0, "/content")
from h3colab import config
from h3colab.setup_colab import start_comfyui, wait_ready

proc = start_comfyui(
    config.COMFY_ROOT,
    lowvram=(gpu["vram_gb"] < 28),
    extra_args=["--use-ck-attention"],
)
ok = wait_ready(timeout=240)
print("ComfyUI ready:", ok)
if not ok:
    log = f"{config.COMFY_ROOT}/logs/comfyui_colab.log"
    print(open(log, encoding="utf-8", errors="ignore").read()[-3000:])
"""))

    cells.append(code("""
# == Cell 5：提示強化（把簡單 brief 變成正確的 H3 提示組） ==
import sys
sys.path.insert(0, "/content")
from h3colab.prompt_compiler import Brief, compile_prompt, enhance

brief = Brief(
    subject="a Taiwanese female news anchor in a navy blazer",
    scene="a modern broadcast studio, dark background, glowing desk",
    look="50mm lens, shallow depth of field, soft key light from the left",
    beats=["looks straight into the lens and opens her mouth to speak",
           "raises one hand and points at the screen behind her"],
    camera="locked tripod, medium shot",
    duration=5.0,
    audio="clear female Mandarin voice",
)
prompt = compile_prompt(brief)
print(prompt)

# 想用 LLM 做強化（中文 brief 也可以）：
#   os.environ["GOOGLE_API_KEY"] = "..."
#   prompt = enhance(brief, backend="llm")
"""))

    cells.append(code("""
# == Cell 6：TTS（繁中）+ 用真實音軌長度回推時間碼 ==
import sys
sys.path.insert(0, "/content")
from h3colab.tts_voice import synth_script, concat_wavs
from h3colab.prompt_compiler import Brief, distribute_beats, retime_beats, compile_prompt

script = [
    "各位觀眾晚安，今天我們要聊聊生成式AI。",
    "這支影片完全是在雲端算力上生出來的。",
]
lines = synth_script(script, "/content/out/tts", voice="zh-TW-HsiaoChenNeural")
for line in lines:
    print(f"{line.duration:5.2f}s  {line.text}")

total = sum(line.duration for line in lines)
beats = distribute_beats([line.text for line in lines], total)
timed = retime_beats(beats, [line.duration for line in lines])
print("時間碼：", [b.render() for b in timed])

brief = Brief(
    subject="a Taiwanese female news anchor in a navy blazer",
    scene="a modern broadcast studio, dark background, glowing desk",
    beats=[line.text for line in lines],
    look="50mm lens, shallow depth of field, soft key light from the left",
    camera="locked tripod, medium shot",
    audio="clear female Mandarin voice",
    voice_driven=True,
    duration=total,
    timed_beats=timed,
)
voice_prompt = compile_prompt(brief)
print(voice_prompt)
import os
os.makedirs("/content/out", exist_ok=True)
open("/content/out/prompt.txt", "w", encoding="utf-8").write(voice_prompt)
concat_wavs([line.path for line in lines], "/content/out/voice.wav")
"""))

    cells.append(code("""
# == Cell 7：生成（r2v + <Audio 1> = 聲音驅動） ==
import sys
sys.path.insert(0, "/content")
from h3colab import config, payloads
from h3colab.client import ComfyClient

client = ComfyClient("http://127.0.0.1:8188", input_dir=f"{config.COMFY_ROOT}/input")
available = client.available_nodes()

# 聲音驅動要走 ref2va UNET（PRISM 就是 ref2va merge）
profile = config.PROFILES["l4_ref"]
frames = config.snap_length(int(total * 24) + 24)
spec = payloads.spec_from_profile(profile, voice_prompt, task="r2v", length=frames, seed=42)
spec.ref_audios = [client.upload_audio("/content/out/voice.wav")]
# 有角色參考圖就掛上（identity 會穩很多）：
# spec.ref_images = [client.upload_image("/content/out/face.png")]
spec.filename_prefix = "video/colab_avatar"

graph, notes = payloads.build_graph(spec, available)
for note in notes:
    print("[注意]", note)
print(f"提交 {spec.width}x{spec.height} / {spec.resolved_length()} 幀 / {spec.steps} 步 / {spec.mode}")

entry, saved = client.run(graph, timeout=3600, dest_dir="/content/out")
for path in saved:
    print("完成：", path)
"""))

    cells.append(code("""
# == Cell 8：存檔（Drive 會活著，/content 不會） ==
import os, shutil, glob

dest = "/content/drive/MyDrive/h3_out"
os.makedirs(dest, exist_ok=True)
for path in glob.glob("/content/out/**/*.mp4", recursive=True):
    shutil.copy(path, dest)
    print("已存：", os.path.join(dest, os.path.basename(path)))
"""))

    cells.append(md("""
## 算力與額度

* **L4（24GB）**：864×480（0.4MP）、124 幀（約 5 秒）、Turbo 8 步 → 大概 5 分鐘內出片。
* **A100（40GB）**：1344×768（1.0MP）、124 幀、官方 25 步 → 品質檔。
* **T4（16GB）**：SM75，sol-attn 不支援、H3 走 bf16 很慢 —— 不建議，省下點數選 L4。
* **sol-attn**：L4 = SM89 官方支援；A100 = SM80 **不在**支援清單（SM86/89/90/100/120/121），
  會自動 dense fallback，功能正常但沒有加速。
* 用完記得中斷 runtime，別讓 VM 掛著吃點數。
"""))

    return {
        "cells": cells,
        "metadata": {
            "colab": {"name": "ComfyUI_H3_Colab.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    notebook = build()
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已寫入 {OUT}（{len(notebook['cells'])} 格）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
