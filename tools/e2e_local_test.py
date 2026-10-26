"""端到端實測（需要本機 ComfyUI 在 127.0.0.1:8188 跑起來）。

流程：真實 Edge TTS → 用真實音軌長度回推時間碼 → 編出提示組 → 上傳音軌 →
r2v + <Audio 1> → 生成 → ffprobe 驗片。

    python tools/e2e_local_test.py [--task r2v] [--frames 124] [--steps 6]

這支腳本用**本機 D:\\AI\\ComfyUI 實際有的權重**（不是 PRISM，PRISM 要另外下），
目的是驗證「ref_audio_0 這種 autogrow 音訊槽在 API 格式下真的能動」。
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from h3colab import config, payloads  # noqa: E402
from h3colab.client import ComfyClient, ComfyError  # noqa: E402
from h3colab.prompt_compiler import (  # noqa: E402
    Brief, compile_prompt, distribute_beats, retime_beats,
)
from h3colab.tts_voice import concat_wavs, synth_script  # noqa: E402

LOCAL_MODELS = {
    "unet": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "clip": "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
    "video_vae": "minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
    "turbo_lora": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
}

SCRIPT = [
    "各位觀眾晚安，今天我們要聊聊生成式AI。",
    "這支影片完全是在雲端算力上生出來的。",
]

# 分鏡描述用英文（視覺），台詞用中文（走 TTS → <Audio 1>）—— 這是正確分工：
# H3 的提示詞負責「畫面長什麼樣」，聲音由參考音訊帶。
VISUAL_BEATS = [
    "the presenter looks straight into the lens and opens her mouth to speak",
    "she raises one hand and points toward the screen behind her",
]


def ffprobe(path: str) -> str:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return "(no ffprobe)"
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries",
         "format=duration,size:stream=codec_type,width,height,avg_frame_rate",
         "-of", "json", path],
        capture_output=True, text=True,
    )
    return result.stdout.strip() or result.stderr.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="r2v", choices=["r2v", "t2v"])
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--mode", default="turbo", choices=["turbo", "official"])
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "_tmp_e2e"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8188")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("== 1. TTS ==")
    lines = synth_script(SCRIPT, out_dir / "tts", voice="zh-TW-HsiaoChenNeural")
    for line in lines:
        print(f"  {line.duration:6.3f}s  {line.text}")
    total = sum(line.duration for line in lines)
    print(f"  總長 {total:.3f}s")

    print("\n== 2. 時間碼（依真實音軌長度） ==")
    beats = distribute_beats(VISUAL_BEATS, total)
    timed = retime_beats(beats, [line.duration for line in lines])
    for beat in timed:
        print("  " + beat.render())

    print("\n== 3. 提示組 ==")
    brief = Brief(
        subject="a Taiwanese female presenter in a navy blazer",
        scene="a modern broadcast studio, dark background, glowing desk",
        look="50mm lens, shallow depth of field, soft key light from the left",
        beats=list(VISUAL_BEATS),
        camera="locked tripod, medium shot",
        audio="clear female Mandarin voice",
        voice_driven=True,
        duration=total,
        timed_beats=timed,
    )
    prompt = compile_prompt(brief)
    print(prompt)
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    print("\n== 4. 上傳音軌 + 建圖 ==")
    merged = concat_wavs([line.path for line in lines], out_dir / "voice.wav")
    client = ComfyClient(
        args.base_url,
        input_dir=str(Path(config.COMFY_ROOT) / "input"),
    )
    available = client.available_nodes()
    audio_name = client.upload_audio(merged)
    print(f"  上傳：{audio_name}")

    spec = payloads.GenSpec(
        task=args.task,
        prompt=prompt,
        width=864, height=480, length=args.frames,
        steps=args.steps, mode=args.mode,
        unet=LOCAL_MODELS["unet"], clip=LOCAL_MODELS["clip"],
        video_vae=LOCAL_MODELS["video_vae"], audio_vae=LOCAL_MODELS["audio_vae"],
        turbo_lora=LOCAL_MODELS["turbo_lora"] if args.mode == "turbo" else None,
        turbo_lowvram=False,
        sol_attn=False, ffn_chunks=1,
        ref_audios=[audio_name] if args.task == "r2v" else [],
        filename_prefix="video/h3colab_e2e",
    )
    graph, notes = payloads.build_graph(spec, available)
    for note in notes:
        print("  [注意]", note)
    print(json.dumps(graph, ensure_ascii=False, indent=1)[:1200])

    print("\n== 5. 執行 ==")
    start = time.time()
    try:
        entry, saved = client.run(graph, timeout=3600, dest_dir=out_dir)
    except ComfyError as exc:
        print(f"  [失敗] {exc}")
        return 1
    print(f"  花費 {time.time() - start:.1f}s")
    for path in saved:
        print(f"  完成：{path}")
        print("  ffprobe:", ffprobe(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
