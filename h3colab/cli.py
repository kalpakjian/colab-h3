"""命令列入口：doctor / setup / prompt / tts / generate / video。

    python -m h3colab.cli doctor
    python -m h3colab.cli setup --profile l4 --download --store drive
    python -m h3colab.cli prompt --brief "..." --backend llm
    python -m h3colab.cli tts --script script.txt --out out/tts
    python -m h3colab.cli video --script script.txt --profile l4_ref \
        --subject "a Taiwanese news anchor" --scene "a modern studio desk"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import config, payloads, prompt_compiler, tts_voice
from .client import ComfyClient, ComfyError


def _print(text: str) -> None:
    print(text, flush=True)


def _client(base_url: str) -> ComfyClient:
    """建立客戶端（附上 ComfyUI 的 input/ 目錄，參考音訊要直接放進去）。"""
    return ComfyClient(base_url, input_dir=os.path.join(config.COMFY_ROOT, "input"))


# --- doctor -----------------------------------------------------------------

def cmd_doctor(args: argparse.Namespace) -> int:
    from .setup_colab import detect_gpu, model_report, sol_attn_supported

    gpu = detect_gpu()
    _print("== GPU ==")
    _print(json.dumps(gpu, ensure_ascii=False))
    if gpu["available"]:
        _print(f"   sol-attn 可用：{sol_attn_supported(gpu['capability'])}（SM{gpu['capability']}）")
    else:
        _print("   [注意] 偵測不到 CUDA（沒裝 torch 或沒有 GPU）。"
               "下面的 profile 建議不可靠，請手動指定 --profile。")
    profile = config.pick_profile(gpu["vram_gb"], needs_ref=args.ref)
    _print(f"   建議 profile：{profile.key} — {profile.label}")
    if profile.warn:
        _print(f"   注意：{profile.warn}")

    _print("\n== ComfyUI ==")
    root = Path(config.COMFY_ROOT)
    _print(f"   {root} {'存在' if (root / 'main.py').exists() else '尚未安裝'}")
    for name in (
        "ComfyUI-MiniMax-H3-Turbo",
        "ComfyUI-sol-attn",
        "ComfyUI_MiniMaxH3_Director",
    ):
        path = root / "custom_nodes" / name
        _print(f"   {name}: {'已安裝' if path.exists() else '未安裝'}")

    _print("\n== 權重 ==")
    for row in model_report(profile):
        mark = "OK " if row["present"] else "缺 "
        _print(f"   {mark}{row['file']}  ({row['size_gb']} GB)")

    _print("\n== TTS ==")
    _print("   " + tts_voice.install_hint())
    return 0


# --- setup ------------------------------------------------------------------

def cmd_setup(args: argparse.Namespace) -> int:
    from .setup_colab import (
        download_profile, ensure_comfyui, ensure_nodes, log,
    )

    profile = config.PROFILES[args.profile]
    log(f"profile：{profile.label}")
    if profile.warn:
        log(f"注意：{profile.warn}")

    ensure_comfyui(config.COMFY_ROOT)
    ensure_nodes(config.COMFY_ROOT, include_optional=not args.minimal)

    if args.download:
        download_profile(profile, store=args.store, materialize=not args.no_copy)
    else:
        log("略過權重下載（加 --download 才下）")
    return 0


# --- prompt -----------------------------------------------------------------

def _brief_from_args(args: argparse.Namespace) -> prompt_compiler.Brief:
    beats = list(args.beat or [])
    if args.script:
        script_path = Path(args.script)
        text = script_path.read_text(encoding="utf-8")
        beats.extend([line.strip() for line in text.splitlines() if line.strip()])
    return prompt_compiler.Brief(
        subject=args.subject or "",
        scene=args.scene or "",
        beats=beats,
        look=args.look or "",
        camera=args.camera or "",
        audio=args.audio,
        voice_driven=args.voice_driven,
        duration=args.duration,
        template=args.template,
        identity_lock=args.identity or "",
    )


def cmd_prompt(args: argparse.Namespace) -> int:
    brief = _brief_from_args(args)
    llm_config = prompt_compiler.LLMConfig.from_env() if args.backend == "llm" else None
    text = prompt_compiler.enhance(
        brief,
        backend=args.backend,
        llm_config=llm_config,
        base_url=args.base_url,
        task_type=args.task,
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        _print(f"已寫入 {args.out}")
    _print(text)
    return 0


# --- tts --------------------------------------------------------------------

def cmd_tts(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.script:
        lines = [
            line.strip()
            for line in Path(args.script).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        lines = [args.text or ""]
    lines = [line for line in lines if line]
    if not lines:
        _print("沒有台詞")
        return 1
    result = tts_voice.synth_script(
        lines, out_dir, voice=args.voice, rate=args.rate, backend=args.backend
    )
    total = sum(item.duration for item in result)
    manifest = {
        "voice": args.voice,
        "total_seconds": round(total, 3),
        "lines": [
            {"text": item.text, "path": item.path, "duration": item.duration}
            for item in result
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


# --- generate ---------------------------------------------------------------

def _build_spec(args: argparse.Namespace, prompt: str, profile: config.Profile):
    overrides = {}
    for key in ("width", "height", "length", "steps", "seed", "task"):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    spec = payloads.spec_from_profile(profile, prompt, **overrides)
    if not getattr(args, "no_sol_attn", False):
        spec.sol_attn = profile.sol_attn
    return spec


def cmd_generate(args: argparse.Namespace) -> int:
    profile = config.PROFILES[args.profile]
    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
    if not prompt:
        _print("沒有提示詞（--prompt 或 --prompt-file）")
        return 1

    client = _client(args.base_url)
    available = client.available_nodes()
    spec = _build_spec(args, prompt, profile)

    if args.ref_image:
        spec.ref_images = [client.upload_image(path) for path in args.ref_image]
    if args.ref_audio:
        spec.ref_audios = [client.upload_audio(path) for path in args.ref_audio]
    if spec.task == "r2v" and not spec.ref_images and not spec.ref_audios:
        _print("提示：r2v 建議至少給一個參考圖（--ref-image）或參考音訊（--ref-audio）")

    spec.filename_prefix = args.prefix or f"video/{profile.key}"
    graph, notes = payloads.build_graph(spec, available)
    for note in notes:
        _print(f"[注意] {note}")

    _print(f"提交：{profile.label} / {spec.width}x{spec.height} / "
           f"{spec.resolved_length()} 幀 / {spec.steps} 步 / {spec.mode}")
    try:
        entry, saved = client.run(graph, timeout=args.timeout, dest_dir=args.out)
    except ComfyError as exc:
        _print(f"[失敗] {exc}")
        return 1
    for path in saved:
        _print(f"完成：{path}")
    if not saved:
        _print("完成，但沒抓到檔案（看 ComfyUI output/ 目錄）")
    return 0


# --- video（端到端：台詞 → TTS → 提示組 → 聲音驅動生成） ---------------------

def cmd_video(args: argparse.Namespace) -> int:
    profile = config.PROFILES[args.profile]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        line.strip()
        for line in Path(args.script).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        _print("台詞檔是空的")
        return 1

    # 1) TTS：逐句合成並量真實長度
    script_lines = tts_voice.synth_script(
        lines, out_dir / "tts", voice=args.voice, rate=args.rate, backend=args.tts_backend
    )
    durations = [item.duration for item in script_lines]
    total = sum(durations)
    _print(f"台詞 {len(script_lines)} 句，總長 {total:.2f}s")

    # 2) 時間碼：依真實音軌長度切
    #    分鏡描述應該是「英文視覺描述」；台詞是中文，走 TTS → <Audio 1>。
    #    沒給 --beat 就用台詞頂替（LLM 後端會翻譯；純規則後端會出中文）。
    visual = list(args.beat or [])
    if not visual:
        visual = [item.text for item in script_lines]
        if args.backend == "rule" and prompt_compiler.has_cjk(" ".join(visual)):
            _print("[注意] 沒給 --beat，分鏡會直接用中文台詞。"
                   "H3 用 Qwen3-VL 能讀中文，但官方測試路徑是英文；"
                   "建議補 --beat（英文視覺描述）或改用 --backend llm。")
    if len(visual) != len(script_lines):
        _print(f"[注意] --beat 有 {len(visual)} 個，時間碼只有 {len(script_lines)} 段；"
               "多出來的節拍會對不上語速。")
        visual = (visual + [item.text for item in script_lines])[:len(script_lines)]
    beats = prompt_compiler.distribute_beats(visual, total)
    timed = prompt_compiler.retime_beats(beats, durations)

    # 3) 提示組
    brief = prompt_compiler.Brief(
        subject=args.subject or "",
        scene=args.scene or "",
        beats=visual,
        look=args.look or "",
        camera=args.camera or "",
        audio=args.audio or "clear spoken voice",
        voice_driven=True,
        duration=total,
        timed_beats=timed,
    )
    llm_config = prompt_compiler.LLMConfig.from_env() if args.backend == "llm" else None
    prompt = prompt_compiler.enhance(
        brief, backend=args.backend, llm_config=llm_config,
        base_url=args.base_url, task_type="r2v",
    )
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    _print("\n--- 提示組 ---\n" + prompt + "\n")

    # 4) 合併音軌 + 放進 input/
    merged = tts_voice.concat_wavs([item.path for item in script_lines], out_dir / "voice.wav")
    client = _client(args.base_url)
    audio_name = client.upload_audio(merged)

    # 5) 幀數要對上音軌長度
    frames = config.snap_length(int(total * 24) + 24)
    spec = payloads.spec_from_profile(
        profile, prompt, task="r2v", length=frames, ref_audios=[audio_name],
    )
    if args.ref_image:
        spec.ref_images = [client.upload_image(path) for path in args.ref_image]
    spec.filename_prefix = args.prefix or f"video/{profile.key}_avatar"

    available = client.available_nodes()
    graph, notes = payloads.build_graph(spec, available)
    for note in notes:
        _print(f"[注意] {note}")

    _print(f"提交：{profile.label} / {spec.width}x{spec.height} / {frames} 幀 / "
           f"{spec.steps} 步 / r2v + <Audio 1>")
    try:
        entry, saved = client.run(graph, timeout=args.timeout, dest_dir=out_dir)
    except ComfyError as exc:
        _print(f"[失敗] {exc}")
        return 1
    for path in saved:
        _print(f"完成：{path}")
    return 0


# --- 入口 -------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="h3colab", description="Colab ⇄ ComfyUI ⇄ MiniMax H3")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="檢查 GPU / ComfyUI / 節點 / 權重")
    p.add_argument("--ref", action="store_true", help="以 r2v（聲音驅動）角度檢查")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("setup", help="安裝 ComfyUI + 節點 + 權重")
    p.add_argument("--profile", default="l4", choices=sorted(config.PROFILES))
    p.add_argument("--store", default="drive", choices=["drive", "local"])
    p.add_argument("--download", action="store_true", help="連權重一起下（預設不下）")
    p.add_argument("--no-copy", action="store_true", help="不要複製到執行目錄")
    p.add_argument("--minimal", action="store_true", help="只裝必要的節點")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("prompt", help="把 brief 編譯成 H3 提示組")
    p.add_argument("--brief", default="")
    p.add_argument("--subject", default="")
    p.add_argument("--scene", default="")
    p.add_argument("--beat", action="append")
    p.add_argument("--script")
    p.add_argument("--look", default="")
    p.add_argument("--camera", default="")
    p.add_argument("--audio", default=None)
    p.add_argument("--voice-driven", action="store_true")
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--template", default="storyboard", choices=["storyboard", "identity_lock"])
    p.add_argument("--identity", default="")
    p.add_argument("--backend", default="rule", choices=["rule", "llm", "director"])
    p.add_argument("--base-url", default="http://127.0.0.1:8188")
    p.add_argument("--task", default="t2v")
    p.add_argument("--out")
    p.set_defaults(func=cmd_prompt)

    p = sub.add_parser("tts", help="合成中文語音並量真實長度")
    p.add_argument("--text")
    p.add_argument("--script")
    p.add_argument("--out", default="out/tts")
    p.add_argument("--voice", default=tts_voice.DEFAULT_VOICE)
    p.add_argument("--rate", default="+0%")
    p.add_argument("--backend", default="edge", choices=["edge", "gtts"])
    p.set_defaults(func=cmd_tts)

    p = sub.add_parser("generate", help="用既有提示詞生成")
    p.add_argument("--profile", default="l4", choices=sorted(config.PROFILES))
    p.add_argument("--prompt")
    p.add_argument("--prompt-file")
    p.add_argument("--task", choices=["t2v", "i2v", "fl2v", "r2v"])
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--length", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--ref-image", action="append")
    p.add_argument("--ref-audio", action="append")
    p.add_argument("--no-sol-attn", action="store_true")
    p.add_argument("--prefix")
    p.add_argument("--out", default="out")
    p.add_argument("--base-url", default="http://127.0.0.1:8188")
    p.add_argument("--timeout", type=int, default=3600)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("video", help="端到端：台詞 → TTS → 提示組 → 聲音驅動數字人")
    p.add_argument("--script", required=True)
    p.add_argument("--profile", default="l4_ref", choices=sorted(config.PROFILES))
    p.add_argument("--subject", default="")
    p.add_argument("--scene", default="")
    p.add_argument("--look", default="")
    p.add_argument("--camera", default="")
    p.add_argument("--audio", default="")
    p.add_argument("--beat", action="append",
                   help="英文視覺描述（與台詞一句對一句）；不給就用台詞頂替")
    p.add_argument("--voice", default=tts_voice.DEFAULT_VOICE)
    p.add_argument("--rate", default="+0%")
    p.add_argument("--tts-backend", default="edge", choices=["edge", "gtts"])
    p.add_argument("--backend", default="rule", choices=["rule", "llm", "director"])
    p.add_argument("--ref-image", action="append")
    p.add_argument("--prefix")
    p.add_argument("--out", default="out/avatar")
    p.add_argument("--base-url", default="http://127.0.0.1:8188")
    p.add_argument("--timeout", type=int, default=3600)
    p.set_defaults(func=cmd_video)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
