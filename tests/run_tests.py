"""h3colab 的自我檢查（不依賴 pytest，直接跑）。

    python tests/run_tests.py

會自動嘗試連本機 ComfyUI（127.0.0.1:8188）；連不上就跳過「活體驗證」那組，
其餘離線測試照跑。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from h3colab import config, payloads, prompt_compiler  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  OK   {name}")
    else:
        FAILED.append(f"{name} :: {detail}")
        print(f"  FAIL {name} :: {detail}")


# --- config -----------------------------------------------------------------

def test_config() -> None:
    print("\n[config]")
    check("snap_length(124)==124", config.snap_length(124) == 124)
    check("snap_length(125)==141", config.snap_length(125) == 141)
    check("snap_length(5)==5", config.snap_length(5) == 5)
    check("snap_length(1)==5", config.snap_length(1) == 5)
    for key, profile in config.PROFILES.items():
        models = profile.models
        check(f"{key} 的權重都在 ALL_MODELS 裡",
              all(model in config.ALL_MODELS.values() for model in models))
        check(f"{key} 的解析度是 32 的倍數",
              profile.width % 32 == 0 and profile.height % 32 == 0,
              f"{profile.width}x{profile.height}")
        check(f"{key} 的長度在 17k+5 網格上", (profile.length - 5) % 17 == 0,
              str(profile.length))
        check(f"{key} 的權重檔名不重複",
              len({m.filename for m in models}) == len(models))
    check("L4 預設開 sol-attn", config.L4.sol_attn is True)
    check("T4 不開 sol-attn（SM75 不支援）", config.T4.sol_attn is False)
    check("pick_profile(24)->l4", config.pick_profile(24).key == "l4")
    check("pick_profile(24, ref)->l4_ref", config.pick_profile(24, needs_ref=True).key == "l4_ref")
    check("pick_profile(40)->a100", config.pick_profile(40).key == "a100")
    check("pick_profile(16)->t4", config.pick_profile(16).key == "t4")
    check("PRISM 是 ref2va 路徑", "r2v" in config.PRISM_W6A8.task)


# --- prompt compiler --------------------------------------------------------

def test_prompt_compiler() -> None:
    print("\n[prompt_compiler]")
    brief = prompt_compiler.Brief(
        subject="a Taiwanese news anchor in a navy blazer",
        scene="a modern broadcast studio with a lit desk",
        look="50mm lens, shallow depth of field, soft key light",
        beats=["turns to camera and smiles", "raises a tablet"],
        camera="steady tripod, medium shot",
        duration=6.0,
        audio="studio room tone",
    )
    text = prompt_compiler.compile_prompt(brief)
    check("有 Storyboard 段", "Storyboard:" in text)
    check("有時間碼", "[0s-3s]" in text and "[3s-6s]" in text, text)
    check("有 Audio 行", "Audio: studio room tone" in text, text)
    check("有禁止項", "No text, logos or watermarks." in text, text)
    check("每個節拍都有相機語言", text.count("steady tripod, medium shot") == 2, text)
    check("不含中文", not prompt_compiler.has_cjk(text), text)

    silent = prompt_compiler.compile_prompt(
        prompt_compiler.Brief(subject="a cat", scene="a desk", duration=3.0)
    )
    check("沒給音訊時是靜音", "No audio - silent video." in silent, silent)

    voice = prompt_compiler.compile_prompt(
        prompt_compiler.Brief(subject="a presenter", scene="a studio",
                              beats=["says hello"], duration=3.0, voice_driven=True)
    )
    check("聲音驅動會提到 <Audio 1>", "<Audio 1>" in voice, voice)
    check("聲音驅動會寫 lip synchronized",
          "synchronized to <Audio 1>" in voice, voice)

    lock = prompt_compiler.compile_prompt(
        prompt_compiler.Brief(subject="a woman in a red coat", scene="a snowy street",
                              beats=["walks forward", "stops and looks back"],
                              duration=4.0,
                              template="identity_lock",
                              identity_lock="a woman in a red coat, 30s, black bob")
    )
    check("identity-lock 版式有 IDENTITY LOCK", "IDENTITY LOCK:" in lock, lock)
    check("identity-lock 版式有 [Shot 1]", "[Shot 1]" in lock, lock)
    check("identity-lock 版式提到接龍",
          "Continuous with previous clip" in lock or "Do not reset pose" in lock, lock)
    check("identity-lock 標頭沒有空逗號", ", ," not in lock, lock)

    try:
        prompt_compiler.enhance(
            prompt_compiler.Brief(subject="一隻貓在桌上", duration=3.0)
        )
        check("中文 brief 在 rule 後端報錯", False, "居然沒報錯")
    except ValueError:
        check("中文 brief 在 rule 後端報錯", True)

    beats = prompt_compiler.distribute_beats(["a", "b", "c"], 9.0)
    check("三節拍平分 9 秒",
          [b.start for b in beats] == [0.0, 3.0, 6.0] and beats[-1].end == 9.0,
          str(beats))
    timed = prompt_compiler.retime_beats(beats, [1.5, 2.0, 3.25])
    check("回填時間碼累積正確",
          timed[0].end == 1.5 and timed[1].end == 3.5 and timed[2].end == 6.75,
          str(timed))
    check("時間碼不超過音軌長度", timed[-1].end <= 9.0)
    try:
        prompt_compiler.retime_beats(beats, [1.0])
        check("長度不符會報錯", False)
    except ValueError:
        check("長度不符會報錯", True)


# --- payloads ---------------------------------------------------------------

def test_payloads() -> None:
    print("\n[payloads]")
    turbo_spec = payloads.spec_from_profile(config.L4, "hello")
    graph, notes = payloads.build_graph(turbo_spec, available=None)
    classes = [node["class_type"] for node in graph.values()]
    check("Turbo 路徑不含 SigmaShift", "MiniMaxH3SigmaShift" not in classes, str(classes))
    check("Turbo 路徑有 TurboLoRA", "MiniMaxH3TurboLoRA" in classes, str(classes))
    check("Turbo 路徑有 TurboSampler", "MiniMaxH3TurboSampler" in classes, str(classes))
    check("Turbo 路徑有 FFN 分塊", "MiniMaxH3ChunkFeedForward" in classes, str(classes))
    check("Turbo 路徑有 sol-attn",
          "MiniMaxH3MemoryEfficientSolAttentionPatch" in classes, str(classes))
    check("條件化是 ImageToVideo", "MiniMaxH3ImageToVideo" in classes, str(classes))
    check("輸出是 SaveVideo", "SaveVideo" in classes, str(classes))

    official_spec = payloads.spec_from_profile(config.A100, "hello")
    off_graph, _ = payloads.build_graph(official_spec, available=None)
    off_classes = [node["class_type"] for node in off_graph.values()]
    check("官方路徑有 SigmaShift", "MiniMaxH3SigmaShift" in off_classes, str(off_classes))
    check("官方路徑用 KSamplerSelect", "KSamplerSelect" in off_classes, str(off_classes))
    check("官方路徑沒有 TurboLoRA", "MiniMaxH3TurboLoRA" not in off_classes, str(off_classes))

    missing = {"UNETLoader", "CLIPLoader", "VAELoader", "MiniMaxH3ImageToVideo",
               "BasicScheduler", "BasicGuider", "RandomNoise", "SamplerCustomAdvanced",
               "VAEDecode", "VAEDecodeAudio", "CreateVideo", "SaveVideo",
               "KSamplerSelect"}
    degraded, degraded_notes = payloads.build_graph(turbo_spec, available=missing)
    check("缺 Turbo 節點會降級並提示",
          any("TurboLoRA" in note for note in degraded_notes), str(degraded_notes))
    check("缺 sol-attn 會提示",
          any("sol-attn" in note for note in degraded_notes), str(degraded_notes))

    r2v_spec = payloads.spec_from_profile(config.L4_REF, "hello", task="r2v")
    r2v_spec.ref_audios = ["voice.wav"]
    r2v_spec.ref_images = ["face.png"]
    r2v_graph, _ = payloads.build_graph(r2v_spec, available=None)
    cond = [node for node in r2v_graph.values()
            if node["class_type"] == "MiniMaxH3ReferenceToVideo"][0]
    check("r2v 掛上 ref_audio_0", cond["inputs"].get("ref_audio_0") == "voice.wav",
          str(cond["inputs"]))
    check("r2v 掛上 ref_image_0", cond["inputs"].get("ref_image_0") == "face.png",
          str(cond["inputs"]))
    check("r2v 帶 audio_vae", "audio_vae" in cond["inputs"], str(cond["inputs"]))

    for name, g in (("turbo", graph), ("official", off_graph), ("r2v", r2v_graph)):
        broken = []
        for node in g.values():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    if value[0] not in g:
                        broken.append((node["class_type"], value))
        check(f"{name} 圖的連結都指向存在的節點", not broken, str(broken))


# --- 活體驗證（需要本機 ComfyUI） -------------------------------------------

def test_live_server() -> None:
    print("\n[live server]")
    try:
        from h3colab.client import ComfyClient

        client = ComfyClient("http://127.0.0.1:8188", timeout=10)
        info = client.object_info()
    except Exception as exc:  # noqa: BLE001
        print(f"  SKIP 沒有可用的 ComfyUI（{exc}）")
        return

    check("ComfyUI 有 MiniMaxH3ImageToVideo", "MiniMaxH3ImageToVideo" in info)
    check("ComfyUI 有 MiniMaxH3ReferenceToVideo", "MiniMaxH3ReferenceToVideo" in info)
    check("ComfyUI 有 MiniMaxH3SigmaShift", "MiniMaxH3SigmaShift" in info)

    spec = payloads.spec_from_profile(config.L4, "hello")
    graph, _ = payloads.build_graph(spec, available=set(info.keys()))
    problems: list[str] = []
    for node_id, node in graph.items():
        schema = info.get(node["class_type"])
        if schema is None:
            problems.append(f"{node_id}: 伺服器沒有 {node['class_type']}")
            continue
        allowed = set((schema.get("input", {}).get("required") or {}).keys())
        allowed |= set((schema.get("input", {}).get("optional") or {}).keys())
        for key in node["inputs"]:
            if key not in allowed:
                problems.append(f"{node_id}.{node['class_type']} 不接受輸入 {key}")
    check("圖中每個節點的輸入名稱都被伺服器接受", not problems, "; ".join(problems[:6]))


def main() -> int:
    print("h3colab 自我檢查")
    test_config()
    test_prompt_compiler()
    test_payloads()
    test_live_server()
    print(f"\n通過 {len(PASSED)} 項，失敗 {len(FAILED)} 項")
    for item in FAILED:
        print(f"  - {item}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
