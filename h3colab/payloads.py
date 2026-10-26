"""ComfyUI API 圖建構：把規格翻成可直接 POST /prompt 的節點圖。

節點接線依據（全部對過真實 ComfyUI /object_info）：

* 官方 25 步路徑：``UNETLoader → MiniMaxH3SigmaShift(12,3) → BasicScheduler(simple)
  → BasicGuider → SamplerCustomAdvanced``
* Turbo 路徑：``UNETLoader → MiniMaxH3TurboLoRA → BasicGuider/BasicScheduler
  + MiniMaxH3TurboSampler``（**不含** SigmaShift — 官方 Turbo 範例工作流與本機
  已驗證 payload 都是這樣）
* 條件化：``MiniMaxH3ImageToVideo``（t2v/i2v/fl2v）或
  ``MiniMaxH3ReferenceToVideo``（r2v，音訊走 ``ref_audio_N`` 槽）

選配：sol-attn 的稀疏注意力 / FFN 分塊（節點沒裝就自動略過）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import snap_length


@dataclass
class GenSpec:
    """一次生成的完整規格。"""

    task: str = "t2v"                 # t2v | i2v | fl2v | r2v
    prompt: str = ""
    width: int = 864
    height: int = 480
    length: int = 124
    fps: float = 24.0
    seed: int = 42
    steps: int = 8
    mode: str = "turbo"               # turbo | official
    sampler: str = "res_multistep"
    scheduler: str = "simple"
    sigma_shift: bool | None = None   # None = 依 mode 自動（turbo 不加、official 加）
    shift_video: float = 12.0
    shift_audio: float = 3.0
    unet: str = ""
    clip: str = ""
    video_vae: str = ""
    audio_vae: str = ""
    turbo_lora: str | None = None
    turbo_lowvram: bool = True
    sol_attn: bool = False
    sol_tau: float = 1.3
    sol_min_tokens: int = 4096
    sol_thresh: str = "diag"
    ffn_chunks: int = 1
    ffn_min_tokens: int = 8192
    first_frame: str | None = None
    last_frame: str | None = None
    ref_images: list[str] = field(default_factory=list)
    ref_audios: list[str] = field(default_factory=list)
    ref_videos: list[str] = field(default_factory=list)
    ref_image_size: str = "match"
    filename_prefix: str = "video/h3"
    video_format: str = "auto"
    video_codec: str = "auto"
    bit_depth: int = 8

    def resolved_length(self) -> int:
        return snap_length(int(self.length))


class _Graph:
    """依序配號的節點圖。"""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}
        self._counter = 0

    def add(self, class_type: str, inputs: dict) -> str:
        self._counter += 1
        node_id = str(self._counter)
        self.nodes[node_id] = {"class_type": class_type, "inputs": inputs}
        return node_id


def build_graph(
    spec: GenSpec,
    available: set[str] | None = None,
) -> tuple[dict, list[str]]:
    """建出 API 格式的節點圖。

    available = 伺服器上實際存在的節點類名；給了才知道選配節點能不能用。
    回傳 (graph, notes)。
    """
    notes: list[str] = []
    graph = _Graph()
    have = available

    def supports(class_type: str) -> bool:
        return have is None or class_type in have

    # --- 模型 ---------------------------------------------------------------
    unet_id = graph.add("UNETLoader", {
        "unet_name": spec.unet,
        "weight_dtype": "default",
    })
    clip_id = graph.add("CLIPLoader", {
        "clip_name": spec.clip,
        "type": "minimax",
        "device": "default",
    })
    video_vae_id = graph.add("VAELoader", {"vae_name": spec.video_vae})
    audio_vae_id = graph.add("VAELoader", {"vae_name": spec.audio_vae})

    model_id = unet_id

    use_shift = spec.sigma_shift
    if use_shift is None:
        use_shift = spec.mode != "turbo"
    if use_shift:
        model_id = graph.add("MiniMaxH3SigmaShift", {
            "model": [model_id, 0],
            "shift_video": spec.shift_video,
            "shift_audio": spec.shift_audio,
        })

    # --- 選配：sol-attn / FFN 分塊 / Turbo LoRA -----------------------------
    if spec.ffn_chunks and spec.ffn_chunks > 1:
        if supports("MiniMaxH3ChunkFeedForward"):
            model_id = graph.add("MiniMaxH3ChunkFeedForward", {
                "model": [model_id, 0],
                "enabled": True,
                "chunks": int(spec.ffn_chunks),
                "min_tokens": int(spec.ffn_min_tokens),
            })
        else:
            notes.append("MiniMaxH3ChunkFeedForward 不在伺服器上，略過 FFN 分塊。")

    if spec.sol_attn:
        if supports("MiniMaxH3MemoryEfficientSolAttentionPatch"):
            model_id = graph.add("MiniMaxH3MemoryEfficientSolAttentionPatch", {
                "model": [model_id, 0],
                "enabled": True,
                "tau": float(spec.sol_tau),
                "min_tokens": int(spec.sol_min_tokens),
                "strict": False,
                "thresh_type": spec.sol_thresh,
                "int8_qk": False,
                "int8_pv": False,
            })
        elif supports("SolAttentionPatch"):
            model_id = graph.add("SolAttentionPatch", {
                "model": [model_id, 0],
                "enabled": True,
                "tau": float(spec.sol_tau),
                "min_tokens": int(spec.sol_min_tokens),
                "strict": False,
                "thresh_type": spec.sol_thresh,
                "int8_qk": False,
                "int8_pv": False,
            })
        else:
            notes.append("sol-attn 節點不在伺服器上，改用預設注意力（速度會慢一些）。")

    if spec.mode == "turbo":
        if not spec.turbo_lora:
            raise ValueError("mode='turbo' 需要 turbo_lora")
        if supports("MiniMaxH3TurboLoRA"):
            model_id = graph.add("MiniMaxH3TurboLoRA", {
                "model": [model_id, 0],
                "lora_name": spec.turbo_lora,
                "strength": 1.0,
                "low_vram": bool(spec.turbo_lowvram),
            })
        else:
            notes.append(
                "MiniMaxH3TurboLoRA 不在伺服器上，改用官方 25 步流程（慢很多）。"
            )

    # --- 條件化 -------------------------------------------------------------
    length = spec.resolved_length()
    if spec.task == "r2v":
        cond_inputs: dict = {
            "clip": [clip_id, 0],
            "vae": [video_vae_id, 0],
            "audio_vae": [audio_vae_id, 0],
            "prompt": spec.prompt,
            "width": int(spec.width),
            "height": int(spec.height),
            "length": length,
            "ref_image_size": spec.ref_image_size,
        }
        for index, name in enumerate(spec.ref_images):
            cond_inputs[f"ref_image_{index}"] = name
        for index, name in enumerate(spec.ref_audios):
            cond_inputs[f"ref_audio_{index}"] = name
        for index, name in enumerate(spec.ref_videos):
            cond_inputs[f"ref_video_{index}"] = name
        cond_id = graph.add("MiniMaxH3ReferenceToVideo", cond_inputs)
    else:
        cond_inputs = {
            "clip": [clip_id, 0],
            "vae": [video_vae_id, 0],
            "prompt": spec.prompt,
            "width": int(spec.width),
            "height": int(spec.height),
            "length": length,
        }
        if spec.first_frame:
            cond_inputs["first_frame"] = spec.first_frame
        if spec.last_frame:
            cond_inputs["last_frame"] = spec.last_frame
        cond_id = graph.add("MiniMaxH3ImageToVideo", cond_inputs)

    # --- 採樣 ---------------------------------------------------------------
    scheduler_id = graph.add("BasicScheduler", {
        "model": [model_id, 0],
        "scheduler": spec.scheduler,
        "steps": int(spec.steps),
        "denoise": 1.0,
    })
    guider_id = graph.add("BasicGuider", {
        "model": [model_id, 0],
        "conditioning": [cond_id, 0],
    })
    if spec.mode == "turbo" and supports("MiniMaxH3TurboSampler"):
        sampler_id = graph.add("MiniMaxH3TurboSampler", {})
    else:
        sampler_id = graph.add("KSamplerSelect", {"sampler_name": spec.sampler})
    noise_id = graph.add("RandomNoise", {"noise_seed": int(spec.seed)})
    sampled_id = graph.add("SamplerCustomAdvanced", {
        "noise": [noise_id, 0],
        "guider": [guider_id, 0],
        "sampler": [sampler_id, 0],
        "sigmas": [scheduler_id, 0],
        "latent_image": [cond_id, 1],
    })

    # --- 解碼 / 輸出 --------------------------------------------------------
    images_id = graph.add("VAEDecode", {
        "samples": [sampled_id, 0],
        "vae": [video_vae_id, 0],
    })
    audio_id = graph.add("VAEDecodeAudio", {
        "samples": [sampled_id, 0],
        "vae": [audio_vae_id, 0],
    })
    video_id = graph.add("CreateVideo", {
        "images": [images_id, 0],
        "fps": float(spec.fps),
        "audio": [audio_id, 0],
        "bit_depth": int(spec.bit_depth),
    })
    graph.add("SaveVideo", {
        "video": [video_id, 0],
        "filename_prefix": spec.filename_prefix,
        "format": spec.video_format,
        "codec": spec.video_codec,
    })
    return graph.nodes, notes


def spec_from_profile(profile, prompt: str, **overrides) -> GenSpec:
    """用算力檔案（Profile）產生一份預設 GenSpec。"""
    spec = GenSpec(
        prompt=prompt,
        width=profile.width,
        height=profile.height,
        length=profile.length,
        steps=profile.steps,
        mode=profile.mode,
        unet=profile.unet.filename,
        clip=profile.clip.filename,
        video_vae=profile.video_vae.filename,
        audio_vae=profile.audio_vae.filename,
        turbo_lora=profile.turbo_lora.filename if profile.turbo_lora else None,
        turbo_lowvram=profile.lowvram,
        sol_attn=profile.sol_attn,
        ffn_chunks=profile.ffn_chunks,
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    return spec

