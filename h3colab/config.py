"""設定：模型檔案來源、部署 profile、節點倉庫。

權重來源（皆經 HuggingFace API 實查檔名與大小）：

* 官方 H3 權重      ``Comfy-Org/MiniMax-H3``
* Turbo LoRA       ``larryvrh/MiniMax-H3-Turbo-Lora``
* PRISM（ref2va merge）``OliviaRossi/Minimax-h3_Singularity-x-10Eros-Max_PRISM``

Profile 是「算力檔案」：一組 (GPU 級別 → 解析度 / 幀數 / 步數 / 權重精度 /
節點選項)。Colab 上先 ``doctor`` 認出 GPU，再 ``setup --profile <key>``。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# --- 路徑 -------------------------------------------------------------------

# Colab 預設是 /content/ComfyUI；本機實測可設 H3COMFY_ROOT=D:\AI\ComfyUI
# .strip() 是必要的：cmd 的 `set VAR=value && cmd` 會把尾端空白帶進值。
COMFY_ROOT = os.environ.get("H3COMFY_ROOT", "/content/ComfyUI").strip()
DRIVE_ROOT = os.environ.get("H3DRIVE_ROOT", "/content/drive/MyDrive/ComfyUI-H3").strip()

# --- 倉庫 -------------------------------------------------------------------

REPO_ORG = "Comfy-Org/MiniMax-H3"
REPO_TURBO = "larryvrh/MiniMax-H3-Turbo-Lora"
REPO_PRISM = "OliviaRossi/Minimax-h3_Singularity-x-10Eros-Max_PRISM"

GIT_TURBO = "https://github.com/larryvrh/ComfyUI-MiniMax-H3-Turbo"
GIT_SOL_ATTN = "https://github.com/Saganaki22/ComfyUI-sol-attn"
GIT_DIRECTOR = "https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director"
GIT_VIDEO_HELPER = "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"


@dataclass(frozen=True)
class ModelFile:
    """一個權重檔案：HF 來源 + 落地位置。"""

    key: str
    repo: str
    filename: str
    subdir: str
    size_gb: float
    task: str
    note: str = ""

    @property
    def local_name(self) -> str:
        return self.filename

    @property
    def local_path(self) -> str:
        return f"{COMFY_ROOT}/models/{self.subdir}/{self.filename}"

    @property
    def drive_path(self) -> str:
        return f"{DRIVE_ROOT}/models/{self.subdir}/{self.filename}"

    @property
    def hf_url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{self.filename}"


# --- 權重清單 ---------------------------------------------------------------

FL2VA_W6A8 = ModelFile(
    "fl2va_w6a8", REPO_ORG, "minimax_h3_fl2va_pruned_w6a8.safetensors",
    "diffusion_models", 14.89, "t2v/i2v/fl2v",
    "pruned + w6a8，24GB 卡首選。",
)
FL2VA_INT8 = ModelFile(
    "fl2va_int8", REPO_ORG, "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "diffusion_models", 19.53, "t2v/i2v/fl2v",
    "本機 D:\\AI\\ComfyUI 用的就是這顆，品質基線。",
)
FL2VA_BF16 = ModelFile(
    "fl2va_bf16", REPO_ORG, "minimax_h3_fl2va_pruned_bf16.safetensors",
    "diffusion_models", 37.46, "t2v/i2v/fl2v",
    "40GB 以下別用。",
)
REF2VA_W6A8 = ModelFile(
    "ref2va_w6a8", REPO_ORG, "minimax_h3_ref2va_pruned_w6a8.safetensors",
    "diffusion_models", 14.89, "r2v/v2v/rv2v",
    "參考主體／參考音訊（聲音驅動）路徑。",
)
REF2VA_INT8 = ModelFile(
    "ref2va_int8", REPO_ORG, "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "diffusion_models", 19.53, "r2v/v2v/rv2v",
)
PRISM_W6A8 = ModelFile(
    "prism_w6a8", REPO_PRISM,
    "Minimax-h3_Singularity_x_10Eros-Max_PRISM_w6a8.safetensors",
    "diffusion_models", 14.81, "r2v/v2v/rv2v",
    "Singularity x 10Eros-Max PRISM merge，ref2va。",
)
PRISM_INT8 = ModelFile(
    "prism_int8", REPO_PRISM,
    "Minimax-h3_Singularity_x_10Eros-Max_PRISM_int8.safetensors",
    "diffusion_models", 18.89, "r2v/v2v/rv2v",
)
PRISM_FP16 = ModelFile(
    "prism_fp16", REPO_PRISM,
    "Minimax-h3_Singularity_x_10Eros-Max_PRISM_fp16.safetensors",
    "diffusion_models", 61.70, "r2v/v2v/rv2v",
    "太肥，Colab 別用。",
)
CLIP_NVFP4 = ModelFile(
    "clip_nvfp4", REPO_ORG, "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "text_encoders", 14.61, "all",
    "Qwen3-VL 32B 4-bit；CLIPLoader 的 type 必須是 minimax。",
)
VIDEO_VAE_FP16 = ModelFile(
    "video_vae_fp16", REPO_ORG, "minimax_h3_video_vae_fp16.safetensors",
    "vae", 4.85, "all",
)
VIDEO_VAE_INT8 = ModelFile(
    "video_vae_int8", REPO_ORG, "minimax_h3_video_vae_int8_convrot.safetensors",
    "vae", 2.62, "all",
    "省 2.2GB，24GB 卡用這個。",
)
AUDIO_VAE_FP32 = ModelFile(
    "audio_vae_fp32", REPO_ORG, "minimax_h3_audio_vae_fp32.safetensors",
    "vae", 0.56, "all",
    "H3 原生立體聲；少了就沒聲音。",
)
TURBO_V4 = ModelFile(
    "turbo_v4", REPO_TURBO, "minimax_h3_turbo_v4_step600_ema.safetensors",
    "loras", 0.73, "fl2va",
    "6–8 步成片；4 步 + 大動作會拖影。",
)
REF2V_TURBO = ModelFile(
    "ref2v_turbo", REPO_ORG, "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
    "loras", 1.82, "ref2va",
    "ref2va 專用 turbo LoRA。",
)

ALL_MODELS: dict[str, ModelFile] = {
    m.key: m
    for m in (
        FL2VA_W6A8, FL2VA_INT8, FL2VA_BF16,
        REF2VA_W6A8, REF2VA_INT8,
        PRISM_W6A8, PRISM_INT8, PRISM_FP16,
        CLIP_NVFP4, VIDEO_VAE_FP16, VIDEO_VAE_INT8, AUDIO_VAE_FP32,
        TURBO_V4, REF2V_TURBO,
    )
}

# --- 算力檔案 ---------------------------------------------------------------

@dataclass(frozen=True)
class Profile:
    """一組「GPU 級別 → 該怎麼跑」的參數。"""

    key: str
    label: str
    vram_gb: int
    width: int
    height: int
    length: int
    steps: int
    mode: str                # "official" = 官方 SigmaShift 流程；"turbo" = Turbo LoRA + Turbo Sampler
    unet: ModelFile
    clip: ModelFile
    video_vae: ModelFile
    audio_vae: ModelFile
    turbo_lora: ModelFile | None = None
    sol_attn: bool = False
    ffn_chunks: int = 1
    lowvram: bool = False
    warn: str = ""
    extra_models: tuple[ModelFile, ...] = ()

    @property
    def models(self) -> tuple[ModelFile, ...]:
        items = [self.unet, self.clip, self.video_vae, self.audio_vae]
        if self.turbo_lora is not None:
            items.append(self.turbo_lora)
        items.extend(self.extra_models)
        return tuple(items)

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1_000_000


L4 = Profile(
    key="l4",
    label="L4 24GB · 0.4MP · 5 秒 · Turbo 8 步",
    vram_gb=24,
    width=864, height=480, length=124, steps=8,
    mode="turbo",
    unet=FL2VA_W6A8, clip=CLIP_NVFP4,
    video_vae=VIDEO_VAE_INT8, audio_vae=AUDIO_VAE_FP32,
    turbo_lora=TURBO_V4,
    sol_attn=True, ffn_chunks=4, lowvram=True,
    warn="L4 = SM89，sol-attn 官方支援；864x480 才是 0.4MP 草稿解析度。",
)

L4_REF = Profile(
    key="l4_ref",
    label="L4 24GB · 0.4MP · 5 秒 · PRISM(ref2va) 聲音驅動",
    vram_gb=24,
    width=864, height=480, length=124, steps=8,
    mode="official",
    unet=PRISM_W6A8, clip=CLIP_NVFP4,
    video_vae=VIDEO_VAE_INT8, audio_vae=AUDIO_VAE_FP32,
    sol_attn=True, ffn_chunks=4, lowvram=True,
    warn="PRISM 是 ref2va merge：走 r2v + <Audio 1>，不能當 t2v/i2v 用。"
         "它是 turbo merge，所以用 8 步（不是官方的 25 步）。",
)

L4_REF_TURBO = Profile(
    key="l4_ref_turbo",
    label="L4 24GB · 0.4MP · 5 秒 · 官方 ref2va + ref2v Turbo 4 步",
    vram_gb=24,
    width=864, height=480, length=124, steps=6,
    mode="turbo",
    unet=REF2VA_W6A8, clip=CLIP_NVFP4,
    video_vae=VIDEO_VAE_INT8, audio_vae=AUDIO_VAE_FP32,
    turbo_lora=REF2V_TURBO,
    sol_attn=True, ffn_chunks=4, lowvram=True,
    warn="這是本機 4090 已驗證的 ref2va + turbo 配方（4–6 步）。",
)

A100 = Profile(
    key="a100",
    label="A100 40GB · 1.0MP · 5 秒 · 官方 25 步",
    vram_gb=40,
    width=1344, height=768, length=124, steps=25,
    mode="official",
    unet=FL2VA_INT8, clip=CLIP_NVFP4,
    video_vae=VIDEO_VAE_FP16, audio_vae=AUDIO_VAE_FP32,
    sol_attn=True, ffn_chunks=2, lowvram=False,
    warn="1344x768 是本機 4090 的正式接龍解析度；A100 才吃得下。",
)

A100_REF = Profile(
    key="a100_ref",
    label="A100 40GB · 1.0MP · 5 秒 · PRISM(ref2va) 聲音驅動",
    vram_gb=40,
    width=1344, height=768, length=124, steps=25,
    mode="official",
    unet=PRISM_INT8, clip=CLIP_NVFP4,
    video_vae=VIDEO_VAE_FP16, audio_vae=AUDIO_VAE_FP32,
    sol_attn=True, ffn_chunks=2, lowvram=False,
)

T4 = Profile(
    key="t4",
    label="T4 16GB · 0.4MP · 5 秒 · 慢（不建議）",
    vram_gb=16,
    width=864, height=480, length=124, steps=8,
    mode="turbo",
    unet=FL2VA_W6A8, clip=CLIP_NVFP4,
    video_vae=VIDEO_VAE_INT8, audio_vae=AUDIO_VAE_FP32,
    turbo_lora=TURBO_V4,
    sol_attn=False, ffn_chunks=4, lowvram=True,
    warn="T4 = SM75：sol-attn 不支援（要 SM86+），H3 走 bf16 還會很慢。"
         "寧可花點數選 L4，也不要 T4。",
)

PROFILES: dict[str, Profile] = {
    p.key: p for p in (L4, L4_REF, L4_REF_TURBO, A100, A100_REF, T4)
}


def pick_profile(vram_gb: float, *, needs_ref: bool = False) -> Profile:
    """依 VRAM 挑一個合理 profile（needs_ref = 聲音驅動 / r2v 路徑）。"""
    if vram_gb >= 38:
        return A100_REF if needs_ref else A100
    if vram_gb >= 22:
        return L4_REF if needs_ref else L4
    return T4


# --- 任務 / 幀格 ------------------------------------------------------------

TASK_TYPES = {
    "t2v": "文生音視頻（fl2va UNET）",
    "i2v": "圖生音視頻（fl2va UNET）",
    "fl2v": "首尾幀生視頻（fl2va UNET）",
    "r2v": "參考主體 + 參考音訊生視頻（ref2va UNET）← 聲音驅動數字人",
}

# MiniMax H3 的 17k+5 幀格：124 = 5.16s、243 = 10.1s、362 = 15.1s
FRAME_GRID = tuple(17 * k + 5 for k in range(0, 22))


def snap_length(frames: int) -> int:
    """把任意幀數吸附到 H3 的 17k+5 網格（>= 目標值）。"""
    if frames <= 5:
        return 5
    k = (frames - 5 + 16) // 17
    return 17 * k + 5

