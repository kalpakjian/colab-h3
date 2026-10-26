"""Colab ⇄ ComfyUI ⇄ MiniMax H3 的客戶端套件。

模組地圖：

* ``config``          —— 模型檔案來源、算力檔案（profile）
* ``prompt_compiler`` —— 提示強化：brief → 正確的 H3 提示組
* ``tts_voice``       —— 中文 TTS → 32kHz wav + 真實音軌長度
* ``payloads``        —— ComfyUI API 節點圖建構
* ``client``          —— ComfyUI HTTP 客戶端
* ``setup_colab``     —— 冪等安裝 / 權重倉庫
* ``cli``             —— 命令列入口
"""

from .config import (
    ALL_MODELS, PROFILES, Profile, ModelFile, pick_profile, snap_length,
)
from .prompt_compiler import (
    Beat, Brief, LLMConfig, compile_prompt, distribute_beats, enhance,
    has_cjk, retime_beats,
)
from .payloads import GenSpec, build_graph, spec_from_profile
from .client import ComfyClient, ComfyError

__all__ = [
    "ALL_MODELS", "PROFILES", "Profile", "ModelFile", "pick_profile", "snap_length",
    "Beat", "Brief", "LLMConfig", "compile_prompt", "distribute_beats", "enhance",
    "has_cjk", "retime_beats",
    "GenSpec", "build_graph", "spec_from_profile",
    "ComfyClient", "ComfyError",
]
