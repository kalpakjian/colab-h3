"""Colab 佈署：冪等安裝 ComfyUI、節點、權重（雲端硬碟當倉庫）。

設計原則
--------
1. **冪等**：每一步都先檢查再做事，重跑不會重複下載、重複 clone。
2. **倉庫與執行分離**：權重可以存在 Google Drive（5TB 慢慢長），執行前再
   複製一份到本機磁碟（Drive 的 FUSE 讀取很慢，直接從 Drive 載入模型會
   慢到不能忍）。``--store local`` 則是單次直落本機，最快但不留檔。
3. **依算力選權重**：24GB 卡用 w6a8 + 4-bit CLIP，40GB 卡用 int8。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import (
    COMFY_ROOT, DRIVE_ROOT, GIT_DIRECTOR, GIT_SOL_ATTN, GIT_TURBO,
    GIT_VIDEO_HELPER, ModelFile, Profile,
)

NODE_REPOS = {
    "ComfyUI-MiniMax-H3-Turbo": GIT_TURBO,
    "ComfyUI-sol-attn": GIT_SOL_ATTN,
    "ComfyUI_MiniMaxH3_Director": GIT_DIRECTOR,
    "ComfyUI-VideoHelperSuite": GIT_VIDEO_HELPER,
}

REQUIRED_NODES = ("ComfyUI-MiniMax-H3-Turbo",)
OPTIONAL_NODES = ("ComfyUI-sol-attn", "ComfyUI_MiniMaxH3_Director", "ComfyUI-VideoHelperSuite")


def log(message: str) -> None:
    print(f"[h3colab] {message}", flush=True)


def run(cmd: list[str], cwd: str | os.PathLike | None = None, check: bool = True) -> int:
    log("$ " + " ".join(str(part) for part in cmd))
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if check and result.returncode != 0:
        raise RuntimeError(f"指令失敗（{result.returncode}）：{' '.join(cmd)}")
    return result.returncode


# --- GPU 偵測 ---------------------------------------------------------------

def detect_gpu() -> dict:
    """回傳 {'name', 'vram_gb', 'capability', 'available'}。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"name": "none", "vram_gb": 0.0, "capability": None, "available": False}
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        return {
            "name": props.name,
            "vram_gb": round(props.total_memory / (1024 ** 3), 1),
            "capability": f"{props.major}{props.minor}",
            "available": True,
        }
    except Exception as exc:  # noqa: BLE001
        log(f"GPU 偵測失敗：{exc}")
        return {"name": "unknown", "vram_gb": 0.0, "capability": None, "available": False}


def sol_attn_supported(capability: str | None) -> bool:
    """sol-attn 支援的架構：SM86 / SM89 / SM90 / SM100 / SM120 / SM121。

    L4 = SM89 ✅、A100 = SM80 ❌、T4 = SM75 ❌。
    """
    if not capability:
        return False
    return capability in {"86", "89", "90", "100", "120", "121"}


# --- ComfyUI ----------------------------------------------------------------

def ensure_comfyui(root: str | os.PathLike = COMFY_ROOT) -> Path:
    """確認 ComfyUI 在位（H3 需要 >= 0.30.0）。"""
    root = Path(root)
    if (root / "main.py").exists():
        log(f"ComfyUI 已在 {root}")
    else:
        log(f"clone ComfyUI -> {root}")
        root.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", "https://github.com/comfyanonymous/ComfyUI.git",
             str(root)])
    ensure_torch(root)
    requirements = root / "requirements.txt"
    if requirements.exists():
        log("安裝 ComfyUI requirements")
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)])
    return root


def ensure_torch(root: str | os.PathLike = COMFY_ROOT) -> None:
    """保證 torch 有 CUDA；Colab 內建 torch 通常已可用。"""
    try:
        import torch

        if torch.cuda.is_available():
            log(f"torch {torch.__version__} 已可用 CUDA")
            return
    except Exception:  # noqa: BLE001
        pass
    log("安裝 CUDA 版 torch（cu124）")
    run([
        sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu124",
    ])


def ensure_nodes(root: str | os.PathLike = COMFY_ROOT, include_optional: bool = True) -> dict[str, str]:
    """clone 需要的 custom nodes（已存在就 git pull）。"""
    custom = Path(root) / "custom_nodes"
    custom.mkdir(parents=True, exist_ok=True)
    wanted = list(REQUIRED_NODES) + (list(OPTIONAL_NODES) if include_optional else [])
    result: dict[str, str] = {}
    for name in wanted:
        url = NODE_REPOS[name]
        path = custom / name
        if (path / ".git").exists():
            log(f"更新 {name}")
            run(["git", "-C", str(path), "pull", "--ff-only"], check=False)
        else:
            log(f"clone {name}")
            if not run(["git", "clone", "--depth", "1", url, str(path)], check=False):
                result[name] = "FAILED"
                continue
        req = path / "requirements.txt"
        if req.exists():
            run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=False)
        result[name] = str(path)
    return result


# --- 權重 -------------------------------------------------------------------

def _hf_download(model: ModelFile, dest_dir: Path) -> Path:
    """用 huggingface_hub 下載單一檔案到 dest_dir（支援續傳）。"""
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    log(f"下載 {model.filename}（{model.size_gb} GB）← {model.repo}")
    path = hf_hub_download(
        repo_id=model.repo,
        filename=model.filename,
        local_dir=str(dest_dir),
    )
    return Path(path)


def store_root(store: str) -> Path:
    """store='drive' → 雲端硬碟倉庫；store='local' → 本機磁碟。"""
    if store == "drive":
        return Path(DRIVE_ROOT)
    if store == "local":
        return Path(COMFY_ROOT)
    raise ValueError(f"未知 store：{store}")


def download_model(
    model: ModelFile,
    store: str = "drive",
    materialize: bool = True,
    root: str | os.PathLike = COMFY_ROOT,
) -> Path:
    """下載一個權重檔，回傳它在 ComfyUI models/ 底下的最終路徑。"""
    target_dir = Path(COMFY_ROOT) / "models" / model.subdir
    final = target_dir / model.filename
    if final.exists() and final.stat().st_size > 0:
        log(f"已存在，略過：{final.name}")
        return final

    warehouse = store_root(store) / "models" / model.subdir
    downloaded = _hf_download(model, warehouse)

    if materialize:
        target_dir.mkdir(parents=True, exist_ok=True)
        if downloaded.resolve() != final.resolve():
            log(f"複製到執行目錄：{final.name}（Drive 讀取慢，複製後才跑得動）")
            shutil.copy2(downloaded, final)
        return final
    return downloaded


def download_profile(
    profile: Profile,
    store: str = "drive",
    materialize: bool = True,
    extra: tuple[ModelFile, ...] = (),
) -> list[Path]:
    """把一個 profile 需要的權重全部備齊。"""
    paths: list[Path] = []
    for model in list(profile.models) + list(extra):
        paths.append(download_model(model, store=store, materialize=materialize))
    return paths


def model_report(profile: Profile, root: str | os.PathLike = COMFY_ROOT) -> list[dict]:
    """檢查權重是否就位。"""
    rows: list[dict] = []
    for model in profile.models:
        path = Path(root) / "models" / model.subdir / model.filename
        rows.append({
            "file": model.filename,
            "subdir": model.subdir,
            "size_gb": model.size_gb,
            "present": path.exists(),
            "path": str(path),
        })
    return rows


# --- ComfyUI 啟動 -----------------------------------------------------------

def start_comfyui(
    root: str | os.PathLike = COMFY_ROOT,
    port: int = 8188,
    lowvram: bool = False,
    extra_args: list[str] | None = None,
    log_path: str | os.PathLike | None = None,
) -> subprocess.Popen:
    """背景啟動 ComfyUI，回傳 Popen（呼叫端要自己收掉）。"""
    root = Path(root)
    log_path = Path(log_path or (root / "logs" / "comfyui_colab.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(port)]
    if lowvram:
        cmd.append("--lowvram")
    cmd.extend(extra_args or [])
    out = open(log_path, "ab")
    log(f"啟動 ComfyUI：{' '.join(cmd)}")
    return subprocess.Popen(
        cmd, cwd=str(root), stdout=out, stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def wait_ready(base_url: str = "http://127.0.0.1:8188", timeout: int = 180) -> bool:
    """等 ComfyUI 起來。"""
    import time

    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/system_stats", timeout=5)
            if response.status_code == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    return False
