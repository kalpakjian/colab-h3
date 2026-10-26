"""TTS：中文語音 → 32kHz 單聲道 wav，並量出**真實音軌長度**。

為什麼要量長度：聲音驅動的數字人，畫面時間碼必須跟語速一致，
不然嘴型對不上、就會「亂講話」。這裡把每句台詞的實際秒數回報出來，
交給 prompt_compiler.retime_beats() 回填時間碼。

後端（依序嘗試）：
* ``edge``  —— edge-tts，免費、無需金鑰，繁中音色最齊
* ``gtts``  —— gTTS，免費備援（音色較機械）
"""

from __future__ import annotations

import asyncio
import os
import shutil
import struct
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 32000  # H3 音訊 VAE 的原生採樣率

VOICES = {
    "zh-TW-HsiaoChenNeural": "台灣女聲（自然、播報感）",
    "zh-TW-HsiaoYuNeural": "台灣女聲（年輕）",
    "zh-TW-YunxiNeural": "台灣男聲（年輕、活潑）",
    "zh-TW-YunYangNeural": "台灣男聲（播報）",
    "zh-CN-XiaoxiaoNeural": "中國女聲",
    "zh-CN-YunxiNeural": "中國男聲",
}

DEFAULT_VOICE = "zh-TW-HsiaoChenNeural"


@dataclass
class ScriptLine:
    """一句台詞 + 它的音檔 + 真實長度。"""

    text: str
    path: str
    duration: float


class TTSError(RuntimeError):
    pass


# --- 音訊工具 ---------------------------------------------------------------

def to_wav(src: str | os.PathLike, dst: str | os.PathLike, sample_rate: int = SAMPLE_RATE) -> str:
    """轉成 32kHz 單聲道 16-bit wav（H3 音訊 VAE 要這個格式）。"""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise TTSError(
            "找不到 ffmpeg：Colab 先跑 !apt-get install -y ffmpeg，"
            "或用 !pip install imageio-ffmpeg 拿一份。"
        )
    cmd = [
        ffmpeg, "-y", "-i", str(src),
        "-ac", "1", "-ar", str(sample_rate), "-acodec", "pcm_s16le",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise TTSError(f"ffmpeg 轉檔失敗：{result.stderr.decode('utf-8', 'ignore')[-400:]}")
    return str(dst)


def probe_duration(path: str | os.PathLike) -> float:
    """量音檔長度（wav 用 wave 模組；其他格式走 ffprobe）。"""
    path = Path(path)
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate() or SAMPLE_RATE
            return round(frames / float(rate), 3)
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            try:
                return round(float(result.stdout.strip()), 3)
            except ValueError:
                pass
    raise TTSError(f"量不到音檔長度：{path}")


def concat_wavs(paths: list[str], dst: str | os.PathLike) -> str:
    """把多段 wav 串成一條（純 PCM 串接，不重採樣）。"""
    if not paths:
        raise TTSError("沒有要串接的音檔")
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(paths[0]), "rb") as first:
        params = first.getparams()
        frames = [first.readframes(first.getnframes())]
    for path in paths[1:]:
        with wave.open(str(path), "rb") as handle:
            if handle.getnchannels() != params.nchannels or \
               handle.getframerate() != params.framerate:
                raise TTSError("串接的音檔格式不一致，請先統一成 32kHz 單聲道")
            frames.append(handle.readframes(handle.getnframes()))
    with wave.open(str(dst), "wb") as out:
        out.setparams(params)
        for chunk in frames:
            out.writeframes(chunk)
    return str(dst)


# --- 合成 -------------------------------------------------------------------

async def _edge_synth(text: str, voice: str, out_path: str, rate: str, pitch: str) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(out_path)


async def _gtts_synth(text: str, out_path: str, lang: str = "zh-TW") -> None:
    from gtts import gTTS

    await asyncio.to_thread(gTTS(text=text, lang=lang).save, out_path)


def synthesize(
    text: str,
    out_path: str | os.PathLike,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    backend: str = "edge",
) -> str:
    """把一句話合成成 wav（32kHz 單聲道），回傳路徑。"""
    text = (text or "").strip()
    if not text:
        raise TTSError("空字串")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = out_path.with_suffix(".raw.mp3" if backend == "edge" else ".raw.mp3")

    try:
        if backend == "edge":
            asyncio.run(_edge_synth(text, voice, str(raw), rate, pitch))
        elif backend == "gtts":
            asyncio.run(_gtts_synth(text, str(raw)))
        else:
            raise TTSError(f"未知 TTS 後端：{backend}")
    except ImportError as exc:
        raise TTSError(
            f"缺套件：{exc}。Colab 請跑 !pip install edge-tts"
            "（備援：!pip install gTTS 後用 backend='gtts'）"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise TTSError(f"TTS 失敗：{exc}") from exc

    to_wav(raw, out_path)
    raw.unlink(missing_ok=True)
    return str(out_path)


def synth_script(
    lines: list[str],
    out_dir: str | os.PathLike,
    voice: str = DEFAULT_VOICE,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    backend: str = "edge",
) -> list[ScriptLine]:
    """逐句合成，回傳每句的音檔路徑與真實長度。

    句與句之間不插入停頓（H3 會自行處理韻律）；要停頓請在文字裡加標點。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: list[ScriptLine] = []
    for index, text in enumerate(lines):
        text = (text or "").strip()
        if not text:
            continue
        path = out_dir / f"line_{index:02d}.wav"
        synthesize(text, path, voice=voice, rate=rate, pitch=pitch, backend=backend)
        result.append(ScriptLine(text=text, path=str(path), duration=probe_duration(path)))
    return result


def list_voices() -> dict[str, str]:
    """本專案推薦的音色表。"""
    return dict(VOICES)


def install_hint() -> str:
    return (
        "Colab 安裝：!pip install -q edge-tts && !apt-get install -y ffmpeg\n"
        "（Edge TTS 是微軟線上服務，免費但需要網路；離線可改用 gTTS）"
    )
