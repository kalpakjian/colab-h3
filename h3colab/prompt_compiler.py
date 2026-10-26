"""提示強化：把「一句話」變成 MiniMax H3 真正吃得下的提示組。

H3 用 Qwen3-VL 自由提示詞，但官方範例工作流的格式很固定：

    Realistic live-action cinematic look: <場景>, <鏡頭>, <景深>, <色調> ...

    Storyboard:
    [0s-2s] <節拍 1：主體 + 動作 + 運鏡>
    [2s-3.5s] <節拍 2>

    Audio: <與畫面同步的具體聲音>
    No text, logos or watermarks.

多數人（包括很多「數字人」工作流）漏掉的就是這三塊：
時間碼分鏡、音軌同步指示、禁止項。少了它們，H3 會自由發揮 → 亂講話、
嘴型對對不上、場景漂移。

本模組提供三種後端：

* ``rule``    —— 純規則，不需網路。英文 brief 直接編譯；中文 brief 會明確報錯
  （沒有 LLM 就別假裝能翻譯）。
* ``llm``     —— 任何 OpenAI 相容端點（OpenRouter / Ollama / LM Studio）或
  Gemini API。先用 LLM 寫出結構化草稿，再由本模組重新編排時間碼。
* ``director``—— 直接呼叫本機 ComfyUI 的 MiniMax H3 Director 節點
  ``POST /minimax/director/enhance``（用它內建的 Bernini 模板）。

「聲音驅動」關鍵：編譯器會把 TTS 的**真實音軌長度**回填成時間碼，
並在 Audio 行加上「嘴型對齊 <Audio 1>」的明示，H3 才知道要跟著聲音動。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))


@dataclass
class Beat:
    """一個分鏡節拍。"""

    start: float
    end: float
    action: str

    def render(self) -> str:
        return f"[{_fmt(self.start)}s-{_fmt(self.end)}s] {self.action.strip()}"


@dataclass
class Brief:
    """一次生成的所有輸入。"""

    subject: str = ""
    scene: str = ""
    beats: list[str] = field(default_factory=list)
    style: str = "Realistic live-action cinematic look"
    look: str = ""            # 鏡頭/景深/顆粒/色調等外觀細節
    camera: str = ""          # 預設運鏡（會附在每個節拍後面）
    audio: str | None = None  # None = 靜音
    voice_driven: bool = False  # 嘴型對齊 <Audio 1>
    duration: float = 5.0
    negatives: str = "No text, logos or watermarks."
    identity_lock: str = ""   # 填了就用 identity-lock 版式
    template: str = "storyboard"   # "storyboard" | "identity_lock"
    timed_beats: list[Beat] | None = None  # 已算好時間碼的節拍（優先於 beats）

    def validate(self) -> None:
        if self.template not in ("storyboard", "identity_lock"):
            raise ValueError(f"未知 template: {self.template}")
        if self.duration <= 0:
            raise ValueError("duration 必須 > 0")


def _fmt(value: float) -> str:
    """0 -> '0'、2.0 -> '2'、3.5 -> '3.5'。"""
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return text or "0"


# --- 時間碼 ------------------------------------------------------------------

def distribute_beats(actions: Sequence[str], duration: float) -> list[Beat]:
    """把節拍平均切在 duration 內（官方範例是等分的時間碼）。"""
    actions = [a for a in actions if a and a.strip()]
    if not actions:
        return [Beat(0.0, duration, "")]
    step = duration / len(actions)
    beats: list[Beat] = []
    for index, action in enumerate(actions):
        start = round(index * step, 2)
        end = round(duration if index == len(actions) - 1 else (index + 1) * step, 2)
        beats.append(Beat(start, end, action))
    return beats


def retime_beats(beats: Sequence[Beat], durations: Sequence[float]) -> list[Beat]:
    """依每句台詞的**真實音軌長度**重新切時間碼。

    durations 與 beats 一一對應；長度不足時最後一段吃掉剩餘時間。
    這是「聲音驅動」的核心：畫面時間碼必須跟語速一致，否則嘴型對不上。
    """
    if len(beats) != len(durations):
        raise ValueError(
            f"beats 與 durations 長度不符：{len(beats)} vs {len(durations)}"
        )
    out: list[Beat] = []
    cursor = 0.0
    for index, (beat, dur) in enumerate(zip(beats, durations)):
        end = cursor + max(0.2, float(dur))
        out.append(Beat(round(cursor, 2), round(end, 2), beat.action))
        cursor = end
    return out


# --- 編譯 --------------------------------------------------------------------

def compile_prompt(brief: Brief) -> str:
    """把 Brief 編譯成 H3 提示組（純規則，不碰網路）。"""
    brief.validate()
    if brief.template == "identity_lock":
        return _compile_identity_lock(brief)
    return _compile_storyboard(brief)


def _compile_storyboard(brief: Brief) -> str:
    blocks: list[str] = []

    head = brief.style.strip()
    detail = brief.scene.strip()
    if brief.subject.strip():
        detail = f"{brief.subject.strip()}, {detail}" if detail else brief.subject.strip()
    if brief.look.strip():
        detail = f"{detail}, {brief.look.strip()}" if detail else brief.look.strip()
    blocks.append(f"{head}: {detail}" if detail else head)

    beats = brief.timed_beats or distribute_beats(
        brief.beats or [brief.subject or brief.scene or "a slow continuous cinematic shot"],
        brief.duration,
    )
    if brief.camera.strip():
        beats = [
            Beat(b.start, b.end, f"{b.action.strip()}, {brief.camera.strip()}")
            for b in beats
        ]
    blocks.append("Storyboard:\n" + "\n".join(b.render() for b in beats))

    if brief.voice_driven:
        audio = (brief.audio or "clear spoken voice").strip()
        blocks.append(
            f"Audio: {audio}, spoken from <Audio 1>, lip movements and jaw "
            f"synchronized to <Audio 1>, steady face, no extra talking."
        )
    elif brief.audio:
        blocks.append(f"Audio: {brief.audio.strip()}")
    else:
        blocks.append("No audio - silent video.")

    if brief.negatives.strip():
        blocks.append(brief.negatives.strip())
    return "\n\n".join(blocks)


def _compile_identity_lock(brief: Brief) -> str:
    """本機 skill 用的版式：IDENTITY LOCK + [Shot n]（接龍風格）。"""
    header_parts = ["Photoreal cinematic", brief.scene.strip(), brief.look.strip()]
    header = ", ".join(part for part in header_parts if part) + ", 16:9, 24fps."
    lines: list[str] = [header]
    if brief.identity_lock.strip():
        lines.append(f"IDENTITY LOCK: {brief.identity_lock.strip()}")
        lines.append(
            "Same subject, same outfit, same location landmarks, "
            "same light direction entire clip."
        )
    else:
        lines.append("Same subject, same outfit, same setting entire clip.")

    beats = distribute_beats(brief.beats or [brief.subject], brief.duration)
    for index, beat in enumerate(beats, start=1):
        action = beat.action.strip()
        if index > 1:
            action = (
                "Continuous with previous clip. Start exactly from the previous "
                f"ending pose. Do not reset pose. {action}"
            )
        if brief.camera.strip():
            action = f"{action}, {brief.camera.strip()}"
        lines.append(f"[Shot {index}] {action}")

    lines.append(
        "Motion rules: grounded feet, no face morph, no outfit change, "
        "no extra people, no text, no logo, no subtitles."
    )
    if brief.voice_driven:
        lines.append(
            "Audio: spoken from <Audio 1>, lip movements synchronized to <Audio 1>."
        )
    elif brief.audio:
        lines.append(f"Audio: {brief.audio.strip()}")
    else:
        lines.append("No audio - silent video.")
    return "\n\n".join(lines)


# --- LLM 後端 ---------------------------------------------------------------

H3_JSON_DIRECTIVE = """You write production prompts for MiniMax H3, a model that
generates video and synchronized audio in one pass.

Return ONLY a JSON object with exactly these keys:
{
  "style_sentence": "one sentence: <style>: <subject>, <setting>, <lens>, <depth of field>, <grain>, <palette>",
  "beats": ["one concrete visible beat per element: subject + action + camera, one sentence each"],
  "audio": "what is heard in sync with the picture (ambient sounds + what is said). Empty string if silent.",
  "subject": "short appearance description of the main subject",
  "scene": "short description of location and lighting"
}

Rules:
- English only, no Chinese characters.
- 2 to 4 beats. Each beat is one visible action, no emotional or literary prose.
- Never add subjects the brief did not mention.
- If the brief is Chinese, translate the intent faithfully and keep proper names in both languages.
- If the brief has someone speaking, put the exact spoken line in "audio".
- No markdown fences, no commentary.
"""


@dataclass
class LLMConfig:
    """LLM 設定。provider = "openai"（OpenAI 相容）或 "gemini"。"""

    provider: str = "openai"
    model: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    timeout: int = 180

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """從環境變數組出設定（Colab 通常用 GOOGLE_API_KEY 或 OPENROUTER_API_KEY）。"""
        google = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if google:
            return cls(
                provider="gemini",
                model=os.environ.get("H3_LLM_MODEL", "gemini-2.5-flash"),
                api_key=google,
            )
        openrouter = os.environ.get("OPENROUTER_API_KEY")
        if openrouter:
            return cls(
                provider="openai",
                model=os.environ.get("H3_LLM_MODEL", "google/gemini-2.5-flash"),
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter,
            )
        ollama = os.environ.get("OLLAMA_URL")
        if ollama:
            return cls(
                provider="openai",
                model=os.environ.get("H3_LLM_MODEL", "qwen3:8b"),
                base_url=f"{ollama.rstrip('/')}/v1",
                api_key="ollama",
            )
        raise RuntimeError(
            "找不到 LLM 設定：請設 GOOGLE_API_KEY / GEMINI_API_KEY / "
            "OPENROUTER_API_KEY / OLLAMA_URL 其中一個"
        )


def _post_json(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    import requests

    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"LLM HTTP {response.status_code}: {response.text[:500]}")
    return response.json()


def llm_complete(config: LLMConfig, user_text: str) -> str:
    """呼叫 LLM 拿純文字回應。"""
    if config.provider == "gemini":
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{config.model}:generateContent?key={config.api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": user_text}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1200},
        }
        data = _post_json(url, payload, {"Content-Type": "application/json"}, config.timeout)
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "".join(part.get("text", "") for part in parts)

    url = f"{config.base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": user_text}],
        "temperature": 0.7,
        "max_tokens": 1200,
    }
    data = _post_json(url, payload, headers, config.timeout)
    return data["choices"][0]["message"]["content"]


def llm_json(config: LLMConfig, user_text: str) -> dict:
    """呼叫 LLM 並解析 JSON（容忍 ```json 包裹與前後雜訊）。"""
    raw = llm_complete(config, user_text).strip()
    cleaned = raw
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.lstrip().lower().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    cleaned = cleaned.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError(f"LLM 沒有回傳 JSON：{raw[:400]}")
    return json.loads(cleaned[start:end + 1])


def _brief_to_text(brief: Brief) -> str:
    return " | ".join(
        part for part in (
            brief.subject, brief.scene, brief.look, brief.camera,
            " ".join(brief.beats), brief.audio or "",
        ) if part
    )


def enhance_with_llm(brief: Brief, config: LLMConfig) -> Brief:
    """用 LLM 把 brief 強化成結構化 Brief（英文）。"""
    payload = llm_json(config, f"{H3_JSON_DIRECTIVE}\n\nBrief: {_brief_to_text(brief)}")
    beats = payload.get("beats") or []
    if isinstance(beats, str):
        beats = [beats]
    # 已經用 TTS 算好時間碼時，節拍數不能被 LLM 改掉（否則嘴型對不上）
    keep_timing = brief.timed_beats is not None
    return Brief(
        subject=payload.get("subject") or brief.subject,
        scene=payload.get("scene") or brief.scene,
        beats=list(brief.beats) if keep_timing else ([str(b) for b in beats] or list(brief.beats)),
        style=brief.style,
        look=payload.get("style_sentence") or brief.look,
        camera=brief.camera,
        audio=payload.get("audio") or brief.audio,
        voice_driven=brief.voice_driven,
        duration=brief.duration,
        negatives=brief.negatives,
        identity_lock=brief.identity_lock,
        template=brief.template,
        timed_beats=brief.timed_beats,
    )


# --- Director 後端 ----------------------------------------------------------

def enhance_with_director(
    brief: Brief,
    base_url: str,
    task_type: str = "t2v",
    llm_config: LLMConfig | None = None,
) -> str:
    """呼叫 ComfyUI 的 MiniMax H3 Director 提示強化路由。

    需要 ComfyUI 裝了 ComfyUI_MiniMaxH3_Director，且該節點有可用的 LLM。
    """
    import requests

    llm_config = llm_config or LLMConfig.from_env()
    response = requests.post(
        f"{base_url.rstrip('/')}/minimax/director/enhance",
        json={
            "prompt": _brief_to_text(brief),
            "task_type": task_type,
            "model": llm_config.model,
            "llm_url": "" if llm_config.provider == "gemini" else llm_config.base_url,
            "api_format": "openai",
            "api_key": llm_config.api_key,
            "output_language": "English",
        },
        timeout=llm_config.timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Director enhance HTTP {response.status_code}: {response.text[:400]}"
        )
    data = response.json()
    text = data.get("text") or data.get("prompt") or data.get("enhanced")
    if not text:
        raise RuntimeError(f"Director 沒有回傳提示詞：{json.dumps(data)[:400]}")
    return text


# --- 統一入口 ---------------------------------------------------------------

def enhance(
    brief: Brief,
    backend: str = "rule",
    llm_config: LLMConfig | None = None,
    base_url: str = "http://127.0.0.1:8188",
    task_type: str = "t2v",
) -> str:
    """把 Brief 變成最終 H3 提示組。

    backend: "rule"（離線編譯）/ "llm"（LLM 強化後再編譯）/ "director"（Director 節點）。
    """
    if backend == "rule":
        if has_cjk(_brief_to_text(brief)):
            raise ValueError(
                "純規則編譯器不做翻譯：中文 brief 請改用 backend='llm' 或 "
                "backend='director'，或直接給英文 beats。"
            )
        return compile_prompt(brief)
    if backend == "llm":
        llm_config = llm_config or LLMConfig.from_env()
        return compile_prompt(enhance_with_llm(brief, llm_config))
    if backend == "director":
        return enhance_with_director(brief, base_url, task_type, llm_config)
    raise ValueError(f"未知 backend: {backend}")



