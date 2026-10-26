"""ComfyUI HTTP API 客戶端：提交、輪詢、收結果、診斷錯誤。"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path


class ComfyError(RuntimeError):
    """ComfyUI 回報的錯誤（節點級錯誤會被整理成可讀訊息）。"""


class ComfyClient:
    """極簡 ComfyUI 客戶端（只用 requests）。

    input_dir：ComfyUI 的 input/ 目錄。**ComfyUI 沒有 /upload/audio 端點**
    （只有 /upload/image 與 /upload/mask），所以參考音訊必須直接放進
    input/ 目錄 —— Colab 上 ComfyUI 與本檔同機器，直接複製即可。
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8188",
        timeout: int = 60,
        input_dir: str | os.PathLike | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())
        self.input_dir = str(input_dir) if input_dir else None

    # --- 低階 ---------------------------------------------------------------

    def _get(self, path: str, **params):
        import requests

        response = requests.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        if response.status_code >= 400:
            raise ComfyError(f"GET {path} -> HTTP {response.status_code}: {response.text[:300]}")
        return response.json()

    def _post(self, path: str, payload: dict):
        import requests

        response = requests.post(
            f"{self.base_url}{path}", json=payload, timeout=self.timeout
        )
        if response.status_code >= 400:
            raise ComfyError(f"POST {path} -> HTTP {response.status_code}: {response.text[:600]}")
        return response.json()

    # --- 查詢 ---------------------------------------------------------------

    def system_stats(self) -> dict:
        return self._get("/system_stats")

    def object_info(self) -> dict:
        return self._get("/object_info")

    def available_nodes(self) -> set[str]:
        return set(self.object_info().keys())

    def queue_status(self) -> dict:
        return self._get("/queue")

    def history(self, prompt_id: str) -> dict:
        return self._get(f"/history/{prompt_id}")

    # --- 上傳 ---------------------------------------------------------------

    def _upload(self, endpoint: str, field: str, path, subfolder: str, mime: str) -> str:
        import requests

        with open(path, "rb") as handle:
            response = requests.post(
                f"{self.base_url}/upload/{endpoint}",
                files={field: (Path(path).name, handle, mime)},
                data={"type": "input", "overwrite": "true", "subfolder": subfolder},
                timeout=self.timeout,
            )
        if response.status_code >= 400:
            raise ComfyError(f"上傳失敗 HTTP {response.status_code}: {response.text[:300]}")
        data = response.json()
        name = data.get("name")
        if data.get("subfolder"):
            name = f"{data['subfolder']}/{name}"
        return name

    def upload_image(self, path: str | os.PathLike, subfolder: str = "") -> str:
        """把本機圖片上傳到 ComfyUI 的 input/，回傳可餵給節點的檔名。"""
        return self._upload("image", "image", path, subfolder, "image/png")

    def upload_audio(self, path: str | os.PathLike, subfolder: str = "") -> str:
        """把音訊放進 ComfyUI 的 input/（ComfyUI 沒有音訊上傳端點）。

        回傳可直接餵給節點的檔名（相對 input/）。
        """
        if not self.input_dir:
            raise ComfyError(
                "ComfyUI 沒有 /upload/audio 端點（只有 image/mask），參考音訊必須"
                "直接放進 input/ 目錄。請用 ComfyClient(input_dir=...) 指定。"
            )
        source = Path(path)
        dest_dir = Path(self.input_dir) / subfolder if subfolder else Path(self.input_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / source.name
        index = 1
        while dest.exists():
            dest = dest_dir / f"{source.stem}_{index}{source.suffix}"
            index += 1
        shutil.copy2(source, dest)
        return f"{subfolder}/{dest.name}" if subfolder else dest.name

    # --- 執行 ---------------------------------------------------------------

    def submit(self, graph: dict) -> str:
        payload = {"prompt": graph, "client_id": self.client_id}
        data = self._post("/prompt", payload)
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ComfyError(f"ComfyUI 沒給 prompt_id：{json.dumps(data)[:400]}")
        return prompt_id

    def wait(self, prompt_id: str, timeout: int = 1800, poll: int = 3) -> dict:
        """等到工作完成，回傳 history 條目；失敗會拋 ComfyError。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = self.history(prompt_id)
            entry = data.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                errors = [
                    m for m in status.get("messages", [])
                    if isinstance(m, list) and m and m[0] == "execution_error"
                ]
                if errors:
                    detail = errors[0][1]
                    raise ComfyError(
                        "ComfyUI 執行失敗："
                        f"{detail.get('node_type')} 節點 {detail.get('node_id')} "
                        f"{detail.get('exception_type')}: {detail.get('exception_message')}"
                    )
                if entry.get("outputs"):
                    return entry
            time.sleep(poll)
        raise ComfyError(f"等待 {prompt_id} 超時（{timeout}s）")

    def download_outputs(self, entry: dict, dest_dir: str | os.PathLike) -> list[str]:
        """把產出的影片抓回本機，回傳檔案路徑清單。"""
        import requests

        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []
        for node_output in (entry.get("outputs") or {}).values():
            for kind in ("videos", "images", "gifs", "audio"):
                for item in node_output.get(kind, []) or []:
                    filename = item.get("filename")
                    if not filename:
                        continue
                    subfolder = item.get("subfolder", "")
                    file_type = item.get("type", "output")
                    url = (
                        f"{self.base_url}/view?filename={filename}"
                        f"&subfolder={subfolder}&type={file_type}"
                    )
                    response = requests.get(url, timeout=self.timeout)
                    if response.status_code >= 400:
                        continue
                    target = dest / subfolder / filename if subfolder else dest / filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(response.content)
                    saved.append(str(target))
        return saved

    def run(self, graph: dict, timeout: int = 1800, dest_dir: str | os.PathLike | None = None):
        """提交 → 等完成 → （可選）抓檔。回傳 (history_entry, saved_paths)。"""
        prompt_id = self.submit(graph)
        entry = self.wait(prompt_id, timeout=timeout)
        saved = self.download_outputs(entry, dest_dir) if dest_dir else []
        return entry, saved
