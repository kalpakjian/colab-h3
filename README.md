# colab-h3 — Colab ⇄ ComfyUI ⇄ MiniMax H3 一條龍

把「Colab 開 ComfyUI 跑 MiniMax H3」這件事變成一條能重複跑的線：
裝機 → 權重 → **正確的 H3 提示組** → TTS → **聲音驅動** → 成片。

為什麼要有這個：本機 `D:\AI\ComfyUI` 有完整的 H3 產線（4090、Turbo、Director、
TRT VAE），但那是「有顯卡的人」的玩法。這裡做的是**雲端那條線**：沒有顯卡、
只有 Google AI Pro 的訂閱額度，也要把片子生出來。

這個套件解決三個常見的坑：

1. **不知道該下哪些模型** —— `config.py` 內建經 HuggingFace API 實查的檔名與大小。
2. **提示寫不好，AI 亂講話** —— `prompt_compiler.py` 產出官方格式的提示組
   （風格句 + 時間碼 Storyboard + Audio 行 + 禁止項），不是丟一句中文就生成。
3. **沒掛 TTS，不算聲音驅動** —— `tts_voice.py` 產出 32kHz wav，並用**真實音軌長度**
   回填時間碼，嘴型才會對上。

## 目錄

```
colab-h3/
  h3colab/
    config.py          模型檔案來源、算力檔案（Profile）
    prompt_compiler.py 提示強化：brief → 正確的 H3 提示組
    tts_voice.py       中文 TTS → wav + 真實音軌長度
    payloads.py        ComfyUI API 節點圖建構
    client.py          ComfyUI HTTP 客戶端
    setup_colab.py     冪等安裝 / 權重倉庫（Drive）
    cli.py             命令列入口
  tools/
    build_notebook.py  產生 ComfyUI_H3_Colab.ipynb
    verify_notebook.py 驗證 notebook（結構 + 內嵌模組一致 + 語法）
    e2e_local_test.py 端到端實測（需要本機 ComfyUI）
    dump_graph.py      印出工作流節點圖
    show_nodes.py      印出 /object_info 的節點 schema
  tests/
    run_tests.py       自我檢查（不需 pytest）
  ComfyUI_H3_Colab.ipynb
```

## 快速開始（Colab）

1. 打開 `ComfyUI_H3_Colab.ipynb`（在 Colab 選 **Runtime → Run all**）。
2. 依序跑完 8 格：環境 → 寫入套件 → 裝 ComfyUI/節點/權重 → 啟動 →
   提示強化 → TTS → 生成 → 存檔。

不用推檔到 GitHub：notebook 用 `%%writefile` 把整包程式碼內嵌進去。

## 命令列（本機或 Colab）

```powershell
cd D:\AI\colab-h3

python -m h3colab.cli doctor                    # 檢查 GPU / ComfyUI / 節點 / 權重
python -m h3colab.cli setup --profile l4 --download --store drive
python -m h3colab.cli prompt --brief "..." --backend rule
python -m h3colab.cli tts --script script.txt --out out/tts
python -m h3colab.cli video --script script.txt --profile l4_ref \
    --subject "a Taiwanese news anchor" --scene "a modern studio desk"
```

`video` 是端到端那條：台詞 → TTS → 時間碼 → 提示組 → r2v + `<Audio 1>` → 成片。

## 提示組長什麼樣

官方 H3 範例工作流的格式（`prompt_compiler` 就是照這個產）：

```
Realistic live-action cinematic look: a Taiwanese female news anchor in a navy blazer,
a modern broadcast studio, dark background, glowing desk, 50mm lens, shallow depth of field.

Storyboard:
[0s-4.61s] 各位觀眾晚安，今天我們要聊聊生成式AI。
[4.61s-9.22s] 這支影片完全是在雲端算力上生出來的。

Audio: clear female Mandarin voice, spoken from <Audio 1>, lip movements and jaw
synchronized to <Audio 1>, steady face, no extra talking.
No text, logos or watermarks.
```

三個關鍵點：

* **時間碼** —— 分鏡有時間範圍，H3 才知道什麼時候做什麼。
* **Audio 行** —— 說清楚聽到什麼、跟畫面什麼關係；`<Audio 1>` 是 r2v 的參考音訊槽。
* **禁止項** —— 避免水、字幕、換臉。

中文 brief 請用 `--backend llm`（需 `GOOGLE_API_KEY` / `OPENROUTER_API_KEY` /
`OLLAMA_URL` 其中一個）；純規則編譯器不做翻譯，會直接報錯而不是生出亂碼。

## 算力檔案

| profile | 解析度 | 幀數 | 步數 | UNET | 適用 |
|---|---|---|---|---|---|
| `l4` | 864×480（0.4MP） | 124（5.2s） | 8（Turbo） | `fl2va_pruned_w6a8` | L4 24GB 草稿 |
| `l4_ref` | 864×480 | 124 | 8 | `PRISM_w6a8`（ref2va） | L4 聲音驅動數字人 |
| `l4_ref_turbo` | 864×480 | 124 | 6（Turbo） | `ref2va_pruned_w6a8` | 本機 4090 已驗證配方 |
| `a100` | 1344×768（1.0MP） | 124 | 25（官方） | `fl2va_pruned_int8_convrot` | A100 40GB 正式 |
| `a100_ref` | 1344×768 | 124 | 25（官方） | `PRISM_int8` | A100 聲音驅動 |
| `t4` | 864×480 | 124 | 8 | `fl2va_pruned_w6a8` | **不建議**（見下） |

### 架構相容性（sol-attn）

sol-attn 的 Triton kernel 只支援 **SM86 / SM89 / SM90 / SM100 / SM120 / SM121**：

* **L4 = SM89 ✅** —— 官方支援，1.14–1.44× vs SageAttention，H3 MLP 峰值記憶體 −37%。
* **A100 = SM80 ❌** —— 不在清單，會自動 dense fallback（功能正常，只是沒加速）。
* **T4 = SM75 ❌** —— 同上，而且 H3 要 bf16，T4 會很慢。寧可花點數選 L4。

## 權重來源（實查）

| 用途 | 檔案 | 大小 | 來源 |
|---|---|---|---|
| UNET (t2v/i2v/fl2v) | `minimax_h3_fl2va_pruned_w6a8.safetensors` | 14.89 GB | `Comfy-Org/MiniMax-H3` |
| UNET (t2v/i2v/fl2v) | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 19.53 GB | 同上 |
| UNET (r2v) | `minimax_h3_ref2va_pruned_w6a8.safetensors` | 14.89 GB | 同上 |
| PRISM (r2v merge) | `Minimax-h3_Singularity_x_10Eros-Max_PRISM_w6a8.safetensors` | 14.81 GB | `OliviaRossi/…PRISM` |
| CLIP | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 14.61 GB | `Comfy-Org/MiniMax-H3` |
| Video VAE | `minimax_h3_video_vae_int8_convrot.safetensors` | 2.62 GB | 同上 |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | 0.56 GB | 同上 |
| Turbo LoRA | `minimax_h3_turbo_v4_step600_ema.safetensors` | 0.73 GB | `larryvrh/MiniMax-H3-Turbo-Lora` |
| ref2v Turbo LoRA | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 1.82 GB | `Comfy-Org/MiniMax-H3` |

> PRISM 是 **ref2va** merge（Singularity × 10Eros-Max），只能走 `r2v`（參考主體／
> 參考音訊），不能當 t2v/i2v 用。要純文字生成就用 `fl2va`。
>
> CLIPLoader 的 **type 必須是 `minimax`**（Qwen3-VL）。

## 權重倉庫（5TB 雲端硬碟）

```powershell
python -m h3colab.cli setup --profile l4 --store drive --download
```

* `--store drive`：下載到 `/content/drive/MyDrive/ComfyUI-H3`，再複製一份到
  `/content/ComfyUI/models` 供執行。**Drive 的 FUSE 讀取很慢**，直接從 Drive 載入
  模型會慢到不能忍，所以要先複製。
* `--store local`：直落本機，單次最快，但 runtime 一關就沒了。

要長期養多個版本（不同精度、不同 merge）就放 Drive；要馬上出片就 local。

## 實測結果（本機 4090，2026-10-26）

`tools\e2e_local_test.py` 的實跑結果（`examples/` 裡有成品）：

| 項目 | 值 |
|---|---|
| TTS | Edge TTS 繁中 2 句 → 4.608s + 4.512s = **9.120s**（32kHz 單聲道 wav） |
| 時間碼 | `[0s-4.6s]` / `[4.6s-9.1s]`（**依真實音軌長度回推**，不是均分） |
| 模型 | `minimax_h3_ref2va_pruned_int8_convrot` + `minimax_h3_ref2v_turbo_4step_v0.1` |
| 任務 | `r2v`，參考音訊走 `ref_audio_0` 槽（ComfyUI 的 autogrow 命名） |
| 產出 | 864×480、24fps、10.125s、**帶 32kHz 立體聲音軌** |
| 耗時 | **69.2 秒**（243 幀、6 步） |
| 音軌比對 | 輸出 mean −18.5dB / max −0.7dB vs 輸入 TTS mean −19.2dB / max −3.6dB → 確認是同一把聲音，不是雜訊 |

畫面檢查（`examples/demo_frame_2s.png`、`demo_frame_6s.png`）：第一句話時嘴型張開，
第二句話時**手真的舉起來指向後方螢幕**（正是分鏡描述），身份鎖定不變。

> 也就是說：「TTS → 時間碼 → 提示組 → 聲音驅動」這條線是**通的**，
> 不是紙上談兵。Colab 上換成 PRISM（ref2va merge）走同一條路。

## 已知限制

* **H3 需要 ComfyUI ≥ 0.30.0**（官方 H3 節點）。本套件 clone 的是 main 分支。
* **幀數必須在 17k+5 網格**（124 / 141 / … / 243 / 362）。`snap_length()` 會自動吸附。
* **ComfyUI 沒有 `/upload/audio` 端點**（只有 image/mask）→ 參考音訊必須直接
  複製進 `input/` 目錄。`ComfyClient(input_dir=...)` 已經處理這件事。
* **TTS 需要網路**（Edge TTS 是微軟線上服務）。離線可改 `--backend gtts`。
* **TTS 音軌長度要和影片長度對得上**：`video` 指令會依音軌長度自動算幀數。
* **分鏡描述用英文、台詞用中文**：提示詞負責「畫面長什麼樣」，聲音走 `<Audio 1>`。
  沒給 `--beat` 就會拿中文台詞頂替（Qwen3-VL 讀得懂，但官方測試路徑是英文）。
* 這個套件**不碰** `D:\AI\ComfyUI` 的環境與模型；兩邊完全獨立。
* **不要用 PowerShell 的 `Get-Content`/`Set-Content` 改這裡的 UTF-8 原始碼** ——
  Windows PowerShell 5.1 會用 GBK 解碼，中文全部變亂碼（已踩過一次）。

## 自我檢查

```powershell
python tests\run_tests.py          # 71 項（含對真實 ComfyUI 的活體驗證）
python tools\verify_notebook.py    # notebook 結構 + 內嵌模組一致
python tools\e2e_local_test.py     # 端到端（需要本機 ComfyUI 在 8188）
```
