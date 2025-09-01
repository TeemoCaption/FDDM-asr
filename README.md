# FDDM-ASR 專案 (Flow-based Discrete Diffusion Models for Automatic Speech Recognition)

基於離散擴散模型的自動語音辨識系統，支援繁體中文語音轉文字。專案實現了 FDDM 架構，包含特徵去相關損失 (L_fd)、先進的位置編碼技術 (RoPE)、跨模態條件控制 (FiLM)，以及高效的跳躍採樣器 (Jumpy Sampler)。

## 專案結構

```
  ├── configs/                    # 模型與訓練設定檔
  │   ├── fddm_zhTW_base.yaml     # 主要訓練配置（繁體中文）
  │   ├── fddm_sweep.yaml         # 超參數搜尋配置
  │   ├── tokenizer_zhTW.yaml     # Tokenizer 訓練設定
  │   └── diffusion.yaml          # 擴散模型參數設定
  ├── fddm/                       # 核心擴散模型程式碼
  │   └── sched/                  # 擴散排程器
  │       └── diffusion_scheduler.py  # 離散擴散排程器實作
  ├── losses/                     # 損失函數
  │   └── fddm_losses.py          # 特徵去相關損失 (L_fd)
  ├── models/                     # 模型定義
  │   ├── acoustic_encoder.py     # 聲學特徵編碼器 (WavLM)
  │   ├── denoise_decoder.py      # 去噪解碼器 (支援 RoPE, FiLM)
  │   └── projection.py           # 特徵投影模組
  ├── sampler/                    # 採樣器
  │   └── jumpy_sampler.py        # 跳躍採樣器 (支援精確/快速模式)
  ├── scripts/                    # 資料處理與訓練腳本
  │   ├── preprocess.py           # 資料預處理腳本
  │   ├── tokenizer_train.py      # Tokenizer 訓練腳本
  │   ├── sanity_check_scheduler.py  # 擴散排程器測試腳本
  │   └── sanity_forward.py       # 模型前向傳播測試腳本
  ├── .vscode/                    # VS Code 設定檔
  ├── .gitignore                 # Git 忽略設定
  ├── README.md                  # 專案說明文件
  ├── requirements.txt           # Python 套件依賴
  ├── train.py                   # 主要訓練腳本
  ├── inference.py               # 推論腳本
  └── 專案路線圖.txt             # 專案開發規劃與進度
```


主要依賴套件已更新至 `requirements.txt`：
- PyTorch 生態系統 (torch, torchaudio)
- Transformers 套件 (transformers, sentencepiece)
- 資料處理套件 (pandas, librosa, soundfile, datasets)
- 配置管理 (pyyaml)

## 快速開始

### 1. 安裝依賴
```bash
pip install -r requirements.txt
```

### 2. 資料前處理
```bash
python scripts/preprocess.py
```

### 3. 訓練 Tokenizer
```bash
python scripts/tokenizer_train.py --config configs/tokenizer_zhTW.yaml
```

### 4. 開始訓練
```bash
python train.py --config configs/fddm_zhTW_base.yaml
```

### 5. 超參數搜尋 (可選)
```bash
# 使用 sweep 配置進行超參數優化
python train.py --config configs/fddm_sweep.yaml
```

### 6. 模型推論
```bash
python inference.py --config configs/fddm_zhTW_base.yaml --checkpoint path/to/checkpoint
```

## 資料格式

前處理後的資料格式：
- **JSON 格式**: 包含 path, normalized_sentence, client_id, len_text, processed_path
- **分割方法**: 使用 Hugging Face Common Voice 內建分割（train/validation/test）
- **分割來源**: 官方 Common Voice 資料集的標準分割，確保資料品質和一致性
- **音檔格式**: 16kHz WAV 格式，單聲道

## 設定說明

- `fddm_zhTW_base.yaml`: 主要訓練配置，包含模型參數、優化器設定、資料路徑等
  - 支援 L_fd 損失配置：lambda_offdiag, n_step_fd, tau
  - 包含 inference 區段：T_infer, r, sampling_mode, posterior_mode
  - 位置編碼配置：pos_emb_type, use_film, rope_base

- `fddm_sweep.yaml`: 超參數搜尋配置檔案
  - 定義完整的 sweep 參數範圍和策略
  - 包含三階段搜尋建議：粗略搜尋 → 精細搜尋 → 消融研究
  - 支持的參數：lambda_offdiag, n_step_fd, tau, lr, batch_size 等

- `tokenizer_zhTW.yaml`: Tokenizer 訓練設定，包含詞彙表大小、訓練參數等
  - 注意特殊 Token 對應：pad_id, bos_id, eos_id, unk_id

- `diffusion.yaml`: 擴散模型專用參數設定
  - 擴散排程器參數、時間步配置等

## 注意事項

- 確保資料路徑正確設置在配置檔案中
- 訓練前先執行 sanity check 腳本驗證模型架構
- 推論時可選擇 exact 或 fast 模式根據需求
- 超參數搜尋建議從粗略範圍開始，逐步精細化
- 監控訓練日誌中的 loss_diff 和 loss_fd 分別值