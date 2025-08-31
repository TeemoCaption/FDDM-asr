# FDDM-ASR 專案

基於離散擴散模型的自動語音辨識系統，支援繁體中文語音轉文字。

## 專案結構

```
FDDM-asr/
  ├── .vscode/                    # VS Code 設定檔
  ├── configs/                    # 模型與訓練設定檔
  │   ├── fddm_zhTW_base.yaml     # 主要訓練配置（繁體中文）
  │   ├── tokenizer_zhTW.yaml     # Tokenizer 訓練設定
  │   └── diffusion.yaml          # 擴散模型參數設定
  ├── data/                       # 原始與處理後的資料
  │   ├── raw/                    # 原始下載資料
  │   └── processed/              # 處理後的資料
  │       ├── train.json          # 訓練集
  │       ├── dev.json            # 開發集
  │       ├── test.json           # 測試集
  │       ├── index.json          # 完整資料集
  │       └── clips/              # 處理後的音檔
  ├── fddm/                       # 核心擴散模型程式碼
  │   └── sched/                  # 擴散排程器
  │       └── diffusion_scheduler.py  # 離散擴散排程器實作
  ├── losses/                     # 損失函數
  │   └── fddm_losses.py          # 特徵去相關損失 (L_fd)
  ├── models/                     # 模型定義
  │   ├── acoustic_encoder.py     # 聲學特徵編碼器 (WavLM)
  │   ├── denoise_decoder.py      # 去噪解碼器
  │   └── projection.py           # 特徵投影模組
  ├── scripts/                    # 資料處理與訓練腳本
  │   ├── preprocess.py           # 資料預處理腳本
  │   ├── tokenizer_train.py      # Tokenizer 訓練腳本
  │   ├── sanity_check_scheduler.py  # 擴散排程器測試腳本
  │   └── sanity_forward.py       # 模型前向傳播測試腳本
  ├── .gitignore                 # Git 忽略設定
  ├── README.md                  # 專案說明文件
  ├── requirements.txt           # Python 套件依賴
  ├── train.py                   # 主要訓練腳本
  └── 專案路線圖.txt             # 專案開發規劃與進度
```

## 依賴套件

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

## 資料格式

前處理後的資料格式：
- **JSON 格式**: 包含 path, normalized_sentence, client_id, len_text, processed_path
- **分割方法**: 使用 Hugging Face Common Voice 內建分割（train/validation/test）
- **分割來源**: 官方 Common Voice 資料集的標準分割，確保資料品質和一致性
- **音檔格式**: 16kHz WAV 格式，單聲道

## 設定說明

- `fddm_zhTW_base.yaml`: 主要訓練配置，包含模型參數、優化器設定、資料路徑等
- `tokenizer_zhTW.yaml`: Tokenizer 訓練設定，包含詞彙表大小、訓練參數等
- `diffusion.yaml`: 擴散模型專用參數設定