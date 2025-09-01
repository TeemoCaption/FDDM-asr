# FDDM-ASR 專案 (Flow-based Discrete Diffusion Models for Automatic Speech Recognition)

基於離散擴散模型的自動語音辨識系統，支援繁體中文語音轉文字。專案實現了 FDDM 架構，包含特徵去相關損失 (L_fd)、先進的位置編碼技術 (RoPE)、跨模態條件控制 (FiLM)，以及高效的跳躍採樣器 (Jumpy Sampler)。

## 專案結構

```
├── data/                      # 資料目錄
│   ├── raw/                   # 原始資料集目錄（由 .gitkeep 保留結構）
│   │   ├── cv-corpus-22.0-2025-06-20/  # Common Voice 資料集
│   │   │   ├── zh-TW/         # 繁體中文資料
│   │   │   │   ├── clips/     # 音檔資料夾
│   │   │   │   ├── train.tsv  # 訓練集索引
│   │   │   │   ├── dev.tsv    # 開發集索引
│   │   │   │   └── test.tsv   # 測試集索引
│   │   │   ├── en/           # 英文資料
│   │   │   └── ... (其他語言)
│   └── processed/             # 處理後的資料目錄
│       ├── clips/             # 轉檔後的 16kHz WAV 檔案
│       ├── zh-TW_train.json   # 繁體中文訓練集索引
│       ├── zh-TW_dev.json     # 繁體中文開發集索引
│       ├── zh-TW_test.json    # 繁體中文測試集索引
│       └── ... (其他語言的索引檔案)
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
│   ├── evaluate.py             # 模型評估工具
│   └── projection.py           # 特徵投影模組
├── sampler/                    # 採樣器
│   └── jumpy_sampler.py        # 跳躍採樣器 (支援精確/快速模式)
├── scripts/                    # 資料處理與訓練腳本
│   ├── preprocess.py           # 資料預處理腳本
│   ├── tokenizer_train.py      # Tokenizer 訓練腳本
│   ├── sanity_check_scheduler.py  # 擴散排程器測試腳本
│   └── sanity_forward.py       # 模型前向傳播測試腳本
├── ckpts/                     # 模型檢查點目錄
├── .vscode/                    # VS Code 設定檔
├── .gitignore                 # Git 忽略設定（保留 data/raw 結構）
├── README.md                  # 專案說明文件
├── requirements.txt           # Python 套件依賴
├── train.py                   # 主要訓練腳本
├── inference.py               # 推論腳本
└── 專案路線圖.txt             # 專案開發規劃與進度
```

## 主要依賴套件

已更新至 `requirements.txt`：
- **PyTorch 生態系統**: torch, torchaudio, torchvision
- **Transformers 套件**: transformers, sentencepiece
- **資料處理套件**: pandas, librosa, soundfile, datasets
- **配置管理**: pyyaml
- **科學計算**: numpy, scipy

## 資料準備

### 1. 下載 Common Voice 資料集

從 [Common Voice](https://commonvoice.mozilla.org/) 下載您需要的語言資料集，例如：
- `cv-corpus-22.0-2025-06-20.tar.gz` (繁體中文)
- `cv-corpus-22.0-2025-06-20.tar.gz` (英文)
- 其他語言版本

### 2. 放置資料集

**將下載的壓縮檔案解壓後，完整拖曳資料集資料夾到專案的 `data/raw/` 目錄下**：

```
data/raw/cv-corpus-22.0-2025-06-20/
├── zh-TW/           # 繁體中文資料
│   ├── clips/       # 音檔
│   ├── train.tsv    # 訓練集索引
│   ├── dev.tsv      # 開發集索引
│   └── test.tsv     # 測試集索引
├── en/             # 英文資料
├── ja/             # 日文資料
└── ... (其他語言)
```

**⚠️ 重要提醒**
- 將**整個解壓後的資料集資料夾**拖到 `data/raw/` 目錄中
- 不要修改資料集的原始結構
- 腳本會自動偵測所有語言資料夾

### 3. 資料前處理

執行自動化前處理腳本：
```bash
python scripts/preprocess.py --dataset_name "cv-corpus-22.0-2025-06-20"
```

腳本會自動：
- 偵測所有語言資料夾
- 處理所有分割 (train/dev/test)
- 轉換音檔為 16kHz WAV 格式
- 產生標準化的索引檔案
- 自動進行時長過濾 (0.1s-30s)
- **統計總錄製時數**：完成後顯示資料集總時長

### 記憶體優化模式

對於大型資料集，建議使用記憶體優化模式：
```bash
# 小型資料集（預設模式）
python scripts/preprocess.py --dataset_name "cv-corpus-22.0-2025-06-20"

# 大型資料集（記憶體優化模式）
python scripts/preprocess.py --dataset_name "cv-corpus-22.0-2025-06-20" --use_memory_optimized --batch_size 500

# 指定特定語言
python scripts/preprocess.py --dataset_name "cv-corpus-22.0-2025-06-20" --language "zh-TW" --use_memory_optimized --batch_size 1000
```

**記憶體優化參數說明：**
- `--use_memory_optimized`：啟用記憶體優化模式（批次處理 + 串流輸出）
- `--batch_size`：批次處理大小（預設 1000，較小的值使用更少記憶體但處理較慢）
- `--language`：指定處理特定語言（留空則自動偵測所有語言）

**記憶體使用比較：**
- **標準模式**：記憶體使用量與資料集大小成正比，適合小於 10,000 個檔案的資料集
- **優化模式**：記憶體使用量固定在批次大小範圍內，適合任何大小的資料集

**處理輸出範例：**
```
批次處理語言：zh-TW（批次大小：500）
語言目錄：data/raw/cv-corpus-22.0-2025-06-20/zh-TW
=== 批次處理 zh-TW - train ===
處理批次 0-500（總共 15230）
已寫入批次資料，釋放記憶體（當前批次記錄數：0）
處理批次 500-1000（總共 15230）
...
完成 zh-TW - train: 音檔 15230 筆 | 索引：data/processed/zh-TW_train.json / data/processed/zh-TW_train.csv | 總時長：45.67 小時
完成語言 zh-TW: 總時長 45.67 小時
資料集總時長：45.67 小時

總錄製時數：45.67 小時
全部完成。你可以在 data/processed/ 找到索引與轉檔後音檔。
```

### 4. Git 設定

專案已配置 `.gitignore`：
- ✅ 保留 `data/raw/` 資料夾結構
- ❌ 忽略 `data/raw/` 內的所有資料集內容
- ❌ 忽略 `data/processed/` 處理後的檔案

這樣可以安全地上傳專案程式碼而不會意外上傳大型資料檔案。

訓練時會自動：
- 載入 train/validation/test 資料集
- 每個 epoch 結束後計算 CER
- 自動保存最佳驗證權重
- 顯示完整的訓練日誌

## 快速開始

### 1. 安裝依賴
```bash
pip install -r requirements.txt
```

### 2. 準備資料集
按照上方「資料準備」章節下載並放置 Common Voice 資料集到 `data/raw/` 目錄。

### 3. 自動化前處理

#### 標準模式（適合小型資料集）
```bash
python scripts/preprocess.py --dataset_name "cv-corpus-22.0-2025-06-20"
```

#### 記憶體優化模式（適合大型資料集）
```bash
python scripts/preprocess.py --dataset_name "cv-corpus-22.0-2025-06-20" --use_memory_optimized --batch_size 500
```

### 4. 訓練 Tokenizer
```bash
python scripts/tokenizer_train.py --config configs/tokenizer_zhTW.yaml
```

### 5. 開始訓練
```bash
python train.py --config configs/fddm_zhTW_base.yaml
```

訓練時會自動：
- 載入 train/validation/test 資料集
- 每個 epoch 結束後計算 CER
- 自動保存最佳驗證權重
- 顯示完整的訓練日誌

### 6. 模型推論

#### 單一音檔推論
```bash
python inference.py --wav data/processed/clips/zh-TW_test_sample.wav --ckpt ckpts/fddm_zhTW_base/best_model.pt --main-config configs/fddm_zhTW_base.yaml --diffusion-config configs/diffusion.yaml --tokenizer data/tokenizer/zh-TW_A/spm_zhTW_A.model --T-infer 20 --r 5 --greedy
```

#### 批次推論
```bash
python inference.py --csv data/processed/zh-TW_test.csv --ckpt ckpts/fddm_zhTW_base/best_model.pt --main-config configs/fddm_zhTW_base.yaml --diffusion-config configs/diffusion.yaml --tokenizer data/tokenizer/zh-TW_A/spm_zhTW_A.model --out-json results/infer_results.json
```

### 損失函數
訓練使用兩種主要損失：
1. **Diffusion KL 損失**: KL[q(xt-1|xt,x0) || p_theta(xt-1|xt,c)]
2. **特徵去相關損失 (L_fd)**: 論文 3.3 節的跨模態對齊損失

### 評估指標
- **CER (Character Error Rate)**: 字元錯誤率，語音辨識標準指標
- **訓練 CER**: 使用訓練資料樣本計算
- **驗證 CER**: 模型選擇和早停依據
- **測試 CER**: 最終模型性能評估

### 權重管理
訓練完成後會產生：
```
ckpts/fddm_zhTW_base/
├── ep001.pt        # 第1個epoch權重
├── ep002.pt        # 第2個epoch權重
├── ...
├── ep010.pt        # 最後一個epoch權重
└── best_model.pt   # 最佳驗證CER權重（推論使用這個！）
```

## 設定說明

> fddm_zhTW_base.yaml (主要訓練配置)
```yaml
# 資料設定
data:
  train_json: data/processed/zh-TW_train.json      # 更新路徑
  val_json: data/processed/zh-TW_dev.json           # 更新路徑
  test_json: data/processed/zh-TW_test.json         # 更新路徑
  vocab_size: 8000
  max_len: 128

# 模型設定
model:
  d_model: 512
  nhead: 8
  num_layers: 6
  dropout: 0.1

# 擴散設定
diffusion:
  T: 200
  beta_max: 0.01

# L_fd 損失設定
lfd:
  lambda_offdiag: 5.0e-3  # 去相關懲罰強度
  n_step_fd: 4            # 每4步加入L_fd
  tau: 1.0               # L_fd總體縮放係數

# 訓練設定
optim:
  lr: 2.0e-4
  batch_size: 4
  num_epochs: 10

# 日誌設定
log:
  ckpt_dir: ckpts/fddm_zhTW_base
  log_every: 50
```

### fddm_sweep.yaml (超參數搜尋)
支援的搜尋參數：
- `lambda_offdiag`: [1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
- `n_step_fd`: [1, 2, 4, 8, 16]
- `tau`: [0.1, 0.5, 1.0, 2.0, 5.0]
- `lr`: [1e-4, 2e-4, 5e-4, 1e-3]
- `batch_size`: [2, 4, 8, 16]

### 推論設定
```yaml
inference:
  T_infer: 20          # 推論步數
  r: 2                 # 跳步長度
  sampling_mode: "exact"  # exact/fast
  posterior_mode: "average"  # average/max
```

## 訓練輸出範例

```
Epoch 1
step=50 loss_diff=2.1456 loss_fd=0.0345 w_t=0.9876 total_loss=2.1801
step=100 loss_diff=1.9876 loss_fd=0.0289 w_t=0.9765 total_loss=2.0165
...
Epoch 1 Train CER: 0.4567
Epoch 1 Validation CER: 0.4234
New best validation CER: 0.4234 at epoch 1
Saved best model to: ckpts/fddm_zhTW_base/best_model.pt
Epoch 1 Test CER: 0.4456

==================================================
TRAINING COMPLETED!
Best validation CER: 0.4234 (Epoch 1)
Best model saved at: ckpts/fddm_zhTW_base/best_model.pt
==================================================
```

## 進階功能

### 位置編碼選項
- **RoPE**: 旋轉位置編碼，支援任意長度序列
- **Sinusoidal**: 標準正弦位置編碼
- **Learned**: 可學習的位置嵌入

### 跨模態條件控制
- **FiLM**: 特徵智慧線性調製
- **Cross-Attention**: 標準跨注意力機制

### 採樣器模式
- **精確模式 (exact)**: 嚴格遵循論文 Algorithm 2
- **快速模式 (fast)**: 計算效率優化版本

## 注意事項

- **資料集放置**: 將 Common Voice 資料集解壓後完整拖曳到 `data/raw/` 目錄中
- **自動語言偵測**: 腳本會自動偵測資料集中的所有語言資料夾
- **檔案路徑**: 處理後的檔案會包含語言前綴，例如 `zh-TW_train.json`
- **記憶體管理**: 大型資料集建議使用 `--use_memory_optimized` 參數
- **批次處理**: 支援分批處理以避免記憶體溢出
- **串流輸出**: 處理過程中即時寫入檔案，減少記憶體佔用
- **超參數調整**: 建議從粗略範圍開始，逐步精細化
- **模型驗證**: 訓練前先執行 sanity check 腳本
- **Git 管理**: 已配置 `.gitignore` 保留 `data/raw/` 結構但忽略內容
- **Diffusion KL 損失**: KL[q(xt-1|xt,x0) || p_theta(xt-1|xt,c)]
- **特徵去相關損失 (L_fd)**: 論文 3.3 節的跨模態對齊損失
- **時間權重係數**: w_t
- **訓練 CER**: 使用訓練資料樣本計算
- **驗證 CER**: 模型選擇和早停依據
- **測試 CER**: 最終模型性能評估