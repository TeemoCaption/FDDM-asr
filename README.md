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

## 主要依賴套件

已更新至 `requirements.txt`：
- **PyTorch 生態系統**: torch, torchaudio, torchvision
- **Transformers 套件**: transformers, sentencepiece
- **資料處理套件**: pandas, librosa, soundfile, datasets
- **配置管理**: pyyaml
- **科學計算**: numpy, scipy

## 快速開始

### 1. 安裝依賴
```bash
pip install -r requirements.txt
```

### 2. 資料前處理
```bash
python scripts/preprocess.py
```
前處理會產生：
- `data/processed/train.json`
- `data/processed/validation.json`
- `data/processed/test.json`

### 3. 訓練 Tokenizer
```bash
python scripts/tokenizer_train.py --config configs/tokenizer_zhTW.yaml
```

### 4. 開始訓練
```bash
python train.py --config configs/fddm_zhTW_base.yaml
```

訓練時會自動：
- 載入 train/validation/test 資料集
- 每個 epoch 結束後計算 CER
- 自動保存最佳驗證權重
- 顯示完整的訓練日誌

### 5. 模型推論

#### 單一音檔推論
```bash
python inference.py --wav data/processed/clips/sample.wav --ckpt ckpts/fddm_zhTW_base/best_model.pt --main-config configs/fddm_zhTW_base.yaml --diffusion-config configs/diffusion.yaml --tokenizer data/tokenizer/zh-TW_A/spm_zhTW_A.model --T-infer 20 --r 5 --greedy
```

#### 批次推論
```bash
python inference.py --csv data/processed/test.csv --ckpt ckpts/fddm_zhTW_base/best_model.pt --main-config configs/fddm_zhTW_base.yaml --diffusion-config configs/diffusion.yaml --tokenizer data/tokenizer/zh-TW_A/spm_zhTW_A.model --out-json results/infer_results.json
```

## 訓練與評估

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

### fddm_zhTW_base.yaml (主要訓練配置)
```yaml
# 資料設定
data:
  train_json: data/processed/train.json
  val_json: data/processed/validation.json
  test_json: data/processed/test.json
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

## 🔧 進階功能

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

- **資料路徑**: 確保配置檔案中的資料路徑正確
- **權重載入**: 推論時優先使用 `best_model.pt`
- **記憶體管理**: 大模型訓練建議使用 GPU
- **超參數調整**: 建議從粗略範圍開始，逐步精細化
- **模型驗證**: 訓練前先執行 sanity check 腳本

## 📈 性能監控

訓練過程中監控的關鍵指標：
- `loss_diff`: Diffusion KL 損失
- `loss_fd`: 特徵去相關損失
- `w_t`: 時間權重係數
- `Train/Val/Test CER`: 字元錯誤率