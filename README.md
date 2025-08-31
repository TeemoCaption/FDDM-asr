# FDDM-ASR 專案

## 專案結構

```
FDDM-asr/
  ├── .vscode/            # VS Code 設定檔
  ├── configs/            # 模型與訓練設定檔
  │   ├── tokenizer_zhTW.yaml  # 繁體中文 tokenizer 設定
  │   └── diffusion.yaml       # 擴散模型訓練參數設定
  ├── data/               # 原始與處理後的資料
  ├── fddm/               # 核心模型程式碼
  │   └── sched/          # 擴散排程器
  │       └── diffusion_scheduler.py  # 離散擴散排程器實作
  ├── scripts/            # 資料處理與訓練腳本
  │   ├── preprocess.py        # 資料預處理腳本
  │   ├── tokenizer_train.py   # Tokenizer 訓練腳本
  │   └── sanity_check_scheduler.py  # 擴散排程器測試腳本
  ├── .gitignore         # Git 忽略設定
  ├── README.md          # 專案說明文件
  ├── requirements.txt   # Python 套件依賴
  └── 專案路線圖.txt      # 專案開發規劃與進度
```