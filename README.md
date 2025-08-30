# FDDM-ASR 專案

## 專案架構

```
fddm/
  configs/ (資料/模型/訓練/解碼設定yaml)
  data/ (原始與處理後資料)
  fddm/
    modules/ (Encoder, Decoder, Diffusion, Projections)
    losses/  (KL, L_cvb, L_fd)
    sched/   (β_t cosine排程、jumpy sampling)
    train.py / infer.py / eval.py
  scripts/ (資料前處理、tokenizer訓練、評測)
  logs/ ckpts/
```