# -*- coding: utf-8 -*-
"""
推論腳本（Jumpy Sampling + Greedy 解碼）
========================================
放置路徑：inference.py（與 train.py 並列）

用法範例
--------
# 重要提醒：請先確保已執行前處理腳本（scripts/preprocess.py）和訓練腳本（scripts/tokenizer_train.py）
# 並且已訓練好模型權重檔案

# 範例1：以單一音檔推論（預設 T=20, r=5, greedy=true）
python inference.py --wav data/processed/clips/sample.wav --ckpt ckpts/fddm_zhTW_base/ep001.pt --main-config configs/fddm_zhTW_base.yaml --diffusion-config configs/diffusion.yaml --tokenizer data/tokenizer/zh-TW_A/spm_zhTW_A.model --T-infer 20 --r 5 --greedy

# 範例2：以 CSV 批次推論（欄位須含 `path` 與可選 `text`）
python inference.py --csv data/processed/test.csv --ckpt ckpts/fddm_zhTW_base/ep001.pt --main-config configs/fddm_zhTW_base.yaml --diffusion-config configs/diffusion.yaml --tokenizer data/tokenizer/zh-TW_A/spm_zhTW_A.model --T-infer 20 --r 2 --out-json runs/infer_results.json

# 範例3：指定 GPU 設備並調整取樣參數
python inference.py --wav data/processed/clips/sample.wav --ckpt ckpts/fddm_zhTW_base/ep001.pt --main-config configs/fddm_zhTW_base.yaml --diffusion-config configs/diffusion.yaml --tokenizer data/tokenizer/zh-TW_A/spm_zhTW_A.model --gpu 0 --T-infer 50 --r 10 --seq-len 128

重要說明
--------
1) 本檔透過 ModelAdapter 統一去噪解碼器（DenoiseDecoder）的呼叫介面。
   若你的 models/denoise_decoder.py forward 參數命名不同，請於 ModelAdapter 調整。
2) DiffusionScheduler 須提供 alpha_bar（1..T_train）等資訊；
   若你目前的 scheduler API 不同，請於 _build_scheduler() 內調整。
3) 聲學條件 cond_c 以你現有的 WavLM-Large 封裝為準（models/acoustic_encoder.py）。
4) 為簡化示範，我提供最基本的「單檔音訊」與「CSV 批次」推論流程；
   真實專案可再加上並行 batch 化與 GPU 設備管理。
"""

from typing import Optional, Dict, Any, List, Tuple
import argparse
import json
import os

import torch
import torchaudio
import torch.nn.functional as F

# ==== 匯入專案內的模組====
from fddm.sched.diffusion_scheduler import DiscreteDiffusionScheduler  # 需存在 alpha_bar, T 等屬性
from models.acoustic_encoder import AcousticEncoder              # WavLM 封裝
from models.denoise_decoder import DenoisingTransformerDecoder   # 去噪解碼器（預測 x̂₀ 分布）
# --------------------------------------------------------------

from sampler.jumpy_sampler import DiffusionJumpySampler, ModelAdapter  # 本次新增


# ==========================
# 基礎工具
# ==========================
def load_wav(path: str, target_sr: int = 16000, device: Optional[torch.device] = None) -> torch.Tensor:
    """讀取單一 wav 檔並轉為目標取樣率（單聲道）"""
    wav, sr = torchaudio.load(path)  # [C, T]
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    wav = wav.to(device or torch.device("cpu"))
    return wav  # [1, T]


def build_device(gpu: Optional[int]) -> torch.device:
    if gpu is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# 建立模組
# ==========================
def _build_scheduler(diffusion_config_path: str, main_config_path: str, device: torch.device) -> DiscreteDiffusionScheduler:
    """
    讀取 diffusion.yaml 和主配置檔案，建立 DiscreteDiffusionScheduler。
    """
    # 讀取擴散配置
    import yaml
    with open(diffusion_config_path, 'r', encoding='utf-8') as f:
        diffusion_config = yaml.safe_load(f)

    # 讀取主配置以獲取 vocab_size
    with open(main_config_path, 'r', encoding='utf-8') as f:
        main_config = yaml.safe_load(f)

    # 建立 DiscreteDiffusionScheduler，參數從對應配置檔案讀取
    scheduler = DiscreteDiffusionScheduler(
        K=main_config['data']['vocab_size'],  # 從主配置獲取詞彙表大小
        T=diffusion_config['diffusion']['T'],  # 從擴散配置獲取總步數
        device=device,
        beta_max=diffusion_config['diffusion']['beta_max']  # 從擴散配置獲取噪聲上限
    )
    return scheduler


def _build_acoustic_encoder(main_config_path: str, device: torch.device) -> AcousticEncoder:
    """
    建立並凍結 WavLM-Large 聲學編碼器。
    從主配置檔案讀取編碼器參數。
    """
    # 讀取主配置檔案
    import yaml
    with open(main_config_path, 'r', encoding='utf-8') as f:
        main_config = yaml.safe_load(f)
    
    # 從配置檔案獲取編碼器參數
    encoder_config = main_config['model']['encoder']
    encoder = AcousticEncoder(
        wavlm_name=encoder_config['wavlm_name'],  # WavLM 模型名稱
        freeze=encoder_config['freeze'],          # 是否凍結參數
        d_model=main_config['model']['d_model'],  # 模型維度
        proj=encoder_config['proj'],              # 投影類型
        pooling=encoder_config['pooling']         # 池化方式
    )
    if hasattr(encoder, "to"):
        encoder = encoder.to(device)
    encoder.eval()
    # 如果配置中設定不凍結，則額外確保凍結
    if encoder_config['freeze']:
        for p in encoder.parameters():
            p.requires_grad = False
    return encoder


def _build_decoder(ckpt_path: str, main_config_path: str, device: torch.device) -> DenoisingTransformerDecoder:
    """
    建立去噪解碼器並載入權重。
    從主配置檔案動態讀取模型參數。
    """
    # 讀取主配置檔案
    import yaml
    with open(main_config_path, 'r', encoding='utf-8') as f:
        main_config = yaml.safe_load(f)

    # 從配置檔案獲取模型參數
    decoder = DenoisingTransformerDecoder(
        vocab_size=main_config['data']['vocab_size'],  # 詞彙表大小
        d_model=main_config['model']['d_model'],      # 模型維度
        nhead=main_config['model']['nhead'],          # 多頭注意力頭數
        num_layers=main_config['model']['num_layers'], # Transformer層數
        dim_ff=main_config['model']['dim_ff'],        # 前饋網路維度
        dropout=main_config['model']['dropout'],      # Dropout機率
        max_len=1024,  # 最大序列長度（可設定為配置項）
        pad_id=main_config['data']['pad_id']          # Padding token ID
    )
    sd = torch.load(ckpt_path, map_location="cpu")
    # train.py 儲存的格式包含多個模型的 state_dict
    if "decoder" in sd:
        # 如果是完整的訓練檢查點，只載入 decoder 部分
        decoder_sd = sd["decoder"]
    elif "state_dict" in sd:
        # 如果是舊格式的檢查點
        decoder_sd = sd["state_dict"]
    else:
        # 如果直接是 state_dict
        decoder_sd = sd
    decoder.load_state_dict(decoder_sd, strict=False)
    decoder = decoder.to(device)
    decoder.eval()
    return decoder


# ==========================
# 聲學條件 & Tokenizer
# ==========================
def extract_condition(encoder: AcousticEncoder, wav: torch.Tensor) -> torch.Tensor:
    """
    將原始波形（[1, T]）轉成條件 c（[1, N, D]）。
    依你的 AcousticEncoder 輸出為主（例如 WavLM 特徵）。
    """
    with torch.no_grad():
        # AcousticEncoder.forward() 回傳 (feats, feat_mask, pooled)
        # 我們只需要 feats 作為條件特徵 [1, N, D]
        feats, feat_mask, pooled = encoder(wav)
    return feats


class SimpleTokenizerAdapter:
    """
    最小 tokenizer 包裝：需提供 encode / decode。
    你若用 sentencepiece 或 huggingface tokenizer，請在此替換成實際載入方式。
    """
    def __init__(self, spm_path: str):
        try:
            import sentencepiece as spm
        except ImportError:
            raise RuntimeError("請先安裝 sentencepiece，或改用你專案中的 tokenizer 讀取法。")
        self.spm = spm.SentencePieceProcessor(model_file=spm_path)

    @property
    def vocab_size(self) -> int:
        return self.spm.vocab_size()

    def encode(self, text: str) -> List[int]:
        return self.spm.encode(text, out_type=int)

    def decode(self, ids: List[int]) -> str:
        return self.spm.decode(ids)


# ==========================
# 推論主流程（單檔）
# ==========================
@torch.no_grad()
def infer_one(
    wav_path: str,
    scheduler: DiscreteDiffusionScheduler,
    encoder: AcousticEncoder,
    decoder: DenoisingTransformerDecoder,
    tokenizer: SimpleTokenizerAdapter,
    T_infer: int,
    r: int,
    greedy: bool,
    posterior_mode: str,
    seq_len: Optional[int],
    device: torch.device,
) -> Dict[str, Any]:
    # 1) 準備聲學條件
    wav = load_wav(wav_path, target_sr=16000, device=device)  # [1, T]
    cond_c = extract_condition(encoder, wav)                  # [1, N, D]

    # 2) 準備取樣器
    K = tokenizer.vocab_size
    T_train = getattr(scheduler, "T", 200)  # 若 diffusion.yaml 內 T 為 200，這裡會取到
    if seq_len is None:
        # 粗略設定序列長度：可改為根據 cond_c 長度與你資料統計決定
        seq_len = 64

    sampler = DiffusionJumpySampler(
        scheduler=scheduler,
        decoder=decoder,
        K=K,
        T_train=T_train,
        T_infer=T_infer,
        r=r,
        greedy=greedy,
        posterior_mode=posterior_mode,
        temperature=1.0,
        device=device,
    )

    # 3) Jumpy Sampling
    x0_idx, p_x0_last = sampler.sample(cond_c=cond_c, seq_len=seq_len, init="uniform")
    ids = x0_idx[0].tolist()
    text = tokenizer.decode(ids)

    return {
        "wav": wav_path,
        "text": text,
        "ids": ids,
        "T_infer": T_infer,
        "r": r,
        "greedy": greedy,
        "posterior_mode": posterior_mode,
    }


# ==========================
# CLI
# ==========================
def parse_args():
    p = argparse.ArgumentParser(description="FDDM-ASR Inference (Jumpy Sampling + Greedy)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--wav", type=str, help="單一音檔路徑（.wav）")
    src.add_argument("--csv", type=str, help="批次推論 CSV（需含欄位 `path`）")

    p.add_argument("--ckpt", type=str, required=True, help="去噪解碼器權重（.pt/.ckpt）")
    p.add_argument("--main-config", type=str, required=True, help="主要配置檔案（configs/fddm_zhTW_base.yaml）")
    p.add_argument("--diffusion-config", type=str, required=True, help="configs/diffusion.yaml")
    p.add_argument("--tokenizer", type=str, required=True, help="SentencePiece 模型路徑（.model）")

    # 取樣參數
    p.add_argument("--T-infer", type=int, default=20, help="推論使用的總步數（建議 10~50）")
    p.add_argument("--r", type=int, default=5, help="每次跳步長 Δ（例如 2 或 5）")
    p.add_argument("--greedy", action="store_true", help="使用 Greedy 解碼（argmax）")
    p.add_argument("--posterior-mode", type=str, default="average", choices=["average", "max"],
                   help="平均後驗（average）或最大後驗（max）近似")
    p.add_argument("--seq-len", type=int, default=None, help="輸出序列長度上限（若不設則取 64）")

    # 其他
    p.add_argument("--gpu", type=int, default=None, help="使用的 GPU 編號；不設則自動選擇")
    p.add_argument("--out-json", type=str, default=None, help="將結果輸出到 JSON")
    return p.parse_args()


def main():
    args = parse_args()
    device = build_device(args.gpu)

    # 建立模組
    scheduler = _build_scheduler(args.diffusion_config, args.main_config, device)
    encoder = _build_acoustic_encoder(args.main_config, device)  # 加入 main_config 參數
    decoder = _build_decoder(args.ckpt, args.main_config, device)
    tokenizer = SimpleTokenizerAdapter(args.tokenizer)

    results: List[Dict[str, Any]] = []

    if args.wav:
        res = infer_one(
            wav_path=args.wav,
            scheduler=scheduler,
            encoder=encoder,
            decoder=decoder,
            tokenizer=tokenizer,
            T_infer=args.T_infer,
            r=args.r,
            greedy=args.greedy,
            posterior_mode=args.posterior_mode,
            seq_len=args.seq_len,
            device=device,
        )
        results.append(res)
        print(json.dumps(res, ensure_ascii=False, indent=2))

    else:
        import pandas as pd
        df = pd.read_csv(args.csv)
        for _, row in df.iterrows():
            path = row["path"]
            try:
                res = infer_one(
                    wav_path=path,
                    scheduler=scheduler,
                    encoder=encoder,
                    decoder=decoder,
                    tokenizer=tokenizer,
                    T_infer=args.T_infer,
                    r=args.r,
                    greedy=args.greedy,
                    posterior_mode=args.posterior_mode,
                    seq_len=args.seq_len,
                    device=device,
                )
            except Exception as e:
                res = {"wav": path, "error": str(e)}
            results.append(res)
            print(json.dumps(res, ensure_ascii=False))

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
