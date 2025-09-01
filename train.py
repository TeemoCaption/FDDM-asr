# -*- coding: utf-8 -*-
"""
FDDM-asr 訓練主程式（最小可跑骨架）
- 將 Step 2（AcousticEncoder）、Step 3（Decoder）、Step 4（L_fd）
  與你的 DiscreteDiffusionScheduler 串接。
- 依論文第 3.1、3.2、3.3 節實作訓練流程；KL(D) 由 scheduler 提供。

使用方式：
    python train.py --config configs/fddm_zhTW_base.yaml

你需要先準備：
- 你的 tokenizer 輸出（targets：x0；pad_id 要與 config 對齊）
- 你的 DiscreteDiffusionScheduler，需具備以下介面（請依你 repo 實際路徑修正 import）：
    class DiscreteDiffusionScheduler:
        def __init__(self, T: int, ...):
            self.T = T
            self.betas = ...           # β_t
            self.alphas_cumprod = ...  # ∏_{s=1}^t (1-β_s)
        def sample_q(self, x0, t):
            \"\"\"前向擴散：給定 x0 與步數 t，回傳 xt（依 (1) q(xt|x0)）。\"\"\"
        def kl_term(self, xt, x0, logits_x0, t):
            \"\"\"Eq.(6)：KL[q(xt-1|xt,x0) || p_theta(xt-1|xt,c)]，
            其中 p_theta 需以 logits_x0（對 x_0 的分佈）構成 posterior q(xt-1|xt, x_hat0)。\"\"\"
        def w_t(self, t):
            \"\"\"論文 (13) 的 w_t = ∏_{s=1}^t (1-β_s)。\"\"\"

- 若你的方法命名不同，請在下方 Adapter 區塊修改對接。
"""
from __future__ import annotations
import os
import argparse
import random
import yaml
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from torch import amp  # 提供 amp.GradScaler 與 amp.autocast（跨裝置統一接口）

# ====== 匯入本專案模組 ======
from models.acoustic_encoder import AcousticEncoder
from models.denoise_decoder import DenoisingTransformerDecoder
from models.projection import SpeechProjector, TextEmbedding, TextProjector
# 匯入評估函數
from models.evaluate import (
    calculate_cer, logits_to_text,
    evaluate_validation_loss, evaluate_cer_with_full_sampling,
    evaluate_cer_with_jumpy_sampling, evaluate_cer_with_multi_sample
)
from losses.fddm_losses import lfd_loss

# 進度條（若沒安裝也能安全退回）
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ====== 重要提醒：Tokenizer 特殊 Token 對應 ======
# 在導入真實資料時，請確保以下 token 與 tokenizer 設定一致：
# - pad_id: padding token ID（通常為 0）
# - bos_id: beginning of sequence token ID（如果使用）
# - eos_id: end of sequence token ID（如果使用）
# - unk_id: unknown token ID（處理 OOV 詞彙）
# 
# 範例配置檔設定：
# data:
#   pad_id: 0      # <pad> token
#   bos_id: 1      # <bos> token（可選）
#   eos_id: 2      # <eos> token（可選）
#   unk_id: 3      # <unk> token
#   vocab_size: 8000  # 包含特殊 token 的總詞彙量

# === 匯入 Scheduler（依實際路徑修正） ===
_SCHED_IMPORT_OK = False
try:
    from fddm.sched.diffusion_scheduler import DiscreteDiffusionScheduler  # 修正為實際模組路徑
    _SCHED_IMPORT_OK = True
except Exception as e:
    print(f"無法匯入 DiscreteDiffusionScheduler: {e}")
    pass

# ============ 資料集骨架（請換成你的前處理） ============
class CVZhTWDataset(Dataset):
    """從 JSON 索引載入真實資料的 Dataset：
    - 從 data/processed/train.json 等載入樣本
    - 載入對應的音頻檔案和文本
    - 使用 tokenizer 將文本轉換為 token IDs
    """
    def __init__(self, json_file: str, tokenizer_vocab_path: str, max_len: int, pad_id: int, bos_id: int = None, eos_id: int = None):
        super().__init__()
        import json
        import sentencepiece as spm
        import librosa
        import soundfile as sf
        
        # 載入 JSON 資料
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.max_len = max_len
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        
        # 載入 tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_vocab_path)
        
        # 設定最大音頻長度（預設 20 秒 * 16kHz）
        self.max_audio_samples = 20 * 16000
        
        # 過濾有效樣本（有音頻檔案的）
        self.valid_indices = []
        for i, item in enumerate(self.data):
            processed_path = item.get('processed_path')
            if processed_path and os.path.exists(processed_path):
                self.valid_indices.append(i)
        
        print(f"載入 {len(self.valid_indices)} 個有效樣本從 {json_file}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        data_idx = self.valid_indices[idx]
        item = self.data[data_idx]
        
        # 載入音頻
        import librosa
        import soundfile as sf
        wav, sr = librosa.load(item['processed_path'], sr=16000)
        wav = torch.tensor(wav, dtype=torch.float32)
        
        # 對齊音頻長度（截斷或零填充）
        if len(wav) > self.max_audio_samples:
            wav = wav[:self.max_audio_samples]
        elif len(wav) < self.max_audio_samples:
            padding = torch.zeros(self.max_audio_samples - len(wav))
            wav = torch.cat([wav, padding], dim=0)
        
        # 文本 → token ids
        text = item['normalized_sentence']
        tokens = self.tokenizer.encode(text)
        
        # 添加特殊 token
        if self.bos_id is not None:
            tokens = [self.bos_id] + tokens
        if self.eos_id is not None:
            tokens = tokens + [self.eos_id]
        
        # 截斷或填充到 max_len
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens += [self.pad_id] * (self.max_len - len(tokens))
        
        x0 = torch.tensor(tokens, dtype=torch.long)
        return wav, x0
        
# ============ CER 評估指標實作已移至 models/evaluate.py ============
@dataclass
class Config:
    seed: int
    data: dict
    model: dict
    diffusion: dict
    inference: dict  # 添加 inference 字段
    optim: dict
    lfd: dict
    log: dict

# ============ 與 Scheduler 的介面對接（Adapter） ============
class SchedulerAdapter:
    def __init__(self, scheduler):
        self.sch = scheduler

    def sample_q(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """前向擴散：x0 → xt（依 multinomial diffusion）"""
        B, L = x0.shape
        vocab_size = self.sch.K
        x0_onehot = torch.zeros(B, L, vocab_size, device=x0.device)
        x0_onehot.scatter_(-1, x0.unsqueeze(-1), 1.0)  # one-hot
        xt_prob = self.sch.q_sample(x0_onehot, t)      # 機率分佈
        # 從機率分佈採樣回離散 token
        return torch.multinomial(xt_prob.view(-1, vocab_size), 1).view(B, L)

    def kl_term(self, xt: torch.Tensor, x0: torch.Tensor, logits_x0: torch.Tensor, t: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        """
        可微分版本的 KL[q(xt-1|xt,x0) || p_theta(xt-1|xt,c)]。
        - 以模型 Softmax(logits_x0) 近似 x̂0 的分佈，構造 posterior。
        - 支援 x_mask，避免將 <pad> 納入 KL 計算。
        """
        B, L, V = logits_x0.shape
        device = logits_x0.device
        dtype = logits_x0.dtype

        # 模型預測的 x̂0 機率
        x0_hat = torch.softmax(logits_x0, dim=-1)  # [B, L, V], requires_grad=True

        # one-hot 的 xt / x0（不需梯度）
        xt_onehot = torch.zeros(B, L, V, device=device, dtype=dtype)
        xt_onehot.scatter_(-1, xt.unsqueeze(-1), 1.0)
        x0_onehot = torch.zeros(B, L, V, device=device, dtype=dtype)
        x0_onehot.scatter_(-1, x0.unsqueeze(-1), 1.0)

        # 取 betas_t 與 betas_{t-1}
        if not hasattr(self.sch, 'betas'):
            raise ValueError("Scheduler 需提供 self.betas (shape [T]) 才能計算 posterior。")
        betas = self.sch.betas.to(device)  # [T]

        beta_t_vec = betas[t - 1]                           # [B]
        prev_idx = (t - 2).clamp(min=0)
        beta_prev_vec = betas[prev_idx]
        beta_prev_vec = torch.where(t.eq(1), torch.zeros_like(beta_prev_vec), beta_prev_vec)

        # 方便 broadcast
        beta_t    = beta_t_vec.view(B, 1, 1)
        beta_prev = beta_prev_vec.view(B, 1, 1)
        one       = torch.ones_like(x0_hat)
        eps       = 1e-8
        K         = float(V)

        # M_t^T x_t = (β_t/K)*1 + (1-β_t)*x_t
        MtT_xt = (beta_t / K) * one + (1.0 - beta_t) * xt_onehot

        # M_{t-1} x
        Mprev_x0    = (1.0 - beta_prev) * x0_onehot + (beta_prev / K) * one
        Mprev_x0hat = (1.0 - beta_prev) * x0_hat    + (beta_prev / K) * one  # requires_grad

        # 分母 x_t^T M_t x
        x0_at_xt    = torch.sum(x0_onehot * xt_onehot, dim=-1)  # [B, L], {0,1}
        x0hat_at_xt = torch.gather(x0_hat, dim=-1, index=xt.unsqueeze(-1)).squeeze(-1)  # [B, L]

        beta_t_scalar = beta_t_vec.unsqueeze(-1)  # [B, 1]
        denom_true = (beta_t_scalar / K) + (1.0 - beta_t_scalar) * x0_at_xt
        denom_pred = (beta_t_scalar / K) + (1.0 - beta_t_scalar) * x0hat_at_xt

        # posterior（真實 / 模型）
        q_post = (MtT_xt * Mprev_x0)    / (denom_true.unsqueeze(-1) + eps)
        p_post = (MtT_xt * Mprev_x0hat) / (denom_pred.unsqueeze(-1) + eps)  # requires_grad

        # KL per token：sum_k q * log(q/p)
        kl_token = torch.sum(q_post * (torch.log(q_post + eps) - torch.log(p_post + eps)), dim=-1)  # [B, L]

        # 掩碼有效 token
        if x_mask is not None:
            valid = x_mask.float()
            kl_per_sample = (kl_token * valid).sum(dim=1) / (valid.sum(dim=1) + eps)
        else:
            kl_per_sample = kl_token.mean(dim=1)

        return kl_per_sample.mean()

    def w_t(self, t: torch.Tensor) -> torch.Tensor:
        """w_t = ∏_{s=1}^t (1-β_s)，若無現成屬性則由 betas 現算。"""
        if hasattr(self.sch, 'alpha_bar'):
            alpha_bar = self.sch.alpha_bar.to(t.device)  # [T]
            return alpha_bar[t-1]
        elif hasattr(self.sch, 'w_prefix'):
            w_prefix = self.sch.w_prefix.to(t.device)  # [T]
            return w_prefix[t-1]
        else:
            if hasattr(self.sch, 'betas'):
                betas = self.sch.betas.to(t.device)  # [T]
                out = torch.ones_like(t, dtype=betas.dtype)
                for i in range(out.numel()):
                    ti = int(t.view(-1)[i].item())
                    out.view(-1)[i] = torch.prod(1.0 - betas[:ti])
                return out
            return torch.ones_like(t, dtype=torch.float32)

def _iter_with_progress(loader, desc: str, total: int | None = None):
    """
    將 DataLoader 包上 tqdm。若環境沒有 tqdm，就印出描述字串後直接回傳原 loader。
    參數：
        loader: 任何可疊代的資料載入器
        desc  : tqdm 左側的說明字串
        total : 總長度（若不傳，tqdm 會嘗試 len(loader)）
    """
    if tqdm is not None:
        try:
            return tqdm(loader, total=(total if total is not None else len(loader)), desc=desc, leave=False)
        except Exception:
            return tqdm(loader, desc=desc, leave=False)
    else:
        print(desc, flush=True)
        return loader

# ============ 訓練主流程 ============
def train_one_epoch(
    encoder: AcousticEncoder,
    decoder: DenoisingTransformerDecoder,
    s_proj: SpeechProjector,
    t_embed: TextEmbedding,
    t_proj: TextProjector,
    scheduler: SchedulerAdapter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: Config,
    global_step: int,
    scaler: amp.GradScaler = None,   # AMP 混合精度的縮放器（可為 None）
    epoch: int = 1,                  # 目前 epoch 編號（用於顯示）
    print_epoch_summary: bool = True,  # 是否在函式結尾印出「本 epoch 平均訓練 loss」
) -> tuple[int, float]:
    """
    單一 epoch 的訓練流程，含進度條與即時統計；回傳 (global_step, avg_train_loss)。

    進度條顯示（精簡版）：
      step, loss（總損失）, diff（KL 主損失）,
      lfd（若該步有併入 L_fd 時才顯示）
    """

    # 模式設定：通常 encoder.eval()，其餘訓練
    encoder.eval()          # 不更新 encoder，僅做條件抽取（節省記憶體與時間）
    decoder.train()
    s_proj.train()
    t_embed.train()
    t_proj.train()

    # 讀取常用設定
    pad_id     = cfg.data['pad_id']            # <pad> 的 token id（用於 mask）
    T_total    = cfg.diffusion['T']            # 擴散總步數（訓練用）
    log_every  = cfg.log['log_every']          # 幾步列印一次訓練日誌
    n_step_fd  = cfg.lfd['n_step_fd']          # 每隔多少 step 加一次 L_fd
    tau        = cfg.lfd.get('tau', 1.0)       # L_fd 的整體權重
    lambda_off = cfg.lfd['lambda_offdiag']     # L_fd 的 off-diagonal 懲罰

    # 用 tqdm 包住 DataLoader；若環境沒裝 tqdm，_iter_with_progress 會自動退回印字
    pbar = _iter_with_progress(loader, desc=f"Epoch {epoch} [train]")

    # 用於計算「epoch 平均訓練 loss」
    epoch_loss_sum = 0.0
    epoch_step_cnt = 0

    # 逐 batch 訓練
    for batch_idx, batch in enumerate(pbar, 1):
        # 取出語音與目標文字（x0）
        wave, x0 = batch                     # wave: [B, T_wav]；x0: [B, L]
        wave = wave.to(device)
        x0   = x0.to(device)
        B, L = x0.shape

        # 前向抽條件 c = c_psi(s)；使用 AMP 減少顯存（cuda 上生效）
        with amp.autocast('cuda'):
            c, c_mask, _ = encoder(wave)     # c: [B, S, d], c_mask: [B, S]

        # 隨機抽樣擴散時間步 t ~ Uniform{1..T}
        t = torch.randint(1, T_total + 1, (B,), device=device)

        # 前向擴散：由 x0 生成 xt（離散）
        xt = scheduler.sample_q(x0, t)       # [B, L]，整數 token ids

        # 解碼器估計 x_0 的分佈：f_theta(xt, t, c) -> logits_x0
        with amp.autocast('cuda'):
            logits = decoder(
                xt, t, c,
                x_mask=(x0 != pad_id),       # 避免模型學到 <pad>
                c_mask=c_mask
            )                                # [B, L, V]

        # 有效 token 的 mask（True=有效）
        x_mask = (x0 != pad_id)

        # 主要損失：Diffusion KL（論文 Eq.(6)）
        loss_diff = scheduler.kl_term(xt, x0, logits, t, x_mask)  # scalar

        # 是否在此步套用跨模態對齊損失 L_fd（論文 §3.3；總損失如式 (13)）
        apply_lfd = (global_step % n_step_fd) == 0
        loss = loss_diff
        loss_fd_value = None  # 僅用於日誌顯示
        if apply_lfd:
            # 將對 x_0 的 logits 投影到文本特徵空間；語音條件亦投影到語音特徵空間
            with amp.autocast('cuda'):
                z_text   = t_proj(t_embed(logits))   # [B, L, d_proj]
                z_speech = s_proj(c)                 # [B, S, d_proj]

            # 時間維度對齊（S 可能與 L 不同）
            S = z_speech.size(1)
            if S >= L:
                z_speech_aligned = z_speech[:, :L, :]
            else:
                pad = z_speech[:, -1:, :].repeat(1, L - S, 1)
                z_speech_aligned = torch.cat([z_speech, pad], dim=1)

            # w_t = ∏_{s=1}^t (1-β_s)（式 (13) 的權重；這裡取 batch mean 當 scalar）
            w_t = scheduler.w_t(t).mean()

            # L_fd：跨模態特徵去相關化 + 對齊（Barlow Twins 風格）
            loss_fd = lfd_loss(z_speech_aligned, z_text, lambda_offdiag=lambda_off)
            loss_fd_value = float(loss_fd.detach().item())

            # 將 L_fd 以 tau * w_t 的權重加到總損失（對齊論文式 (13)）
            loss = loss + tau * w_t * loss_fd

        # 反向傳播與最佳化
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            # 先取消縮放再做梯度裁剪（包含 decoder / s_proj / t_embed / t_proj）
            trainable_params = (
                list(decoder.parameters()) +
                list(s_proj.parameters()) +
                list(t_embed.parameters()) +
                list(t_proj.parameters())
            )
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            trainable_params = (
                list(decoder.parameters()) +
                list(s_proj.parameters()) +
                list(t_embed.parameters()) +
                list(t_proj.parameters())
            )
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            optimizer.step()

        # 每步更新進度條資訊（精簡：step / loss / diff，若有 L_fd 再顯示 lfd）
        loss_diff_val  = float(loss_diff.detach().item())
        total_loss_val = float(loss.detach().item())
        postfix_info = {
            "step": global_step,
            "loss": f"{total_loss_val:.3f}",
            "diff": f"{loss_diff_val:.3f}",
        }
        if loss_fd_value is not None:
            postfix_info["lfd"] = f"{loss_fd_value:.3f}"
        try:
            pbar.set_postfix(postfix_info)
        except Exception:
            pass

        # 累積 epoch 統計
        epoch_loss_sum += total_loss_val
        epoch_step_cnt += 1
        global_step += 1

    # 計算並回傳「本 epoch 平均訓練 loss」
    avg_loss = (epoch_loss_sum / max(1, epoch_step_cnt))
    if print_epoch_summary:
        print(f"[Summary] Epoch {epoch} Avg Train Loss: {avg_loss:.4f}", flush=True)
    return global_step, avg_loss


# ============ 訓練 CER 計算函數 ============


def main():
    parser = argparse.ArgumentParser(description="FDDM-ASR Training Script")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # 讀設定
    with open(args.config, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    cfg = Config(**raw)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(args.device)

    # ==== 建立模型 ====
    d_model = cfg.model['d_model']
    vocab = cfg.data['vocab_size']
    pad_id = cfg.data['pad_id']

    encoder = AcousticEncoder(**cfg.model['encoder'], d_model=d_model).to(device)
    decoder = DenoisingTransformerDecoder(
        vocab_size=vocab,
        d_model=d_model,
        nhead=cfg.model['nhead'],
        num_layers=cfg.model['num_layers'],
        dim_ff=cfg.model['dim_ff'],
        dropout=cfg.model['dropout'],
        max_len=1024,
        pad_id=pad_id,
    ).to(device)

    s_proj = SpeechProjector(d_in=d_model, d_proj=cfg.model['projector']['d_proj']).to(device)
    t_embed = TextEmbedding(vocab=vocab, d_out=cfg.model['projector']['d_proj'], mode='logits').to(device)
    t_proj  = TextProjector(d_in=cfg.model['projector']['d_proj'], d_proj=cfg.model['projector']['d_proj']).to(device)

    # ==== Scheduler ====
    if not _SCHED_IMPORT_OK:
        raise ImportError("請把 DiscreteDiffusionScheduler 加入匯入路徑，或修改 train.py 上方的 import。")
    scheduler = SchedulerAdapter(DiscreteDiffusionScheduler(
        K=vocab, 
        T=cfg.diffusion['T'], 
        device=device,
        beta_max=cfg.diffusion['beta_max']
    ))

    # ==== Optimizer ====
    params = list(decoder.parameters()) + list(s_proj.parameters()) + list(t_embed.parameters()) + list(t_proj.parameters())
    optim = torch.optim.AdamW(params, lr=cfg.optim['lr'], weight_decay=cfg.optim['weight_decay'])
    
    # ==== AMP GradScaler ====
    scaler = amp.GradScaler('cuda') if torch.cuda.is_available() else None
    if scaler is not None:
        print("AMP 已啟用：使用混合精度訓練以提升效能和數值穩定性")
    else:
        print("AMP 未啟用：使用標準精度訓練")

    # ==== DataLoader（從 train.json, validation.json, test.json 載入真實資料） ====
    train_json = cfg.data.get('train_json', 'data/processed/train.json')
    val_json = cfg.data.get('val_json', 'data/processed/validation.json')
    test_json = cfg.data.get('test_json', 'data/processed/test.json')
    tokenizer_model_path = cfg.data.get('tokenizer_model_path', 'data/tokenizer/zh-TW_A/spm_zhTW_A.model')
    
    # 載入訓練資料集
    train_set = CVZhTWDataset(
        json_file=train_json,
        tokenizer_vocab_path=tokenizer_model_path,
        max_len=cfg.data.get('max_len', 128),
        pad_id=pad_id,
        bos_id=cfg.data.get('bos_id'),
        eos_id=cfg.data.get('eos_id')
    )
    train_loader = DataLoader(train_set, batch_size=cfg.optim['batch_size'], shuffle=True, drop_last=True)
    
    # 載入驗證資料集（如果檔案存在）
    val_loader = None
    if os.path.exists(val_json):
        val_set = CVZhTWDataset(
            json_file=val_json,
            tokenizer_vocab_path=tokenizer_model_path,
            max_len=cfg.data.get('max_len', 128),
            pad_id=pad_id,
            bos_id=cfg.data.get('bos_id'),
            eos_id=cfg.data.get('eos_id')
        )
        val_loader = DataLoader(val_set, batch_size=cfg.optim['batch_size'], shuffle=False, drop_last=False)
    
    # 載入測試資料集（如果檔案存在）
    test_loader = None
    if os.path.exists(test_json):
        test_set = CVZhTWDataset(
            json_file=test_json,
            tokenizer_vocab_path=tokenizer_model_path,
            max_len=cfg.data.get('max_len', 128),
            pad_id=pad_id,
            bos_id=cfg.data.get('bos_id'),
            eos_id=cfg.data.get('eos_id')
        )
        test_loader = DataLoader(test_set, batch_size=cfg.optim['batch_size'], shuffle=False, drop_last=False)
    
    # 載入 tokenizer 用於 CER 計算
    import sentencepiece as spm
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_model_path)

    # ==== 訓練 ====
    os.makedirs(cfg.log['ckpt_dir'], exist_ok=True)
    global_step = 1
    
    best_val_cer = float('inf')  # 初始化為無限大
    best_epoch = 0
    
    for epoch in range(1, cfg.optim['num_epochs']+1):
        print(f"Epoch {epoch}")
        global_step, train_loss = train_one_epoch(
            encoder, decoder, s_proj, t_embed, t_proj,
            scheduler, train_loader, optim, device, cfg, global_step,
            scaler  # 傳入 AMP GradScaler
        )
        
        # —— 每個 epoch 結束後：驗證（如有）——
        val_loss = None
        val_cer  = None
        if val_loader is not None:
            val_cer = evaluate_cer_with_jumpy_sampling(
                encoder, decoder, scheduler, val_loader, device, cfg, tokenizer
            )
            val_loss = evaluate_validation_loss(
                encoder, decoder, s_proj, t_embed, t_proj,
                scheduler, val_loader, device, cfg
            )

            # 更新最佳權重
            if val_cer < best_val_cer:
                best_val_cer = val_cer
                best_epoch = epoch
                best_ckpt = {
                    'decoder': decoder.state_dict(),
                    's_proj': s_proj.state_dict(),
                    't_embed': t_embed.state_dict(),
                    't_proj': t_proj.state_dict(),
                    'epoch': epoch,
                    'step': global_step,
                    'best_val_cer': best_val_cer,
                    'config': raw,
                }
                best_path = os.path.join(cfg.log['ckpt_dir'], 'best_model.pt')
                torch.save(best_ckpt, best_path)
                print(f"Saved BEST model (epoch {best_epoch}, val_cer {best_val_cer:.4f}) → {best_path}")

        # —— 每個 epoch 結束後：測試（如有）——
        test_cer = None
        if test_loader is not None:
            test_cer = evaluate_cer_with_jumpy_sampling(
                encoder, decoder, scheduler, test_loader, device, cfg, tokenizer
            )

        # —— 統一列印「Epoch 摘要」：只在這裡輸出驗證/測試指標 —— 
        msg = f"[Epoch {epoch} Summary] train_loss={train_loss:.4f}"
        if val_loss is not None:
            msg += f" | val_loss={val_loss:.4f}"
        if val_cer is not None:
            msg += f" | val_cer={val_cer:.4f}"
        if test_cer is not None:
            msg += f" | test_cer={test_cer:.4f}"
        print(msg)

        # 儲存每個 epoch 的權重（用於恢復訓練）
        ckpt = {
            'decoder': decoder.state_dict(),
            's_proj': s_proj.state_dict(),
            't_embed': t_embed.state_dict(),
            't_proj': t_proj.state_dict(),
            'step': global_step,
            'epoch': epoch,
            'config': raw,
        }
        torch.save(ckpt, os.path.join(cfg.log['ckpt_dir'], f"ep{epoch:03d}.pt"))
    
    # 訓練結束後報告最佳權重資訊
    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print(f"Best validation CER: {best_val_cer:.4f} (Epoch {best_epoch})")
    print(f"Best model saved at: {os.path.join(cfg.log['ckpt_dir'], 'best_model.pt')}")
    print("="*50)

if __name__ == '__main__':
    main()
