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
from models.evaluate import (
    calculate_cer, logits_to_text, evaluate_train_cer, evaluate_cer,
    evaluate_validation_loss, evaluate_cer_with_full_sampling,
    evaluate_cer_with_jumpy_sampling, evaluate_cer_with_multi_sample
)  # 匯入評估函數
from losses.fddm_losses import lfd_loss

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
        
        # 計算最大音頻長度（從配置中讀取，或使用預設值）
        # max_seconds 通常在配置中設定，預設為 20 秒
        self.max_audio_samples = 20 * 16000  # 預設 20 秒 * 16kHz
        
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
        
        # 確保音頻長度一致：截斷或填充到固定長度
        if len(wav) > self.max_audio_samples:
            # 截斷過長的音頻
            wav = wav[:self.max_audio_samples]
        elif len(wav) < self.max_audio_samples:
            # 填充過短的音頻（用零填充）
            padding = torch.zeros(self.max_audio_samples - len(wav))
            wav = torch.cat([wav, padding], dim=0)
        
        # 處理文本
        text = item['normalized_sentence']
        tokens = self.tokenizer.encode(text)
        
        # 添加特殊 token
        if self.bos_id is not None:
            tokens = [self.bos_id] + tokens
        if self.eos_id is not None:
            tokens = tokens + [self.eos_id]
        
        # 截斷或填充
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens += [self.pad_id] * (self.max_len - len(tokens))
        
        x0 = torch.tensor(tokens, dtype=torch.long)
        
        return wav, x0
        
# ============ CER 評估指標實作 ============
# CER 評估相關函數已移至 models/evaluate.py
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
        # 轉換 x0 為 one-hot 格式並呼叫 scheduler 的 q_sample 方法
        B, L = x0.shape
        vocab_size = self.sch.K
        x0_onehot = torch.zeros(B, L, vocab_size, device=x0.device)
        x0_onehot.scatter_(-1, x0.unsqueeze(-1), 1.0)  # 轉為 one-hot
        xt_prob = self.sch.q_sample(x0_onehot, t)  # 呼叫實際的方法名稱
        # 從機率分佈採樣回離散 token
        return torch.multinomial(xt_prob.view(-1, vocab_size), 1).view(B, L)
    def kl_term(self, xt: torch.Tensor, x0: torch.Tensor, logits_x0: torch.Tensor, t: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        """
        可微分版本的 KL[q(xt-1|xt,x0) || p_theta(xt-1|xt,c)]。
        參數：
            xt:         [B, L]，第 t 步的離散樣本（整數 id）
            x0:         [B, L]，真實目標（整數 id）
            logits_x0:  [B, L, V]，模型對 x_0 的類別 logits（可微）
            t:          [B]，每個樣本對應的時間步（1..T）
            x_mask:     [B, L]，True 表有效 token（避免 pad 影響）
        回傳：
            樣本/序列聚合後的 scalar（需梯度）
        """
        B, L, V = logits_x0.shape
        device = logits_x0.device
        dtype = logits_x0.dtype  # 用 float32/float16 皆可

        # 將 logits 轉機率（保留梯度）
        x0_hat = torch.softmax(logits_x0, dim=-1)                 # [B, L, V], requires_grad=True

        # one-hot（不需要梯度，但不會阻斷 x0_hat 的梯度）
        xt_onehot = torch.zeros(B, L, V, device=device, dtype=dtype)
        xt_onehot.scatter_(-1, xt.unsqueeze(-1), 1.0)             # [B, L, V]
        x0_onehot = torch.zeros(B, L, V, device=device, dtype=dtype)
        x0_onehot.scatter_(-1, x0.unsqueeze(-1), 1.0)             # [B, L, V]

        # 取 betas_t 與 betas_{t-1}（t=1 時視作 beta_{0}=0 ⇒ M_0 為單位轉移）
        if not hasattr(self.sch, 'betas'):
            raise ValueError("Scheduler 需提供 self.betas (shape [T]) 才能計算 posterior。")
        betas = self.sch.betas.to(device)                         # [T]

        beta_t_vec = betas[t - 1]                                 # [B]
        prev_idx = (t - 2).clamp(min=0)                           # [B]；t=1 時先指向 0
        beta_prev_vec = betas[prev_idx]                           # [B]
        beta_prev_vec = torch.where(t.eq(1),                      # t=1 強制設為 0
                                    torch.zeros_like(beta_prev_vec),
                                    beta_prev_vec)

        # 方便 broadcast 用的 shape
        beta_t     = beta_t_vec.view(B, 1, 1)                     # [B,1,1]
        beta_prev  = beta_prev_vec.view(B, 1, 1)                  # [B,1,1]
        one        = torch.ones_like(x0_hat)                      # [B, L, V]
        eps        = 1e-8
        K          = float(V)

        # M_t^T x_t = (β_t/K)*1 + (1-β_t)*x_t
        MtT_xt = (beta_t / K) * one + (1.0 - beta_t) * xt_onehot  # [B, L, V]

        # M_{t-1} x ；"真實"用 x0 的 one-hot，"模型"用 x0_hat 的機率
        Mprev_x0    = (1.0 - beta_prev) * x0_onehot + (beta_prev / K) * one   # [B, L, V]
        Mprev_x0hat = (1.0 - beta_prev) * x0_hat    + (beta_prev / K) * one   # [B, L, V] requires_grad

        # 分母 x_t^T M_t x
        # 真實：x = x0（one-hot）；模型：x = x0_hat（softmax 機率）
        # gather：取出 x 在 xt 所指類別的值，shape [B,L]
        x0_at_xt    = torch.sum(x0_onehot * xt_onehot, dim=-1)                      # [B, L], {0,1}
        x0hat_at_xt = torch.gather(x0_hat, dim=-1, index=xt.unsqueeze(-1)).squeeze(-1)  # [B, L]

        beta_t_scalar = beta_t_vec.unsqueeze(-1)                                     # [B, 1]
        denom_true = (beta_t_scalar / K) + (1.0 - beta_t_scalar) * x0_at_xt          # [B, L]
        denom_pred = (beta_t_scalar / K) + (1.0 - beta_t_scalar) * x0hat_at_xt       # [B, L]

        # posterior：按元素相乘後除以分母（對每個 token 正規化）
        q_post = (MtT_xt * Mprev_x0)    / (denom_true.unsqueeze(-1) + eps)           # [B, L, V]
        p_post = (MtT_xt * Mprev_x0hat) / (denom_pred.unsqueeze(-1) + eps)           # [B, L, V], requires_grad

        # KL per token：sum_k q * log(q/p)
        kl_token = torch.sum(q_post * (torch.log(q_post + eps) - torch.log(p_post + eps)), dim=-1)  # [B, L]

        # 掩碼有效 token 再做平均
        if x_mask is not None:
            valid = x_mask.float()                                                   # [B, L]
            kl_per_sample = (kl_token * valid).sum(dim=1) / (valid.sum(dim=1) + eps) # [B]
        else:
            kl_per_sample = kl_token.mean(dim=1)

        return kl_per_sample.mean()
    def w_t(self, t: torch.Tensor) -> torch.Tensor:
        # 使用 scheduler 的 alpha_bar (即 w_prefix) 屬性
        if hasattr(self.sch, 'alpha_bar'):
            alpha_bar = self.sch.alpha_bar.to(t.device)  # [T]
            return alpha_bar[t-1]  # t 是 1-indexed
        elif hasattr(self.sch, 'w_prefix'):
            w_prefix = self.sch.w_prefix.to(t.device)  # [T]
            return w_prefix[t-1]  # t 是 1-indexed
        else:
            # 備用方案：從 betas 計算
            if hasattr(self.sch, 'betas'):
                betas = self.sch.betas.to(t.device)  # [T]
                out = torch.ones_like(t, dtype=betas.dtype)
                for i in range(out.numel()):
                    ti = int(t.view(-1)[i].item())
                    out.view(-1)[i] = torch.prod(1.0 - betas[:ti])
                return out
            return torch.ones_like(t, dtype=torch.float32)

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
    scaler: GradScaler = None,  # 添加 AMP scaler 參數
) -> tuple[int, float]:  # 返回 global_step 和平均訓練損失
    encoder.eval()   # 預設凍結
    decoder.train()
    s_proj.train()
    t_embed.train()
    t_proj.train()

    pad_id = cfg.data['pad_id']
    T_total = cfg.diffusion['T']
    log_every = cfg.log['log_every']

    # 初始化訓練損失累積器
    epoch_loss_sum = 0.0
    epoch_step_count = 0

    for batch_idx, (wave, x0) in enumerate(loader, start=1):
        wave = wave.to(device)
        x0 = x0.to(device)
        B, L = x0.shape

        # 取聲學條件 c = c_psi(s)
        # 注意：即使編碼器被凍結，我們仍然需要它的輸出來計算梯度
        with amp.autocast('cuda'):  # AMP: 混合精度前向傳播
            c, c_mask, _ = encoder(wave)   # [B, S, d]

        # 抽樣時間步 t ~ U[1, T]
        t = torch.randint(1, T_total+1, (B,), device=device)

        # 前向擴散得到 x_t
        xt = scheduler.sample_q(x0, t)     # [B, L]

        # 解碼器 f_theta(xt, t, c) -> logits 對 x_0 的分佈
        # 注意：x_mask 排除 padding token，確保模型不學習預測 padding
        with amp.autocast('cuda'):  # AMP: 混合精度前向傳播
            logits = decoder(xt, t, c, x_mask=(x0!=pad_id), c_mask=c_mask)  # [B, L, V]

        # 建立 token mask（排除 padding）
        x_mask = (x0 != pad_id)  # [B, L] True=有效 token
        
        # Diffusion KL（論文 Eq.(6)）
        loss_diff = scheduler.kl_term(xt, x0, logits, t, x_mask)  # 傳入 mask 進行精確計算
        
        # 初始化總損失
        loss = loss_diff
        loss_fd_value = 0.0  # 用於日誌記錄
        
        # 每 n_step_fd 次加入 L_fd（論文 3.3）
        n_step_fd = cfg.lfd['n_step_fd']
        apply_lfd = (global_step % n_step_fd) == 0
        
        if apply_lfd:
            lambda_off = cfg.lfd['lambda_offdiag']
            tau = cfg.lfd.get('tau', 1.0)
            
            # 計算跨模態特徵投影
            with amp.autocast('cuda'):  # AMP: 混合精度前向傳播
                z_text = t_proj(t_embed(logits))  # [B, L, d_proj]
                z_speech = s_proj(c)             # [B, S, d_proj]
            
            # 對齊序列長度（語音特徵通常比文字長）
            S = z_speech.size(1)
            if S >= L:
                z_speech_aligned = z_speech[:, :L, :]  # 截取前 L 個時間步
            else:
                # 重複最後一個時間步來補齊
                pad = z_speech[:, -1:, :].repeat(1, L - S, 1)
                z_speech_aligned = torch.cat([z_speech, pad], dim=1)
            
            # 計算權重 w_t = ∏_{s=1}^t (1-β_s)
            w_t = scheduler.w_t(t).mean()  # 對 batch 取平均
            
            # 計算 L_fd 損失
            loss_fd = lfd_loss(z_speech_aligned, z_text, lambda_offdiag=lambda_off)
            loss_fd_value = loss_fd.detach().item()  # 記錄數值用於日誌
            
            # 加入總損失
            loss = loss + tau * w_t * loss_fd

        optimizer.zero_grad(set_to_none=True)
        
        # AMP: 使用 GradScaler 進行反向傳播
        if scaler is not None:
            scaler.scale(loss).backward()
            
            # 擴展梯度裁剪範圍：包含所有可訓練模型參數
            # 避免梯度爆炸，提升訓練穩定性
            all_trainable_params = (
                list(decoder.parameters()) +
                list(s_proj.parameters()) +
                list(t_embed.parameters()) +
                list(t_proj.parameters())
            )
            scaler.unscale_(optimizer)  # 取消縮放以進行梯度裁剪
            torch.nn.utils.clip_grad_norm_(all_trainable_params, 5.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # 非 AMP 模式：傳統反向傳播
            loss.backward()
            
            # 擴展梯度裁剪範圍：包含所有可訓練模型參數
            # 避免梯度爆炸，提升訓練穩定性
            all_trainable_params = (
                list(decoder.parameters()) +
                list(s_proj.parameters()) +
                list(t_embed.parameters()) +
                list(t_proj.parameters())
            )
            torch.nn.utils.clip_grad_norm_(all_trainable_params, 5.0)
            
            optimizer.step()

        if (global_step % log_every) == 0:
            loss_diff_value  = loss_diff.detach().item()
            total_loss_value = loss.detach().item()
            log_msg = f"step={global_step} loss_diff={loss_diff_value:.4f}"
            if apply_lfd:
                log_msg += f" loss_fd={loss_fd_value:.4f} w_t={float(w_t):.4f}"  # w_t 通常不需要梯度
            log_msg += f" total_loss={total_loss_value:.4f}"
            print(log_msg)
        
        # 累積訓練損失
        epoch_loss_sum += total_loss_value
        epoch_step_count += 1
        
        global_step += 1
    
    # 計算平均訓練損失
    avg_train_loss = epoch_loss_sum / epoch_step_count if epoch_step_count > 0 else 0.0
    return global_step, avg_train_loss

# ============ 訓練 CER 計算函數 ============
# CER 評估相關函數已移至 models/evaluate.py

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
    # 使用正確的參數初始化 scheduler：K (vocab_size), T (diffusion steps), device, beta_max
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
    # 提升數值穩定性和訓練效率，特別適用於 FP16 訓練
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
    
    # 初始化最佳權重追蹤
    best_val_cer = float('inf')  # 初始化為無限大
    best_epoch = 0
    
    for epoch in range(1, cfg.optim['num_epochs']+1):
        print(f"Epoch {epoch}")
        global_step, train_loss = train_one_epoch(
            encoder, decoder, s_proj, t_embed, t_proj,
            scheduler, train_loader, optim, device, cfg, global_step,
            scaler  # 傳入 AMP GradScaler
        )
        
        # 每個 epoch 結束後進行訓練 CER 評估
        train_cer = evaluate_train_cer(
            encoder, decoder, s_proj, t_embed, t_proj,
            scheduler, train_loader, device, cfg, tokenizer, max_batches=5
        )
        print(f"Epoch {epoch} Train CER: {train_cer:.4f} | Train Loss: {train_loss:.4f}")
        
        # 每個 epoch 結束後進行驗證
        if val_loader is not None:
            val_cer = evaluate_cer_with_jumpy_sampling(
                encoder, decoder, scheduler, val_loader, device, cfg, tokenizer
            )
            # 計算驗證損失
            val_loss = evaluate_validation_loss(
                encoder, decoder, s_proj, t_embed, t_proj,
                scheduler, val_loader, device, cfg
            )
            print(f"Epoch {epoch} Validation CER: {val_cer:.4f} | Validation Loss: {val_loss:.4f}")
            
            # 檢查是否為最佳驗證 CER
            if val_cer < best_val_cer:
                best_val_cer = val_cer
                best_epoch = epoch
                print(f"New best validation CER: {best_val_cer:.4f} at epoch {best_epoch}")
                
                # 保存最佳權重
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
                print(f"Saved best model to: {best_path}")
        
        # 每個 epoch 結束後進行測試
        if test_loader is not None:
            test_cer = evaluate_cer_with_jumpy_sampling(
                encoder, decoder, scheduler, test_loader, device, cfg, tokenizer
            )
            print(f"Epoch {epoch} Test CER: {test_cer:.4f}")
        
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
