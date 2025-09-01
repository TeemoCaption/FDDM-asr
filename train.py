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

# ====== 匯入本專案模組 ======
from models.acoustic_encoder import AcousticEncoder
from models.denoise_decoder import DenoisingTransformerDecoder
from models.projection import SpeechProjector, TextEmbedding, TextProjector
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
        
# ============ CER 評估指標實作 ============
# CER (Character Error Rate) 計算函數
# 計算字元錯誤率：(插入 + 刪除 + 替換) / 總字元數
def calculate_cer(pred_text: str, target_text: str) -> float:
    # 移除空白和標點符號，轉小寫進行比較
    import re
    
    # 簡化文字預處理：移除空白
    pred_text = pred_text.replace(' ', '')
    target_text = target_text.replace(' ', '')
    
    # 計算編輯距離
    def levenshtein_distance(s1: str, s2: str) -> int:
        # 使用動態規劃計算最小編輯距離
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 初始化第一行和第一列
        for i in range(m + 1):
            dp[i][0] = i  # 刪除操作
        for j in range(n + 1):
            dp[0][j] = j  # 插入操作
        
        # 填充 dp 表格
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]  # 字符相同，無需操作
                else:
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,     # 刪除
                        dp[i][j - 1] + 1,     # 插入
                        dp[i - 1][j - 1] + 1  # 替換
                    )
        
        return dp[m][n]
    
    # 計算編輯距離
    edit_distance = levenshtein_distance(pred_text, target_text)
    
    # 計算 CER：編輯距離 / 真實文字長度
    if len(target_text) == 0:
        return 0.0 if len(pred_text) == 0 else 1.0
    
    cer = edit_distance / len(target_text)
    return cer

# 從 logits 轉換為預測文字的輔助函數
def logits_to_text(logits: torch.Tensor, tokenizer, pad_id: int, bos_id: int = None, eos_id: int = None) -> str:
    # logits: [L, V] 或 [B, L, V]，這裡假設 [L, V]
    if logits.dim() == 3:
        # 如果是 batch，取第一個樣本
        logits = logits[0]  # [L, V]
    
    # 取得預測的 token IDs（取 argmax）
    pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()  # [L]
    
    # 移除 padding token
    valid_ids = []
    for token_id in pred_ids:
        if token_id == pad_id:
            break
        valid_ids.append(token_id)
    
    # 移除 bos 和 eos token（如果存在）
    if bos_id is not None and valid_ids and valid_ids[0] == bos_id:
        valid_ids = valid_ids[1:]
    if eos_id is not None and valid_ids and valid_ids[-1] == eos_id:
        valid_ids = valid_ids[:-1]
    
    # 將 token IDs 轉換為文字
    if valid_ids:
        text = tokenizer.decode(valid_ids)
    else:
        text = ''
    
    return text
class Config:
    seed: int
    data: dict
    model: dict
    diffusion: dict
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
        # 計算 KL divergence: KL[q(xt-1|xt,x0) || p_theta(xt-1|xt,c)]
        # 改進維度聚合：先對 token 維做均值，再對 batch 做均值
        B, L, V = logits_x0.shape
        
        # 轉換為 one-hot 格式
        xt_onehot = torch.zeros(B, L, V, device=xt.device)
        xt_onehot.scatter_(-1, xt.unsqueeze(-1), 1.0)
        
        x0_onehot = torch.zeros(B, L, V, device=x0.device)
        x0_onehot.scatter_(-1, x0.unsqueeze(-1), 1.0)
        
        # 從 logits 得到預測的 x0 機率分佈
        x0hat_prob = torch.softmax(logits_x0, dim=-1)
        
        # 計算真實後驗 q(xt-1|xt,x0)
        q_posterior = self.sch.q_posterior(xt_onehot, x0_onehot, t)
        
        # 計算預測後驗 p_theta(xt-1|xt,c) ≈ q(xt-1|xt,x0hat)
        p_posterior = self.sch.q_posterior(xt_onehot, x0hat_prob, t)
        
        # KL divergence，改進數值穩定性
        eps = 1e-8
        kl_per_token = torch.sum(q_posterior * torch.log((q_posterior + eps) / (p_posterior + eps)), dim=-1)  # [B, L]
        
        # 精細的維度聚合：考慮 mask 的影響
        if x_mask is not None:
            # 只對有效 token 計算平均
            kl_per_sample = (kl_per_token * x_mask.float()).sum(dim=1) / (x_mask.sum(dim=1).float() + eps)  # [B]
        else:
            # 對所有 token 位置取均值
            kl_per_sample = kl_per_token.mean(dim=1)  # [B]
        
        # 對 batch 取均值
        return kl_per_sample.mean()  # scalar
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
) -> int:
    encoder.eval()   # 預設凍結
    decoder.train()
    s_proj.train()
    t_embed.train()
    t_proj.train()

    pad_id = cfg.data['pad_id']
    T_total = cfg.diffusion['T']
    log_every = cfg.log['log_every']

    for batch_idx, (wave, x0) in enumerate(loader, start=1):
        wave = wave.to(device)
        x0 = x0.to(device)
        B, L = x0.shape

        # 取聲學條件 c = c_psi(s)
        with torch.no_grad():
            c, c_mask, _ = encoder(wave)   # [B, S, d]

        # 抽樣時間步 t ~ U[1, T]
        t = torch.randint(1, T_total+1, (B,), device=device)

        # 前向擴散得到 x_t
        xt = scheduler.sample_q(x0, t)     # [B, L]

        # 解碼器 f_theta(xt, t, c) -> logits 對 x_0 的分佈
        # 注意：x_mask 排除 padding token，確保模型不學習預測 padding
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
            loss_fd_value = float(loss_fd)  # 記錄數值用於日誌
            
            # 加入總損失
            loss = loss + tau * w_t * loss_fd

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
        optimizer.step()

        if (global_step % log_every) == 0:
            log_msg = f"step={global_step} loss_diff={float(loss_diff):.4f}"
            if apply_lfd:
                log_msg += f" loss_fd={loss_fd_value:.4f} w_t={float(w_t):.4f}"
            log_msg += f" total_loss={float(loss):.4f}"
            print(log_msg)
        global_step += 1

# ============ 訓練 CER 計算函數 ============
def evaluate_train_cer(
    encoder: AcousticEncoder,
    decoder: DenoisingTransformerDecoder,
    s_proj: SpeechProjector,
    t_embed: TextEmbedding,
    t_proj: TextProjector,
    scheduler: SchedulerAdapter,
    train_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    tokenizer,
    max_batches: int = 5  # 只計算前 max_batches 個 batch 來節省時間
) -> float:
    # 設定模型為評估模式
    encoder.eval()
    decoder.eval()
    s_proj.eval()
    t_embed.eval()
    t_proj.eval()
    
    pad_id = cfg.data['pad_id']
    total_cer = 0.0  # 累計 CER
    total_samples = 0   # 總樣本數
    
    with torch.no_grad():
        for batch_idx, (wave, x0) in enumerate(train_loader, start=1):
            if batch_idx > max_batches:
                break  # 只計算前幾個 batch
            
            wave = wave.to(device)
            x0 = x0.to(device)
            B, L = x0.shape
            
            # 取聲學條件 c = c_psi(s)
            c, c_mask, _ = encoder(wave)  # [B, S, d]
            
            # 在評估時，使用 t=1（最小的雜訊）進行推理
            t = torch.ones(B, dtype=torch.long, device=device)  # [B]
            
            # 對於評估，我們使用 x0 作為 xt（無雜訊輸入）
            xt = x0.clone()  # [B, L]
            
            # 解碼器推理：f_theta(xt, t, c) -> logits 對 x_0 的分佈
            logits = decoder(xt, t, c, x_mask=(x0!=pad_id), c_mask=c_mask)  # [B, L, V]
            
            # 對每個樣本計算 CER
            for b in range(B):
                # 取得預測文字
                pred_logits = logits[b]  # [L, V]
                pred_text = logits_to_text(
                    pred_logits, 
                    tokenizer, 
                    pad_id=pad_id, 
                    bos_id=cfg.data.get('bos_id'), 
                    eos_id=cfg.data.get('eos_id')
                )
                
                # 取得真實文字（從 x0 還原）
                target_ids = x0[b].cpu().numpy()  # [L]
                valid_target_ids = []
                for token_id in target_ids:
                    if token_id == pad_id:
                        break
                    valid_target_ids.append(token_id)
                
                # 移除 bos 和 eos token
                if cfg.data.get('bos_id') is not None and valid_target_ids and valid_target_ids[0] == cfg.data['bos_id']:
                    valid_target_ids = valid_target_ids[1:]
                if cfg.data.get('eos_id') is not None and valid_target_ids and valid_target_ids[-1] == cfg.data['eos_id']:
                    valid_target_ids = valid_target_ids[:-1]
                
                # 將 token IDs 轉換為文字
                if valid_target_ids:
                    target_text = tokenizer.decode(valid_target_ids)
                else:
                    target_text = ''
                
                # 計算 CER
                cer = calculate_cer(pred_text, target_text)
                total_cer += cer
                total_samples += 1
    
    # 計算平均 CER
    avg_cer = total_cer / total_samples if total_samples > 0 else 0.0
    return avg_cer

def main():
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
        global_step = train_one_epoch(
            encoder, decoder, s_proj, t_embed, t_proj,
            scheduler, train_loader, optim, device, cfg, global_step
        )
        
        # 每個 epoch 結束後進行訓練 CER 評估
        train_cer = evaluate_train_cer(
            encoder, decoder, s_proj, t_embed, t_proj,
            scheduler, train_loader, device, cfg, tokenizer, max_batches=5
        )
        print(f"Epoch {epoch} Train CER: {train_cer:.4f}")
        
        # 每個 epoch 結束後進行驗證
        if val_loader is not None:
            val_cer = evaluate_cer(
                encoder, decoder, s_proj, t_embed, t_proj,
                scheduler, val_loader, device, cfg, tokenizer
            )
            print(f"Epoch {epoch} Validation CER: {val_cer:.4f}")
            
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
            test_cer = evaluate_cer(
                encoder, decoder, s_proj, t_embed, t_proj,
                scheduler, test_loader, device, cfg, tokenizer
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
    print("🏆 TRAINING COMPLETED!")
    print(f"Best validation CER: {best_val_cer:.4f} (Epoch {best_epoch})")
    print(f"Best model saved at: {os.path.join(cfg.log['ckpt_dir'], 'best_model.pt')}")
    print("="*50)

if __name__ == '__main__':
    main()
