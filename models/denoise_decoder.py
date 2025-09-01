# -*- coding: utf-8 -*-
"""
DenoisingTransformerDecoder: 非自回歸去噪解碼器（支援 cross-attention）
- 論文第 3.1 節：f_theta(x_t, t, c) ->
    輸出對 x_0 的類別分佈（logits），供 scheduler 計算 posterior。
- 設計理念：
    * 輸入 noisy tokens x_t（離散），先嵌入到向量空間。
    * 注入時間步 t 的 embedding（sinusoidal + MLP）。
    * 自注意力（不加 causal mask，因為整串平行預測）。
    * Cross-attention 到聲音條件 c（來自 AcousticEncoder）。
    * 前饋層 + 殘差 + LayerNorm。
    * 支援多種位置編碼：RoPE、sinusoidal、learned embedding。
    * 支援 FiLM 門控機制增強跨模態條件控制。

回傳：
- logits: [B, L, vocab]，對每個位置預測 x_0 的分佈
"""
from typing import Optional, Tuple, Literal
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- RoPE (Rotary Position Embedding) 實現 ----
class RoPEEmbedding(nn.Module):
    """旋轉位置編碼，支援任意長度序列而無需預先定義最大長度"""
    def __init__(self, d_model: int, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.base = base
        # 預計算頻率
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 cos 和 sin 位置編碼矩陣"""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, d_model//2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, d_model]
        return emb.cos(), emb.sin()
    
    @staticmethod
    def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """將 RoPE 應用到輸入張量 x"""
        # x: [B, L, d_model]
        # cos, sin: [L, d_model]
        x1, x2 = x[..., ::2], x[..., 1::2]  # 分離奇偶維度
        # 旋轉變換
        rotated = torch.cat([
            x1 * cos[..., ::2] - x2 * sin[..., 1::2],
            x1 * sin[..., ::2] + x2 * cos[..., 1::2]
        ], dim=-1)
        return rotated

# ---- Sinusoidal 位置編碼（備選方案）----
class SinusoidalPositionEmbedding(nn.Module):
    """標準 sinusoidal 位置編碼，支援任意長度"""
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """生成 sinusoidal 位置編碼"""
        pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * 
                           -(math.log(self.max_len) / self.d_model))
        pe = torch.zeros(seq_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe

# ---- FiLM (Feature-wise Linear Modulation) 門控機制 ----
class FiLMLayer(nn.Module):
    """特徵線性調製層，用於條件控制"""
    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        # 生成縮放和偏移參數
        self.scale_proj = nn.Linear(cond_dim, d_model)
        self.shift_proj = nn.Linear(cond_dim, d_model)
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, d_model] 輸入特徵
        cond: [B, d_model] 條件向量（通常是池化後的聲學特徵）
        """
        scale = self.scale_proj(cond).unsqueeze(1)  # [B, 1, d_model]
        shift = self.shift_proj(cond).unsqueeze(1)  # [B, 1, d_model]
        return x * (1 + scale) + shift

# ---- 實用模組：時間步嵌入 (sinusoidal + MLP) ----
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model: int, max_steps: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_steps = max_steps
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] 或 [1] 的整數時間步。輸出 [B, d_model]。"""
        if t.dim() == 0:
            t = t[None]
        device = t.device
        half = self.d_model // 2
        # 按 DDPM 慣例製作週期基底
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0), math.log(self.max_steps), half, device=device
            ) * (-1)
        )  # [half]
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [B, d_model]
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))  # 若 d_model 為奇數，右側補 1 維
        return self.mlp(emb)

# ---- Transformer Block，含 cross-attention 和 FiLM 門控 ----
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1, 
                 use_film: bool = True, pos_emb_type: str = "rope"):
        super().__init__()
        self.use_film = use_film  # 是否使用 FiLM 門控
        self.pos_emb_type = pos_emb_type  # 位置編碼類型
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # FiLM 門控層（可選）
        if self.use_film:
            self.film_layer = FiLMLayer(d_model, d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                 # [B, L, d]
        cond: torch.Tensor,              # [B, S, d]
        x_mask: Optional[torch.Tensor],  # [B, L] True=keep
        c_mask: Optional[torch.Tensor],  # [B, S] True=keep
        rope_cos: Optional[torch.Tensor] = None,  # RoPE cos 編碼
        rope_sin: Optional[torch.Tensor] = None,  # RoPE sin 編碼
    ) -> torch.Tensor:
        # 應用 RoPE 位置編碼到自注意力的 query 和 key
        if self.pos_emb_type == "rope" and rope_cos is not None and rope_sin is not None:
            q = k = RoPEEmbedding.apply_rotary_pos_emb(x, rope_cos, rope_sin)
            v = x  # value 不需要位置編碼
        else:
            q = k = v = x
        
        # 自注意力（不設 causal mask，因為非自回歸）
        x2, _ = self.self_attn(q, k, v, key_padding_mask=(~x_mask) if x_mask is not None else None)
        x = x + self.drop(x2)
        x = self.norm1(x)

        # Cross-attention 到聲學條件 cond
        x2, _ = self.cross_attn(
            query=x,
            key=cond,
            value=cond,
            key_padding_mask=(~c_mask) if c_mask is not None else None,
        )
        x = x + self.drop(x2)
        x = self.norm2(x)
        
        # 應用 FiLM 門控（如果啟用）
        if self.use_film:
            # 將條件向量池化為單一向量
            if c_mask is not None:
                # 使用 mask 進行加權平均
                cond_pooled = (cond * c_mask.unsqueeze(-1).float()).sum(dim=1) / c_mask.sum(dim=1, keepdim=True).float()
            else:
                cond_pooled = cond.mean(dim=1)  # [B, d_model]
            x = self.film_layer(x, cond_pooled)

        # 前饋層
        x2 = self.ff(x)
        x = x + self.drop(x2)
        x = self.norm3(x)
        return x

class DenoisingTransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 6,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 2048,
        pad_id: int = 0,
        pos_emb_type: Literal["rope", "sinusoidal", "learned"] = "rope",
        use_film: bool = True,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        self.pos_emb_type = pos_emb_type  # 位置編碼類型
        self.use_film = use_film  # 是否使用 FiLM 門控
        
        # 文字嵌入
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        
        # 位置編碼（根據類型選擇）
        if pos_emb_type == "rope":
            self.pos_emb = RoPEEmbedding(d_model, base=rope_base)
        elif pos_emb_type == "sinusoidal":
            self.pos_emb = SinusoidalPositionEmbedding(d_model, max_len)
        elif pos_emb_type == "learned":
            self.pos_emb = nn.Embedding(max_len, d_model)
        else:
            raise ValueError(f"不支援的位置編碼類型: {pos_emb_type}")
        
        # 時間步嵌入
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        # 一個線性層把時間步資訊注入 token 表徵
        self.time_proj = nn.Linear(d_model, d_model)

        # 疊 K 層 DecoderBlock
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, nhead, dim_ff, dropout, use_film, pos_emb_type) 
            for _ in range(num_layers)
        ])

        # 輸出頭：預測對 x_0 的 logits（每個 token 位置 K 類別）
        self.head = nn.Linear(d_model, vocab_size)

        self.pad_id = pad_id

    def forward(
        self,
        xt: torch.Tensor,                 # [B, L] noisy tokens（離散 id）
        t: torch.Tensor,                  # [B] 或 [1] 時間步（整數）
        cond: torch.Tensor,               # [B, S, d_model] 聲學條件
        x_mask: Optional[torch.Tensor] = None,  # [B, L] True=keep
        c_mask: Optional[torch.Tensor] = None,  # [B, S] True=keep
    ) -> torch.Tensor:
        B, L = xt.shape
        device = xt.device

        # 文字嵌入
        tok = self.tok_emb(xt)  # [B, L, d_model]
        
        # 位置編碼（根據類型處理）
        rope_cos, rope_sin = None, None
        if self.pos_emb_type == "rope":
            # RoPE 不直接加到 token embedding，而是在注意力中應用
            rope_cos, rope_sin = self.pos_emb(L, device)
            x = tok
        elif self.pos_emb_type == "sinusoidal":
            # Sinusoidal 位置編碼
            pos_emb = self.pos_emb(L, device)  # [L, d_model]
            x = tok + pos_emb.unsqueeze(0)  # [B, L, d_model]
        elif self.pos_emb_type == "learned":
            # 學習的位置編碼
            pos_ids = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
            x = tok + self.pos_emb(pos_ids)  # [B, L, d_model]

        # 時間步嵌入，Broadcast 到每個序列位置
        t_emb = self.time_emb(t)  # [B, d_model]
        t_bias = self.time_proj(t_emb).unsqueeze(1)  # [B, 1, d_model]
        x = x + t_bias  # [B, L, d_model]

        # 若未提供 mask，基於 pad_id 自動建立 x_mask
        if x_mask is None:
            x_mask = (xt != self.pad_id)

        # 通過多層解碼器區塊
        h = x
        for blk in self.blocks:
            h = blk(h, cond, x_mask, c_mask, rope_cos, rope_sin)

        # 輸出 logits
        logits = self.head(h)  # [B, L, vocab_size]
        return logits

    @torch.no_grad()
    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
                   x_mask: Optional[torch.Tensor] = None, c_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """回傳對 x_0 的機率分佈（softmax 後）。"""
        logits = self.forward(xt, t, cond, x_mask, c_mask)      # [B, L, V]
        probs = logits.softmax(dim=-1)                          # [B, L, V]
        return probs