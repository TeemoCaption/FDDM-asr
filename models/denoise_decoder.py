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

回傳：
- logits: [B, L, vocab]，對每個位置預測 x_0 的分佈
"""
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# ---- Transformer Block，含 cross-attention ----
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
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
    ) -> torch.Tensor:
        # 自注意力（不設 causal mask，因為非自回歸）
        x2, _ = self.self_attn(x, x, x, key_padding_mask=(~x_mask) if x_mask is not None else None)
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
    ) -> None:
        super().__init__()
        # 文字嵌入與可學的位置編碼（簡單起見用 nn.Embedding）
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.time_emb = SinusoidalTimeEmbedding(d_model)

        # 一個線性層把時間步資訊注入 token 表徵
        self.time_proj = nn.Linear(d_model, d_model)

        # 疊 K 層 DecoderBlock
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)
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

        # 文字嵌入 + 位置編碼
        tok = self.tok_emb(xt)                                 # [B, L, d]
        pos_ids = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
        x = tok + self.pos_emb(pos_ids)                        # [B, L, d]

        # 時間步嵌入，Broadcast 到每個序列位置
        t_emb = self.time_emb(t)               # [B, d]
        t_bias = self.time_proj(t_emb).unsqueeze(1)  # [B, 1, d]
        x = x + t_bias                          # [B, L, d]

        # 若未提供 mask，基於 pad_id 自動建立 x_mask
        if x_mask is None:
            x_mask = (xt != self.pad_id)

        # 通過多層解碼器區塊
        h = x
        for blk in self.blocks:
            h = blk(h, cond, x_mask, c_mask)

        # 輸出 logits
        logits = self.head(h)  # [B, L, vocab]
        return logits

    @torch.no_grad()
    def predict_x0(self, xt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
                   x_mask: Optional[torch.Tensor] = None, c_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """回傳對 x_0 的機率分佈（softmax 後）。"""
        logits = self.forward(xt, t, cond, x_mask, c_mask)      # [B, L, V]
        probs = logits.softmax(dim=-1)                          # [B, L, V]
        return probs