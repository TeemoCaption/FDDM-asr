# -*- coding: utf-8 -*-
"""
投影模組 & 文字嵌入抽取
- h_{phi_a}: SpeechProjector，把聲學條件映射到共同子空間
- g_omega:   TextEmbedding，將解碼器對 x_0 的預測（logits/probs）轉成向量（可用 logit 特徵）
- h_{phi_b}: TextProjector，把文字向量映射到共同子空間

這些投影會用在 L_fd（特徵去相關）計算時，對應論文第 3.2 節。
"""
from typing import Literal
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, hidden: int = 0, act: Literal['gelu','relu']='gelu'):
        super().__init__()
        layers = []
        if hidden > 0:
            layers += [nn.Linear(dim_in, hidden), nn.GELU() if act=='gelu' else nn.ReLU(), nn.Linear(hidden, dim_out)]
        else:
            layers += [nn.Linear(dim_in, dim_out)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class SpeechProjector(nn.Module):
    def __init__(self, d_in: int, d_proj: int, hidden: int = 0):
        super().__init__()
        self.proj = MLP(d_in, d_proj, hidden)
    def forward(self, c: torch.Tensor) -> torch.Tensor:
        # c: [B, S, d_in] -> [B, S, d_proj]
        return self.proj(c)

class TextEmbedding(nn.Module):
    def __init__(self, vocab: int, d_out: int, mode: Literal['logits','probs']='logits'):
        super().__init__()
        self.vocab = vocab
        self.mode = mode
        # 用一層線性把 one-hot/soft 分佈映射到 d_out，等價於詞嵌入學習（但輸入為分佈）
        self.proj = nn.Linear(vocab, d_out, bias=False)
    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        # dist: [B, L, V]（logits 或 機率）-> 先轉成機率分佈再線性投影
        if self.mode == 'logits':
            probs = dist.softmax(dim=-1)
        else:
            probs = dist
        return self.proj(probs)  # [B, L, d_out]

class TextProjector(nn.Module):
    def __init__(self, d_in: int, d_proj: int, hidden: int = 0):
        super().__init__()
        self.proj = MLP(d_in, d_proj, hidden)
    def forward(self, z_text: torch.Tensor) -> torch.Tensor:
        # z_text: [B, L, d_in] -> [B, L, d_proj]
        return self.proj(z_text)