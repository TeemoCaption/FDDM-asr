# -*- coding: utf-8 -*-
"""
L_fd: Cross-Modality Feature Decorrelation（論文第 3.2 節）
- 依據 Barlow Twins 風格：
    L_fd = Σ_j (1 - C_jj)^2 + λ Σ_j Σ_{k!=j} C_jk^2
- 這裡 C 是以 batch 維度做標準化後的 cross-correlation：
    C = (Z_a^~)^T Z_b^~
- Z_a 來自語音（h_{phi_a}(c)），Z_b 來自文字（h_{phi_b}(g_omega(\hat{x}_0)））

注意：
- 我們在 token 維度逐位置計算，然後對 (B, L) 聚合平均。
- 若要節省成本，可在訓練時隨機抽取一部分 token 位置計算。
"""
from typing import Tuple
import torch
import torch.nn as nn

def _standardize(x: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """沿 batch 維標準化：減均值/除以標準差。
    x: [B, T, D]
    回傳：x_tilde, mean, std
    """
    mean = x.mean(dim=0, keepdim=True)                 # [1, T, D]
    var = x.var(dim=0, unbiased=False, keepdim=True)   # [1, T, D]
    std = torch.sqrt(var + eps)
    x_tilde = (x - mean) / std
    return x_tilde, mean, std

def lfd_loss(
    z_a: torch.Tensor,   # [B, T, D] 語音投影後特徵（SpeechProjector 輸出）
    z_b: torch.Tensor,   # [B, T, D] 文字投影後特徵（TextProjector 輸出）
    lambda_offdiag: float = 5.0e-3,  # 調整預設值為 5e-3（與配置檔一致）
    eps: float = 1e-5,
) -> torch.Tensor:
    B, T, D = z_a.shape
    assert z_b.shape == (B, T, D), "z_b 形狀需與 z_a 相同"

    # 沿 batch 維度標準化（每個 token 位置、每個通道獨立標準化）
    za, _, _ = _standardize(z_a, eps=eps)  # [B, T, D]
    zb, _, _ = _standardize(z_b, eps=eps)  # [B, T, D]

    # 將 token 維度展平，合併到 batch 維，計算更穩定的相關：
    #   (B*T, D) 的橫向互相關 -> C: [D, D]
    za_bt = za.reshape(B * T, D)
    zb_bt = zb.reshape(B * T, D)

    # Cross-correlation: C = (Z_a^~)^T Z_b^~ / (B*T)
    C = (za_bt.T @ zb_bt) / (B * T)  # [D, D]

    # 對角線與非對角線損失
    diag = torch.diagonal(C)
    off = C - torch.diag(diag)

    loss_diag = torch.sum((1.0 - diag) ** 2)
    loss_off = torch.sum(off ** 2)

    loss = loss_diag + lambda_offdiag * loss_off
    return loss