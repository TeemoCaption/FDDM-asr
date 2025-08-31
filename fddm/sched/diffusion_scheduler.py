# fddm/sched/diffusion_scheduler.py
# 說明：
#  - 依論文的離散擴散（Multinomial/Discrete）實作均勻轉移矩陣 M_t（式(8)）
#  - β_t 採 cosine 噪聲排程；提供 q(x_t|x_0) 與 q(x_{t-1}|x_t,x̂0) 的向量化計算（式(1)(4)(7)）
#  - 回傳分佈皆為 (batch, length, K) 的機率向量；另外提供 w_t（式(13)）
# 參數：
#  - K (int): 類別數（= tokenizer vocab_size）
#  - T (int): 擴散總步數（訓練建議 200）
#  - beta_max (float): cos 排程的上限幅度（可先 0.2）
#  - device: torch 裝置
# 備註：
#  - 需安裝 torch

# 低記憶體版：不建立 T×K×K；用 (1-β)I + (β/K)11^T 的封閉解。
import math
from typing import Tuple
import torch

class DiscreteDiffusionScheduler:
    def __init__(self, K: int, T: int, device: torch.device,
                 beta_max: float = 0.2, eps: float = 1e-8):
        self.K = int(K)
        self.T = int(T)
        self.device = device
        self.eps = float(eps)

        t = torch.arange(1, T + 1, device=device, dtype=torch.float32)
        # cosine 調度：β_t = beta_max * sin^2( (t/T)*π/2 )
        self.betas = beta_max * torch.sin(0.5 * math.pi * (t / float(T)))**2   # [T]
        # ᾱ_t = ∏_{s=1}^t (1 - β_s)
        self.alpha_bar = torch.cumprod(1.0 - self.betas, dim=0)               # [T]

    @torch.no_grad()
    def q_sample(self, x0_prob: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        前向：q(x_t|x_0) = ᾱ_t x0 + (1-ᾱ_t) u；u = 1/K * 1
        x0_prob: (B, L, K)  one-hot 或機率分佈
        t      : (B,)       時間步 (1..T)
        回傳   : (B, L, K)
        """
        x0_prob = x0_prob.to(self.device).float()
        t = t.to(self.device).long()
        B, L, K = x0_prob.shape
        assert K == self.K

        alpha_bar = self.alpha_bar[t - 1].view(B, 1, 1)               # (B,1,1)
        u = torch.full_like(x0_prob, 1.0 / self.K)                    # (B,L,K)

        xt = alpha_bar * x0_prob + (1.0 - alpha_bar) * u
        xt = xt.clamp_min(self.eps)
        xt = xt / xt.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return xt

    @torch.no_grad()
    def q_posterior(self, xt_prob: torch.Tensor, x0hat_prob: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        後驗：q(x_{t-1}|x_t, x̂0) 的向量化計算（只用標量 a=1-β, b=β/K）
        xt_prob    : (B, L, K)
        x0hat_prob : (B, L, K)
        t          : (B,)
        回傳       : (B, L, K)
        """
        xt_prob = xt_prob.to(self.device).float()
        x0hat_prob = x0hat_prob.to(self.device).float()
        t = t.to(self.device).long()
        B, L, K = xt_prob.shape
        assert K == self.K

        beta_t = self.betas[t - 1].view(B, 1, 1)                       # (B,1,1)
        a_t = 1.0 - beta_t
        b_t = beta_t / self.K

        t_prev = torch.clamp(t - 2, min=0)
        beta_tm1 = self.betas[t_prev].view(B, 1, 1)                    # (B,1,1)
        a_tm1 = 1.0 - beta_tm1
        b_tm1 = beta_tm1 / self.K

        ones = torch.ones_like(xt_prob)                                # (B,L,K)

        # A = M_t^T x_t = a_t * x_t + b_t * 1
        A = a_t * xt_prob + b_t * ones

        # Bv = M_{t-1} x̂0 = a_{t-1} * x̂0 + b_{t-1} * 1
        Bv = a_tm1 * x0hat_prob + b_tm1 * ones

        # denom = x_t^T M_t x̂0 = a_t * (x_t · x̂0) + b_t * 1
        dot = (xt_prob * x0hat_prob).sum(dim=-1, keepdim=True)         # (B,L,1)
        denom = a_t * dot + b_t                                        # (B,L,1)

        post = (A * Bv) / denom.clamp_min(self.eps)
        post = post / post.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return post

    @property
    def w_prefix(self):
        """w_t = ∏_{s=1}^t (1-β_s)，給 L_fd 權重用（shape: [T] on device）。"""
        return self.alpha_bar
