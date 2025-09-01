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

        # 處理邊界條件：當 t=1 時，t-1=0 對應 M₀=I（β₀=0 ⇒ a₀=1, b₀=0）
        t_prev = t - 1  # t_prev 範圍是 [0, T-1]
        
        # 對於 t_prev=0 的情況（即 t=1），設定 a_{t-1}=1, b_{t-1}=0
        # 對於 t_prev>0 的情況，使用 betas[t_prev-1]（因為 betas 是 1-indexed）
        mask_t_prev_zero = (t_prev == 0).view(B, 1, 1)  # (B,1,1)
        
        # 當 t_prev > 0 時，取 betas[t_prev-1]；當 t_prev = 0 時，設為 0
        beta_tm1 = torch.where(
            mask_t_prev_zero,
            torch.zeros_like(beta_t),  # t_prev=0 時，β₀=0
            self.betas[torch.clamp(t_prev - 1, min=0)].view(B, 1, 1)  # t_prev>0 時，取對應的 beta
        )
        
        a_tm1 = 1.0 - beta_tm1  # t_prev=0 時為 1，其他時候為 1-β_{t-1}
        b_tm1 = beta_tm1 / self.K  # t_prev=0 時為 0，其他時候為 β_{t-1}/K

        ones = torch.ones_like(xt_prob)                                # (B,L,K)

        # A = M_t^T x_t = a_t * x_t + b_t * 1
        A = a_t * xt_prob + b_t * ones

        # Bv = M_{t-1} x̂0 = a_{t-1} * x̂0 + b_{t-1} * 1
        Bv = a_tm1 * x0hat_prob + b_tm1 * ones

        # denom = x_t^T M_t x̂0 = a_t * (x_t · x̂0) + b_t * 1
        dot = (xt_prob * x0hat_prob).sum(dim=-1, keepdim=True)         # (B,L,1)
        denom = a_t * dot + b_t                                        # (B,L,1)
        
        # 後驗機率：q(x_{t-1}|x_t, x̂0) = (A * Bv) / denom
        posterior = (A * Bv) / denom.clamp_min(self.eps)               # (B,L,K)
        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(self.eps)  # 歸一化
        
        return posterior

    @torch.no_grad()
    def q_posterior_multi_step(self, xt_prob: torch.Tensor, x0hat_prob: torch.Tensor, t: torch.Tensor, delta: int) -> torch.Tensor:
        """
        多步後驗：q(x_{t-Δ}|x_t, x̂0) 的精確計算（基於論文 Algorithm 2）
        使用轉移矩陣積 M_{t:t-Δ+1} 的閉式表達式，實現嚴格的多步跳躍
        
        參數說明：
        xt_prob    : (B, L, K) x_t 的機率分佈
        x0hat_prob : (B, L, K) 預測的 x̂0 機率分佈  
        t          : (B,) 當前步數 (1..T)
        delta      : int 要跳的步數 (1..t)
        
        回傳：
        posterior  : (B, L, K) x_{t-Δ} 的機率分佈
        
        數學原理：
        q(x_{t-Δ}|x_t, x̂0) ∝ (M_{t:t-Δ+1}^T x_t) ⊙ (M_{t-Δ} x̂0) / (x_t^T M_{t:t-Δ+1} x̂0)
        其中 M_s = (1-β_s)I + (β_s/K)11^T，⊙ 表示 Hadamard 乘積
        """
        xt_prob = xt_prob.to(self.device).float()
        x0hat_prob = x0hat_prob.to(self.device).float()
        t = t.to(self.device).long()
        B, L, K = xt_prob.shape
        assert K == self.K
        
        # 邊界檢查：確保 delta 不超過當前步數
        delta = min(delta, t.min().item())
        if delta <= 0:
            return xt_prob
            
        # 目標步數 t_target = t - delta
        t_target = torch.clamp(t - delta, min=0)
        
        # 計算多步轉移矩陣積 M_{t:t-Δ+1} 的閉式係數
        # 對於每個 batch element，計算其對應的轉移矩陣積
        a_cumulative = torch.ones(B, 1, 1, device=self.device, dtype=torch.float32)  # 累積的對角項係數
        b_cumulative = torch.zeros(B, 1, 1, device=self.device, dtype=torch.float32)  # 累積的均勻項係數
        
        # 對每個可能的步數進行迭代計算
        for batch_idx in range(B):
            t_current = t[batch_idx].item()
            t_end = t_target[batch_idx].item()
            
            # 從 t_current 向下到 t_end+1 逐步累積轉移矩陣
            for step in range(t_current, t_end, -1):
                if step >= 1 and step <= self.T:
                    beta_s = self.betas[step - 1]  # step 是 1-indexed
                    a_s = 1.0 - beta_s  # 對角項係數
                    b_s = beta_s / self.K  # 均勻項係數
                    
                    # 矩陣積更新：M_new = M_s @ M_old
                    # 新係數計算：
                    # a_new = a_s * a_old
                    # b_new = a_s * b_old + b_s * (a_old + K * b_old)
                    a_old = a_cumulative[batch_idx, 0, 0]
                    b_old = b_cumulative[batch_idx, 0, 0]
                    
                    a_cumulative[batch_idx, 0, 0] = a_s * a_old
                    b_cumulative[batch_idx, 0, 0] = a_s * b_old + b_s * (a_old + self.K * b_old)
        
        # 計算 M_{t-Δ} 的係數（用於 x̂0 項）
        # 處理邊界情況：當 t_target = 0 時，M_0 = I（即 a=1, b=0）
        mask_target_zero = (t_target == 0).view(B, 1, 1)  # (B,1,1)
        
        a_target = torch.ones(B, 1, 1, device=self.device, dtype=torch.float32)
        b_target = torch.zeros(B, 1, 1, device=self.device, dtype=torch.float32)
        
        # 對於 t_target > 0 的情況，計算對應的 M_{t-Δ} 係數
        for batch_idx in range(B):
            t_tgt = t_target[batch_idx].item()
            if t_tgt > 0 and t_tgt <= self.T:
                beta_tgt = self.betas[t_tgt - 1]  # t_tgt 是 1-indexed
                a_target[batch_idx, 0, 0] = 1.0 - beta_tgt
                b_target[batch_idx, 0, 0] = beta_tgt / self.K
        
        # 應用邊界條件：t_target=0 時保持 a=1, b=0
        a_target = torch.where(mask_target_zero, torch.ones_like(a_target), a_target)
        b_target = torch.where(mask_target_zero, torch.zeros_like(b_target), b_target)
        
        # 計算後驗機率的各個組件
        ones = torch.ones_like(xt_prob)  # (B,L,K)
        
        # A = M_{t:t-Δ+1}^T @ x_t = a_cumulative * x_t + b_cumulative * (1^T @ x_t) * 1
        sum_xt = xt_prob.sum(dim=-1, keepdim=True)  # (B,L,1)
        A = a_cumulative * xt_prob + b_cumulative * sum_xt * ones  # (B,L,K)
        
        # B = M_{t-Δ} @ x̂0 = a_target * x̂0 + b_target * (1^T @ x̂0) * 1  
        sum_x0hat = x0hat_prob.sum(dim=-1, keepdim=True)  # (B,L,1)
        B_term = a_target * x0hat_prob + b_target * sum_x0hat * ones  # (B,L,K)
        
        # 分母：x_t^T @ M_{t:t-Δ+1} @ x̂0
        # = x_t^T @ (a_cumulative * x̂0 + b_cumulative * sum(x̂0) * 1)
        # = a_cumulative * (x_t · x̂0) + b_cumulative * sum(x̂0) * sum(x_t)
        dot_xt_x0hat = (xt_prob * x0hat_prob).sum(dim=-1, keepdim=True)  # (B,L,1)
        denom = a_cumulative * dot_xt_x0hat + b_cumulative * sum_x0hat * sum_xt  # (B,L,1)
        
        # 後驗機率：q(x_{t-Δ}|x_t, x̂0) = (A ⊙ B) / denom
        posterior = (A * B_term) / denom.clamp_min(self.eps)  # (B,L,K)
        
        # 歸一化確保機率分佈有效
        posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        
        return posterior

    @property
    def w_prefix(self):
        """w_t = ∏_{s=1}^t (1-β_s)，給 L_fd 權重用（shape: [T] on device）。"""
        return self.alpha_bar
