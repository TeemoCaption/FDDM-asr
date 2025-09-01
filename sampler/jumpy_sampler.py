# -*- coding: utf-8 -*-
"""
Jumpy Sampling（跳步還原）與 Greedy 解碼器
=================================================
放置路徑：sampler/jumpy_sampler.py

功能概要
-------
本模組提供「離散擴散（multinomial/discrete diffusion）」在推論期的
快速取樣（jumpy sampling）實作。其核心思想是：在步數 t 的狀態 x_t 下，
用去噪解碼器（Transformer Decoder）預測 x̂₀ 的機率分佈，然後直接近似
q(x_{t-Δ} | x_t, x̂₀) 來一次跳回 t-Δ，而非逐步 t→t-1 的完整還原。

相依介面（請依你專案版面微調）
------------------------------
- DiffusionScheduler（fddm.sched.diffusion_scheduler.DiffusionScheduler）
  需提供：
    - K: 類別數（詞彙表大小）
    - T: 總擴散步數（訓練期可為 200；推論可設更小，如 20）
    - alpha_bar: 長度 T+1 的張量/陣列，1..T 的累積保留率（cosine 噪聲排程）
- DenoiseDecoder（models.denoise_decoder.DenoiseDecoder）
  預期 forward 介面（若不同，請於 ModelAdapter 改成你實際的呼叫）：
      logits_x0 = decoder(x_t_idx, t, cond_c)
  其中：
    - x_t_idx: LongTensor [B, L]，每個位置是 0..K-1 的類別索引
    - t: LongTensor [B] 或 [B, 1]，目前的擴散步數
    - cond_c: Tensor [B, N, D]，聲學條件（例如 WavLM-Large 的輸出）
  回傳：
    - logits_x0: FloatTensor [B, L, K]，每個位置對「原始乾淨序列 x₀」的 logits

使用重點
--------
1) Δ（r）為每次要跳的步長（例如 r=2 或 r=5）。
2) 若 greedy=True，則每次根據 p(x_{t-Δ} | x_t, x̂₀) 取 argmax（可視為 MAP）。
   若 greedy=False，則會以 Categorical 抽樣（可加上 temperature）。
3) 若要「平均後驗/最大後驗」兩種策略切換，可用 posterior_mode：
   - "average": 近似使用 mix_with_uniform(p(x̂₀), ᾱ_{t-Δ}) 作為 x_{t-Δ} 分佈（忽略 x_t）
   - "max":     對平均後驗再取 argmax，相當於 Maximum Posterior 的近似

學術對齊
--------
- 本實作遵循原論文的「快速取樣」直覺：依據模型對 x̂₀ 的估計，透過關於
  ᾱ_t 的閉式混合（與均勻分布）來近似後驗，達成「跳步」還原與 Greedy 解碼。 
  （你在專案路線圖中規劃的 T=200 訓練、推論 T<<200、r=2/5 的設定可直接套用）
"""

from typing import Optional, Literal, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical


class ModelAdapter:
    """
    將你的去噪解碼器統一成 predict_x0_logits(x_t_idx, t, cond_c) 介面。

    若你的 decoder.forward 介面不同，請在此改寫呼叫方式即可。
    """
    def __init__(self, decoder):
        self.decoder = decoder

    @torch.no_grad()
    def predict_x0_logits(self, x_t_idx: Tensor, t: Tensor, cond_c: Tensor) -> Tensor:
        """
        參數
        ----
        x_t_idx : LongTensor [B, L]
            目前擴散步（t）的離散序列索引（每位置 0..K-1）
        t : LongTensor [B] 或 [B, 1]
            當前步數
        cond_c : Tensor [B, N, D]
            聲學條件向量（例：WavLM-Large 的輸出）

        回傳
        ----
        logits_x0 : FloatTensor [B, L, K]
            對原始乾淨序列 x₀ 的 logits 預測
        """
        # 預期你的 decoder.forward(x_t_idx, t, cond_c) 即可使用；
        # 若你的 forward 需要嵌入或 one-hot，請在此包裝。
        logits_x0 = self.decoder(x_t_idx, t, cond_c)
        return logits_x0


class DiffusionJumpySampler:
    """
    Multinomial/離散擴散的 Jumpy Sampling 實作，支援精確模式和快速模式。

    重要超參
    --------
    - T_infer: 推論使用的總步數，通常遠小於訓練時的 T（如 20 vs 200）
    - r:      每次跳步長（Δ），如 2 或 5
    - greedy: True 時以 argmax 取樣；False 時使用 Categorical 抽樣
    - posterior_mode: "average" or "max"，對應「平均後驗 / 最大後驗」兩種近似
    - sampling_mode: "exact" or "fast"，控制使用精確或快速採樣模式
    - temperature: 取樣溫度；>1 更隨機，<1 更保守（僅在 greedy=False 時生效）

    採樣模式說明：
    - "exact": 使用論文 Algorithm 2 的嚴格多步後驗 q(x_{t-Δ}|x_t,x̂0)，包含完整的 M^Δ 項
    - "fast": 使用 ᾱ_t 與均勻分布凸組合的快速近似，計算效率更高但精度略低
    """

    def __init__(
        self,
        scheduler,
        decoder,            # 你的 DenoiseDecoder 實例
        K: int,
        T_train: int,
        T_infer: int,
        r: int = 2,
        greedy: bool = True,
        posterior_mode: Literal["average", "max"] = "average",
        sampling_mode: Literal["exact", "fast"] = "exact",  # 新增：控制採樣精確度
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.scheduler = scheduler
        self.model = ModelAdapter(decoder)
        self.K = int(K)
        self.T_train = int(T_train)
        self.T_infer = int(T_infer)
        self.r = int(r)
        self.greedy = bool(greedy)
        self.posterior_mode = posterior_mode
        self.sampling_mode = sampling_mode  # 採樣模式控制
        self.temperature = float(temperature)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 需要 ᾱ_t（t=0..T_train），以下以 1-based 的慣例存放（方便對照論文）
        alpha_bar = getattr(self.scheduler, "alpha_bar", None)
        if alpha_bar is None:
            raise ValueError("scheduler 需提供 alpha_bar（形如長度 T+1 的張量，index=0 保留）。")
        self.alpha_bar = torch.as_tensor(alpha_bar, dtype=torch.float32, device=self.device)

    # ---------------------------
    # 一些實用小工具
    # ---------------------------
    def _mix_with_uniform(self, p_x0: Tensor, alpha_bar_t: Tensor) -> Tensor:
        """
        快速模式：依 ᾱ_t 與均勻分布做凸組合近似後驗
            q(x_{t-Δ} | x₀) ≈ ᾱ_{t-Δ} * p(x₀) + (1 - ᾱ_{t-Δ}) * U
        其中 U 為 K 類均勻分佈；此處 p(x₀) 用模型 softmax(logits_x0) 近似。
        這是原始快速採樣的近似方法，計算效率高但忽略了 x_t 的信息。
        """
        B, L, K = p_x0.shape
        u = torch.full((1, 1, K), 1.0 / K, device=p_x0.device, dtype=p_x0.dtype)
        # alpha_bar_t 允許 shape 為 [B] 或純 scalar（擴展到 [B,1,1] 以便廣播）
        if alpha_bar_t.ndim == 1:
            alpha_bar_t = alpha_bar_t[:, None, None]
        return alpha_bar_t * p_x0 + (1.0 - alpha_bar_t) * u

    def _to_indices(self, probs: Tensor) -> Tensor:
        """將類別分佈轉為索引（greedy argmax 或 Categorical 抽樣）。"""
        if self.greedy:
            return probs.argmax(dim=-1)
        else:
            if self.temperature != 1.0:
                logits = (probs.clamp_min(1e-12).log()) / self.temperature
                probs = F.softmax(logits, dim=-1)
            cat = Categorical(probs=probs)
            return cat.sample()

    # ---------------------------
    # 單次「跳 Δ 步」的核心
    # ---------------------------
    @torch.no_grad()
    def _jump_once(
        self,
        x_t_idx: Tensor,    # [B, L] 目前的離散索引
        t_scalar: int,      # 目前步數（int, 1..T_infer）
        delta: int,         # 本次要跳的步數（通常等於 r；若 t 不足則為 t）
        cond_c: Tensor,     # [B, N, D] 聲學條件
        seq_len: int,       # 序列長度
    ) -> Tuple[Tensor, Tensor]:
        """
        單次跳躍採樣，支援精確模式和快速模式
        
        回傳：
            x_t_minus_delta_idx: [B, L]，跳後的新索引
            p_x0: [B, L, K]，本次模型的 x̂₀ 概率（可在最後用來 Greedy 解碼）
        """
        device = x_t_idx.device
        B = x_t_idx.size(0)

        # 以目前 x_t 預測 x̂₀ 的分佈
        t_tensor = torch.full((B,), t_scalar, device=device, dtype=torch.long)
        logits_x0 = self.model.predict_x0_logits(x_t_idx, t_tensor, cond_c)  # [B, L, K]
        p_x0 = F.softmax(logits_x0, dim=-1)  # [B, L, K]

        # 根據採樣模式選擇不同的後驗計算方法
        if self.sampling_mode == "exact":
            # 精確模式：使用論文 Algorithm 2 的嚴格多步後驗
            # 將離散索引轉換為機率分佈格式（one-hot）
            xt_onehot = torch.zeros(B, seq_len, self.K, device=device)
            xt_onehot.scatter_(-1, x_t_idx.unsqueeze(-1), 1.0)  # [B, L, K]
            
            # 使用精確的多步後驗方法 q(x_{t-Δ}|x_t, x̂₀)
            # 包含完整的轉移矩陣積 M_{t:t-Δ+1} 項，對應論文式(4)(5)的多步推廣
            p_xtmd = self.scheduler.q_posterior_multi_step(
                xt_onehot, p_x0, t_tensor, delta
            )  # [B, L, K]
            
        else:  # sampling_mode == "fast"
            # 快速模式：使用 ᾱ_t 與均勻分布的凸組合近似
            # 這忽略了 x_t 的具體信息，但計算效率更高
            target_t = max(0, t_scalar - delta)
            alpha_bar_target = self._alpha_bar_at_t_train(target_t)
            p_xtmd = self._mix_with_uniform(p_x0, alpha_bar_target)  # [B, L, K]

        # 根據 posterior_mode 決定取樣策略
        if self.posterior_mode == "max":
            x_t_minus_delta_idx = p_xtmd.argmax(dim=-1)
        else:
            x_t_minus_delta_idx = self._to_indices(p_xtmd)

        return x_t_minus_delta_idx, p_x0

    def _alpha_bar_at_t_train(self, t_infer_scalar: int) -> Tensor:
        """
        將推論用的 t（1..T_infer）對映回訓練時間軸 1..T_train 的比例位置，
        再取對應的 ᾱ（線性插值）。

        例如：T_train=200、T_infer=20、t_infer=10，
              對映到 t_train=100，取 alpha_bar[100]。
        """
        if t_infer_scalar <= 0:
            # 定義 ᾱ₀ = 1（無雜訊）
            return torch.tensor(1.0, device=self.device, dtype=torch.float32)
        ratio = float(t_infer_scalar) / float(max(1, self.T_infer))
        t_train_float = ratio * float(self.T_train)
        # 夾住到 [1, T_train]
        t_train_float = max(1.0, min(float(self.T_train), t_train_float))
        # 做最簡單的線性取整（也可改線性插值）
        t_train_idx = int(round(t_train_float))
        return self.alpha_bar[t_train_idx]  # scalar tensor（之後會擴成 [B,1,1]）

    # ---------------------------
    # 主要外部介面
    # ---------------------------
    @torch.no_grad()
    def sample(
        self,
        cond_c: Tensor,         # [B, N, D] 聲學條件（WavLM-Large 輸出）
        seq_len: int,           # 輸出文字序列長度（可用訓練資料的平均長，或 beam 外掛長度模型）
        init: Literal["uniform", "random"] = "uniform",
    ) -> Tuple[Tensor, Tensor]:
        """
        以 Jumpy Sampling 從 x_T 逐步跳到 x_0，支援精確和快速兩種模式。

        參數
        ----
        cond_c : Tensor [B, N, D]
            聲學條件（WavLM-Large 輸出）
        seq_len : int
            輸出文字序列長度
        init : Literal["uniform", "random"]
            初始化方式

        回傳
        ----
        x_0_idx : LongTensor [B, L]
            還原後的最終序列索引（可直接用 tokenizer decode）
        p_x0_last : FloatTensor [B, L, K]
            最後一次模型對 x̂₀ 的分佈（常拿來做 Greedy 解碼或計分）

        採樣模式說明
        -----------
        - exact: 使用論文 Algorithm 2 的嚴格多步後驗，精度高但計算量大
        - fast: 使用 ᾱ_t 近似，計算效率高但可能精度略低
        """
        B = cond_c.size(0)
        device = cond_c.device

        # 0) 初始化 x_T
        if init == "uniform":
            # 以 K 類均勻機率抽樣索引
            x_t_idx = torch.randint(low=0, high=self.K, size=(B, seq_len), device=device)
        else:  # "random"
            x_t_idx = torch.randint(low=0, high=self.K, size=(B, seq_len), device=device)

        t = self.T_infer
        p_x0_last = None

        while t > 0:
            delta = min(self.r, t)
            x_t_idx, p_x0_last = self._jump_once(x_t_idx, t_scalar=t, delta=delta, cond_c=cond_c, seq_len=seq_len)
            t -= delta

        # 最終輸出處理：對 p_x0_last 取 argmax 作為最終序列
        # 這確保了最終輸出是基於模型對 x̂₀ 的最佳預測
        x_0_idx = p_x0_last.argmax(dim=-1)
        return x_0_idx, p_x0_last
    
    def get_sampling_info(self) -> dict:
        """
        回傳當前採樣器的配置信息，便於調試和記錄
        """
        return {
            "sampling_mode": self.sampling_mode,
            "posterior_mode": self.posterior_mode,
            "T_infer": self.T_infer,
            "r": self.r,
            "greedy": self.greedy,
            "temperature": self.temperature,
            "K": self.K
        }
