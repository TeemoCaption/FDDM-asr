# -*- coding: utf-8 -*-
"""
AcousticEncoder: WavLM-large 封裝
- 論文第 3.1 節對應：c = c_psi(s)
- 功能：將 raw waveform -> WavLM hidden states -> (可選)投影到 d_model，
        供解碼器 cross-attention 使用。

安裝相依：
    pip install transformers torchaudio

重要參數：
- wavlm_name: HF 權重名稱 (預設 'microsoft/wavlm-large')
- freeze: 是否凍結參數 (預設 True，論文做法是 frozen acoustic encoder)
- d_model: 目標維度，需與解碼器一致
- proj: 'linear' | 'none'，若 'linear' 會加一層線性投影把 WavLM hidden_size 對齊 d_model
- pooling: 'none' | 'mean'，若需要句向量可用 'mean' 取得 (選用)

回傳：
- features: Tensor[B, S, d_model] 供 cross-attention
- feat_mask: Optional[B, S]，時間軸 mask (True=可用, False=padding)
- pooled: Optional[B, d_model]，若 pooling != 'none'
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn

try:
    from transformers import WavLMModel
except Exception as e:
    raise ImportError(
        "需要 transformers 套件。請先安裝：pip install transformers"
    )

class AcousticEncoder(nn.Module):
    def __init__(
        self,
        wavlm_name: str = "microsoft/wavlm-large",
        freeze: bool = True,
        d_model: int = 768,
        proj: str = "linear",
        pooling: str = "none",
    ) -> None:
        super().__init__()
        # 讀取預訓練 WavLM
        self.backbone = WavLMModel.from_pretrained(wavlm_name)
        hidden = self.backbone.config.hidden_size

        # 是否凍結 WavLM 參數
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # 將 WavLM hidden 對齊至 d_model，供解碼器使用
        self.use_proj = (proj == "linear") and (hidden != d_model)
        self.proj = nn.Linear(hidden, d_model) if self.use_proj else nn.Identity()

        # 句向量 pooling（選用）
        assert pooling in {"none", "mean"}
        self.pooling = pooling

    @torch.no_grad()
    def _make_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """建立時間軸 mask（True 表示有效位置）。
        lengths: [B] WavLM frame 數（已經轉換後的長度）
        max_len: 最大 frame 數（通常是 batch 中的 S）
        """
        device = lengths.device
        # [B, S]，每列 0..len-1 為 True
        ids = torch.arange(max_len, device=device).unsqueeze(0)  # [1, S]
        mask = ids < lengths.unsqueeze(1)  # [B, 1] < [1, S] -> [B, S]
        return mask
    
    def _compute_wavlm_frame_length(self, waveform_length: torch.Tensor) -> torch.Tensor:
        """計算 WavLM 輸出的 frame 數量。
        WavLM-large 使用 320 點 hop_length，但實際上可能有微調。
        這裡使用保守的估算方式。
        """
        # WavLM 的下採樣比率約為 320 (根據 transformers 實現)
        # 但為了穩定性，我們使用保守估算
        hop_length = 320
        frame_length = (waveform_length + hop_length - 1) // hop_length  # 向上取整
        return frame_length

    def forward(
        self,
        waveforms: torch.Tensor,                # [B, T]，16kHz raw waveform
        lengths: Optional[torch.Tensor] = None, # [B]，實際長度（樣本點數），可選
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        B, T = waveforms.shape
        device = waveforms.device
        
        # 建立 waveform 的 attention_mask 給 WavLM backbone
        attention_mask = None
        if lengths is not None:
            # 建立 waveform 層級的 attention_mask
            attention_mask = self._make_mask(lengths, T)  # [B, T]
        
        # WavLM forward，傳入 attention_mask 確保 padding 區域不被處理
        out = self.backbone(
            waveforms, 
            attention_mask=attention_mask,  # 關鍵修正：傳入 mask
            output_hidden_states=False
        )
        feats = out.last_hidden_state  # [B, S, hidden]

        # 投射到 d_model
        feats = self.proj(feats)       # [B, S, d_model]

        # 建立特徵層級的 mask（如果給了 lengths）
        feat_mask = None
        if lengths is not None:
            B, S, _ = feats.shape
            # 正確計算每個樣本的 WavLM frame 數量
            feat_lengths = self._compute_wavlm_frame_length(lengths)  # [B]
            # 確保不超過實際輸出長度
            feat_lengths = torch.clamp(feat_lengths, max=S)
            feat_mask = self._make_mask(feat_lengths, S)  # [B, S] 正確的 mask

        pooled = None
        if self.pooling == "mean":
            if feat_mask is None:
                pooled = feats.mean(dim=1)  # [B, d_model]
            else:
                # mask 平均：只對有效位置取平均
                denom = feat_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)  # [B,1]
                pooled = (feats * feat_mask.unsqueeze(-1)).sum(dim=1) / denom

        return feats, feat_mask, pooled