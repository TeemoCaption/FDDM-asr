# -*- coding: utf-8 -*-
"""
最小可跑的 sanity check：
- 生成隨機 waveform 與隨機 noisy tokens x_t
- 通過 AcousticEncoder 取得 cond c
- 通過 DenoisingTransformerDecoder 取得對 x_0 的 logits
- 以 projection + L_fd 計算一次 decorrelation loss（僅為驗證，實際訓練時會按論文的 wt 週期混合）

執行：
    python scripts/sanity_forward.py
"""
import torch
from models.acoustic_encoder import AcousticEncoder
from models.denoise_decoder import DenoisingTransformerDecoder
from models.projection import SpeechProjector, TextEmbedding, TextProjector
from losses.fddm_losses import lfd_loss

# 超參數（請與 configs/ 中的設計對齊）
B = 2            # batch size
T_wav = 16000*2  # 每段 2 秒，16kHz
L_tok = 64       # token 長度（padding 後）
V = 4000         # 詞彙表大小（依你的 tokenizer 調整）
D = 768          # 模型維度

# 1) 準備隨機輸入（真實訓練時請改為資料載入器）
wave = torch.randn(B, T_wav)  # [B, T]
xt = torch.randint(low=0, high=V, size=(B, L_tok))  # [B, L]

# 2) 聲學編碼器
enc = AcousticEncoder(
    wavlm_name="microsoft/wavlm-large", freeze=True, d_model=D, proj="linear", pooling="none"
)
with torch.no_grad():
    c, c_mask, _ = enc(wave)  # c: [B, S, D]

# 3) 去噪解碼器（非自回歸）
dec = DenoisingTransformerDecoder(
    vocab_size=V, d_model=D, nhead=12, num_layers=4, dim_ff=2048, dropout=0.1, max_len=L_tok, pad_id=0
)
# 單一時間步 t 的測試（實際訓練會隨機抽樣 t）
t = torch.randint(low=1, high=200, size=(B,))
logits = dec(xt, t, c, x_mask=None, c_mask=None)  # [B, L, V]
print("logits:", logits.shape)

# 4) 特徵去相關（僅示範計算）
s_proj = SpeechProjector(d_in=D, d_proj=256, hidden=0)
te = TextEmbedding(vocab=V, d_out=256, mode='logits')
t_proj = TextProjector(d_in=256, d_proj=256, hidden=0)

with torch.no_grad():
    # 將 decoder 對 x_0 的預測（logits）轉為文字嵌入
    z_text = te(logits)                 # [B, L, 256]
    z_text = t_proj(z_text)             # [B, L, 256]
    z_speech = s_proj(c)                # [B, S, 256]

    # 將語音序列對齊到文字長度（簡化：截斷或插值；實務建議用 attention 對齊或平均）
    # 這裡為簡化示範：只取前 L_tok 個 frame
    S = z_speech.size(1)
    if S >= L_tok:
        z_speech_aligned = z_speech[:, :L_tok, :]
    else:
        # 若語音序列較短，重複最後一幾步到 L_tok
        pad = z_speech[:, -1:, :].repeat(1, L_tok - S, 1)
        z_speech_aligned = torch.cat([z_speech, pad], dim=1)

    # 計算 L_fd
    loss_fd = lfd_loss(z_speech_aligned, z_text, lambda_offdiag=1.0)
    print("L_fd:", float(loss_fd))

print("Sanity check 完成。")