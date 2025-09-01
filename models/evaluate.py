# -*- coding: utf-8 -*-
"""
FDDM-ASR 評估指標實作
包含 CER (Character Error Rate) 計算和相關輔助函數
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional, Any

# 匯入模型類型
from models.acoustic_encoder import AcousticEncoder
from models.denoise_decoder import DenoisingTransformerDecoder
from models.projection import SpeechProjector, TextEmbedding, TextProjector
from fddm.sched.diffusion_scheduler import DiscreteDiffusionScheduler  # 根據實際路徑調整


# ============ CER 評估指標實作 ============
def calculate_cer(pred_text: str, target_text: str) -> float:
    """
    計算字元錯誤率 (Character Error Rate)

    參數：
        pred_text: 預測文字
        target_text: 目標文字

    回傳：
        CER 值 (0.0-1.0)
    """
    # 移除空白和標點符號，轉小寫進行比較
    import re

    # 簡化文字預處理：移除空白
    pred_text = pred_text.replace(' ', '')
    target_text = target_text.replace(' ', '')

    # 計算編輯距離
    def levenshtein_distance(s1: str, s2: str) -> int:
        """使用動態規劃計算最小編輯距離"""
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


def logits_to_text(
    logits: torch.Tensor,
    tokenizer: Any,
    pad_id: int,
    bos_id: Optional[int] = None,
    eos_id: Optional[int] = None
) -> str:
    """
    從 logits 轉換為預測文字

    參數：
        logits: [L, V] 或 [B, L, V]，模型預測的 logits
        tokenizer: 文字編碼器
        pad_id: padding token ID
        bos_id: beginning of sequence token ID (可選)
        eos_id: end of sequence token ID (可選)

    回傳：
        預測的文字
    """
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


def evaluate_train_cer(
    encoder: AcousticEncoder,
    decoder: DenoisingTransformerDecoder,
    s_proj: SpeechProjector,
    t_embed: TextEmbedding,
    t_proj: TextProjector,
    scheduler: Any,  # SchedulerAdapter
    train_loader: DataLoader,
    device: torch.device,
    cfg: Any,  # Config
    tokenizer: Any,
    max_batches: int = 5  # 只計算前 max_batches 個 batch 來節省時間
) -> float:
    """
    訓練時的 CER 評估（快速版，只評估前幾個 batch）

    參數：
        encoder: 聲學編碼器
        decoder: 去噪解碼器
        s_proj: 語音投影器
        t_embed: 文字嵌入
        t_proj: 文字投影器
        scheduler: 擴散排程器
        train_loader: 訓練資料載入器
        device: 計算裝置
        cfg: 配置物件
        tokenizer: 文字編碼器
        max_batches: 最大評估 batch 數

    回傳：
        平均 CER 值
    """
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


def evaluate_cer(
    encoder: AcousticEncoder,
    decoder: DenoisingTransformerDecoder,
    s_proj: SpeechProjector,
    t_embed: TextEmbedding,
    t_proj: TextProjector,
    scheduler: Any,  # SchedulerAdapter
    loader: DataLoader,
    device: torch.device,
    cfg: Any,  # Config
    tokenizer: Any,
) -> float:
    """
    完整 CER 評估（用於驗證和測試集）

    參數：
        encoder: 聲學編碼器
        decoder: 去噪解碼器
        s_proj: 語音投影器
        t_embed: 文字嵌入
        t_proj: 文字投影器
        scheduler: 擴散排程器
        loader: 資料載入器
        device: 計算裝置
        cfg: 配置物件
        tokenizer: 文字編碼器

    回傳：
        平均 CER 值
    """
    encoder.eval()
    decoder.eval()
    s_proj.eval()
    t_embed.eval()
    t_proj.eval()

    pad_id = cfg.data['pad_id']
    total_cer = 0.0
    total_samples = 0

    with torch.no_grad():
        for wave, x0 in loader:
            wave = wave.to(device)
            x0   = x0.to(device)
            B, L = x0.shape

            # 條件向量（語音編碼），評估也要跑編碼器，但不回傳梯度
            c, c_mask, _ = encoder(wave)

            # 評估時給最小雜訊 t=1，並以 x0 當作 xt（不擾動）
            t  = torch.ones(B, dtype=torch.long, device=device)
            xt = x0.clone()

            logits = decoder(xt, t, c, x_mask=(x0!=pad_id), c_mask=c_mask)  # [B, L, V]

            # 逐樣本解碼成文字，計算 CER
            for b in range(B):
                pred_text = logits_to_text(
                    logits[b], tokenizer,
                    pad_id=pad_id,
                    bos_id=cfg.data.get('bos_id'),
                    eos_id=cfg.data.get('eos_id')
                )

                tgt_ids = x0[b].tolist()
                # 去掉 pad / bos / eos
                if pad_id in tgt_ids:
                    tgt_ids = tgt_ids[:tgt_ids.index(pad_id)]
                if cfg.data.get('bos_id') is not None and len(tgt_ids)>0 and tgt_ids[0] == cfg.data['bos_id']:
                    tgt_ids = tgt_ids[1:]
                if cfg.data.get('eos_id') is not None and len(tgt_ids)>0 and tgt_ids[-1] == cfg.data['eos_id']:
                    tgt_ids = tgt_ids[:-1]

                target_text = tokenizer.decode(tgt_ids) if tgt_ids else ''
                total_cer += calculate_cer(pred_text, target_text)
                total_samples += 1

    return (total_cer / total_samples) if total_samples > 0 else 0.0
