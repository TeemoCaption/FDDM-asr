# models/evaluate.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple
import torch

def _ids_to_text_one(
    ids_tensor: torch.Tensor,
    tokenizer,
    pad_id: int,
    bos_id: int | None = None,
    eos_id: int | None = None,
) -> str:
    """
    將單一序列的 token ids 轉成文字（安全版）。
    參數：
        ids_tensor: [L] 的 long Tensor（可在 GPU）
        tokenizer : sentencepiece.SentencePieceProcessor
        pad_id    : <pad> 的 id（會被略過）
        bos_id    : <bos> 的 id（若提供會被略過）
        eos_id    : <eos> 的 id（若提供遇到就停止）
    備註：
        - 一律轉成 Python list[int] 再解碼，避免 RuntimeError: unknown output or input type
        - 使用 DecodeIds 而非 Decode，更明確指定輸入型別
    """
    # 轉到 CPU，確保是 Python list[int]
    ids_list: List[int] = ids_tensor.detach().to("cpu").tolist()

    # 過濾特殊 token
    clean_ids: List[int] = []
    for tid in ids_list:
        if tid == pad_id:
            continue
        if bos_id is not None and tid == bos_id:
            continue
        if eos_id is not None and tid == eos_id:
            break
        clean_ids.append(int(tid))  # 明確轉成 int

    # 安全解碼（首選 DecodeIds）
    try:
        text = tokenizer.DecodeIds(clean_ids)
    except Exception:
        # 舊版/不同包裝的相容處理
        try:
            text = tokenizer.decode(clean_ids)
        except Exception:
            text = tokenizer.Decode(clean_ids)  # 最後退路
    return text

def logits_to_text(
    logits: torch.Tensor,
    tokenizer,
    pad_id: int,
    bos_id: int | None = None,
    eos_id: int | None = None,
) -> List[str]:
    """
    將模型對 x_0 的 logits 轉成文字序列（先 argmax 取得 token id）。
    參數：
        logits: [B, L, V]，模型對 x_0 的類別 logits
    回傳：
        長度為 B 的 list[str]
    """
    # 取出預測 token id： [B, L]
    pred_ids: torch.Tensor = torch.argmax(logits, dim=-1)

    texts: List[str] = []
    for i in range(pred_ids.size(0)):
        seq_ids = pred_ids[i]  # [L]
        text = _ids_to_text_one(seq_ids, tokenizer, pad_id, bos_id, eos_id)
        texts.append(text)
    return texts

def calculate_cer(ref: str, hyp: str) -> float:
    """
    簡易 CER（Levenshtein on characters）。
    你可以換成更完整的實作，但介面保持不變即可被 train.py 使用。
    """
    import numpy as np

    r = list(ref)
    h = list(hyp)
    dp = np.zeros((len(r)+1, len(h)+1), dtype=np.int32)
    for i in range(len(r)+1):
        dp[i, 0] = i
    for j in range(len(h)+1):
        dp[0, j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i, j] = min(
                dp[i-1, j] + 1,      # deletion
                dp[i, j-1] + 1,      # insertion
                dp[i-1, j-1] + cost  # substitution
            )
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return float(dp[len(r), len(h)]) / float(len(r))

@torch.no_grad()
def evaluate_train_cer(
    encoder,
    decoder,
    s_proj,
    t_embed,
    t_proj,
    scheduler,
    train_loader,
    device: torch.device,
    cfg,
    tokenizer,
    max_batches: int = 5,
) -> float:
    """
    在部分訓練 batch 上快速估算 CER，做為訓練監控用。
    注意：這裡使用的是「直接 argmax 還原」的簡化推論，而非完整 diffusion 取樣流程。
    """
    encoder.eval()
    decoder.eval()
    s_proj.eval()
    t_embed.eval()
    t_proj.eval()

    pad_id = cfg.data['pad_id']
    bos_id = cfg.data.get('bos_id')
    eos_id = cfg.data.get('eos_id')

    cer_list: List[float] = []
    n_done = 0

    for batch in train_loader:
        wave, x0 = batch
        wave = wave.to(device)
        x0 = x0.to(device)  # [B, L] 真實目標 ids

        # 取得條件 c
        c, c_mask, _ = encoder(wave)  # [B, S, d]

        # 這裡用 “t=1 的簡化版”：把 x_t 當作 x_0 直接餵（或自行選固定 t）
        # 你也可以呼叫完整的 diffusion 反推流程做更嚴謹的評估（較慢）
        t = torch.ones(x0.size(0), dtype=torch.long, device=device)
        xt = x0.clone()

        logits = decoder(xt, t, c, x_mask=(x0 != pad_id), c_mask=c_mask)  # [B, L, V]

        # 還原文字
        hyps = logits_to_text(logits, tokenizer, pad_id, bos_id, eos_id)

        # 準備參考文字：把 x0 也安全解碼
        refs = []
        for i in range(x0.size(0)):
            refs.append(
                _ids_to_text_one(x0[i], tokenizer, pad_id, bos_id, eos_id)
            )

        # 計算每個樣本的 CER
        for r, h in zip(refs, hyps):
            cer_list.append(calculate_cer(r, h))

        n_done += 1
        if n_done >= max_batches:
            break

    if len(cer_list) == 0:
        return 1.0
    return sum(cer_list) / len(cer_list)

@torch.no_grad()
def evaluate_cer(
    encoder,
    decoder,
    s_proj,
    t_embed,
    t_proj,
    scheduler,
    data_loader,
    device: torch.device,
    cfg,
    tokenizer,
) -> float:
    """
    驗證/測試集 CER（同上，為了速度用簡化 argmax 推論）。
    若你已完成「Cross-Modality Diffusion 的完整取樣流程」，可替換為真正的解碼器。
    """
    encoder.eval()
    decoder.eval()
    s_proj.eval()
    t_embed.eval()
    t_proj.eval()

    pad_id = cfg.data['pad_id']
    bos_id = cfg.data.get('bos_id')
    eos_id = cfg.data.get('eos_id')

    all_cer: List[float] = []

    for batch in data_loader:
        wave, x0 = batch
        wave = wave.to(device)
        x0 = x0.to(device)

        c, c_mask, _ = encoder(wave)
        t = torch.ones(x0.size(0), dtype=torch.long, device=device)
        xt = x0.clone()

        logits = decoder(xt, t, c, x_mask=(x0 != pad_id), c_mask=c_mask)
        hyps = logits_to_text(logits, tokenizer, pad_id, bos_id, eos_id)
        refs = [
            _ids_to_text_one(x0[i], tokenizer, pad_id, bos_id, eos_id)
            for i in range(x0.size(0))
        ]
        for r, h in zip(refs, hyps):
            all_cer.append(calculate_cer(r, h))

    if len(all_cer) == 0:
        return 1.0
    return sum(all_cer) / len(all_cer)
