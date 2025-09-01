# models/evaluate.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple
import torch

# 進度條（若沒安裝也能安全退回）
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

def _iter_with_progress(iterable, desc: str, total: int | None = None):
    """
    將可疊代物件包上 tqdm；若環境沒有 tqdm，就印出描述字串後直接回傳原 iterable。
    """
    if tqdm is not None:
        try:
            return tqdm(iterable, desc=desc, total=total, leave=False)
        except Exception:
            return tqdm(iterable, desc=desc, leave=False)
    else:
        print(desc, flush=True)
        return iterable

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

def calculate_wer(ref: str, hyp: str) -> float:
    """
    基本 WER：對 ref/hyp 做空白切詞，計算詞級 Levenshtein / 參考詞數。
    """
    import numpy as np
    r = ref.strip().split()
    h = hyp.strip().split()
    dp = np.zeros((len(r)+1, len(h)+1), dtype=np.int32)
    for i in range(len(r)+1): dp[i, 0] = i
    for j in range(len(h)+1): dp[0, j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            cost = 0 if r[i-1] == h[j-1] else 1
            dp[i, j] = min(dp[i-1, j] + 1, dp[i, j-1] + 1, dp[i-1, j-1] + cost)
    return 0.0 if len(r) == 0 else float(dp[len(r), len(h)]) / float(len(r))

@torch.no_grad()
def evaluate_wer_with_jumpy_sampling(
    encoder, decoder, scheduler, data_loader, device, cfg, tokenizer
) -> float:
    # 載入與 evaluate_cer_with_jumpy_sampling 相同的 sampling_config，
    # 但一定要保證 posterior_mode='map'
    sampling_config = {
        'T_infer': cfg.get('inference', {}).get('T_infer', 20),
        'r': cfg.get('inference', {}).get('r', 5),
        'greedy': True,
        'posterior_mode': 'map',
        'sampling_mode': cfg.get('inference', {}).get('sampling_mode', 'exact'),
        'temperature': cfg.get('inference', {}).get('temperature', 1.0),
    }
    from sampler.jumpy_sampler import DiffusionJumpySampler

    encoder.eval(); decoder.eval()
    pad_id = cfg.data['pad_id']; vocab_size = cfg.data['vocab_size']
    T_train = cfg.diffusion['T']

    total_wer, total_samples = 0.0, 0
    for batch in data_loader:
        wave, x0 = batch
        wave = wave.to(device)
        B, L = x0.shape
        c, c_mask, _ = encoder(wave)

        hyps = []
        for b in range(B):
            sampler = DiffusionJumpySampler(
                scheduler=scheduler.sch if hasattr(scheduler, 'sch') else scheduler,
                decoder=decoder, K=vocab_size, T_train=T_train,
                T_infer=sampling_config['T_infer'], r=sampling_config['r'],
                greedy=True, posterior_mode='map',
                sampling_mode=sampling_config['sampling_mode'],
                temperature=sampling_config['temperature'],
                device=device,
            )
            x_pred, _ = sampler.sample(cond_c=c[b:b+1], seq_len=L)  # [1, L]
            hyp = _ids_to_text_one(
                x_pred[0], tokenizer, pad_id, cfg.data.get('bos_id'), cfg.data.get('eos_id')
            )
            hyps.append(hyp)

        refs = [
            _ids_to_text_one(x0[i], tokenizer, pad_id, cfg.data.get('bos_id'), cfg.data.get('eos_id'))
            for i in range(B)
        ]
        for ref, hyp in zip(refs, hyps):
            total_wer += calculate_wer(ref, hyp)
            total_samples += 1

    return (total_wer / total_samples) if total_samples > 0 else 0.0

@torch.no_grad()
def evaluate_validation_loss(
    encoder,
    decoder,
    s_proj,
    t_embed,
    t_proj,
    scheduler,
    data_loader,
    device: torch.device,
    cfg,
) -> float:
    """
    計算驗證集的平均損失（用於監控訓練過程），與訓練相同的 KL 項，不反傳。
    """
    encoder.eval()
    decoder.eval()
    s_proj.eval()
    t_embed.eval()
    t_proj.eval()

    pad_id = cfg.data['pad_id']
    total_loss = 0.0
    total_tokens = 0  # 以樣本數或 token 數做平均都可；這裡用 batch 累計 * B

    try:
        total = len(data_loader)
    except Exception:
        total = None
    iterator = _iter_with_progress(data_loader, desc="Eval[val loss]", total=total)

    for batch in iterator:
        wave, x0 = batch
        wave = wave.to(device)
        x0 = x0.to(device)
        B, L = x0.shape

        c, c_mask, _ = encoder(wave)
        t = torch.ones(B, dtype=torch.long, device=device)
        xt = x0.clone()

        logits = decoder(xt, t, c, x_mask=(x0 != pad_id), c_mask=c_mask)
        x_mask = (x0 != pad_id)
        loss_diff = scheduler.kl_term(xt, x0, logits, t, x_mask)

        total_loss += float(loss_diff.item()) * B
        total_tokens += B

        try:
            avg = total_loss / max(1, total_tokens)
            if tqdm is not None:
                iterator.set_postfix({"avg_loss": f"{avg:.3f}"})
        except Exception:
            pass

    return (total_loss / total_tokens) if total_tokens > 0 else 0.0

@torch.no_grad()
def evaluate_cer_with_full_sampling(
    encoder,
    decoder,
    scheduler,
    data_loader,
    device: torch.device,
    cfg,
    tokenizer,
    sampling_config: dict | None = None,
) -> float:
    """
    使用完整 diffusion（含 jumpy）採樣的 CER 評估（真實推論場景）。
    對齊論文：預設 greedy=True、posterior_mode='map'（Greedy/MAP）。

    參數：
        sampling_config 允許覆寫：
            - T_infer, r, greedy, posterior_mode('map'|'average'), sampling_mode, temperature
    """
    from sampler.jumpy_sampler import DiffusionJumpySampler

    # 預設採樣配置（預設改為 MAP）
    if sampling_config is None:
        sampling_config = {}
    T_infer = sampling_config.get('T_infer', cfg.get('inference', {}).get('T_infer', 20))
    r = sampling_config.get('r', cfg.get('inference', {}).get('r', 5))
    greedy = sampling_config.get('greedy', cfg.get('inference', {}).get('greedy', True))
    posterior_mode = sampling_config.get(
        'posterior_mode',
        cfg.get('inference', {}).get('posterior_mode', 'map')  # 預設 'map'
    )
    sampling_mode = sampling_config.get('sampling_mode', cfg.get('inference', {}).get('sampling_mode', 'exact'))
    temperature = sampling_config.get('temperature', cfg.get('inference', {}).get('temperature', 1.0))

    encoder.eval()
    decoder.eval()

    pad_id = cfg.data['pad_id']
    vocab_size = cfg.data['vocab_size']
    T_train = cfg.diffusion['T']

    try:
        total = len(data_loader)
    except Exception:
        total = None
    iterator = _iter_with_progress(data_loader, desc="Eval[full sampling CER]", total=total)

    total_cer = 0.0
    total_samples = 0

    # 這個 sampler 可以每個 batch 重用（條件 c 會換）
    sampler = DiffusionJumpySampler(
        scheduler=scheduler.sch if hasattr(scheduler, 'sch') else scheduler,
        decoder=decoder,
        K=vocab_size,
        T_train=T_train,
        T_infer=T_infer,
        r=r,
        greedy=greedy,
        posterior_mode=posterior_mode,   # 預設 'map'
        sampling_mode=sampling_mode,
        temperature=temperature,
        device=device,
    )

    for batch in iterator:
        wave, x0 = batch
        wave = wave.to(device)
        x0 = x0.to(device)
        B, L = x0.shape

        # 取得條件
        c, c_mask, _ = encoder(wave)

        # 採樣（回傳 token ids）
        x_pred, _ = sampler.sample(cond_c=c, seq_len=L)   # [B, L], long

        # 直接用 ids 安全解碼
        hyps = [_ids_to_text_one(x_pred[i], tokenizer, pad_id, cfg.data.get('bos_id'), cfg.data.get('eos_id'))
                for i in range(B)]
        refs = [_ids_to_text_one(x0[i],    tokenizer, pad_id, cfg.data.get('bos_id'), cfg.data.get('eos_id'))
                for i in range(B)]

        for ref, hyp in zip(refs, hyps):
            total_cer += calculate_cer(ref, hyp)
            total_samples += 1

        try:
            avg = total_cer / max(1, total_samples)
            if tqdm is not None:
                iterator.set_postfix({"avg_CER": f"{avg:.3f}"})
        except Exception:
            pass

    return (total_cer / total_samples) if total_samples > 0 else 0.0


@torch.no_grad()
def evaluate_cer_with_multi_sample(
    encoder,
    decoder,
    scheduler,
    data_loader,
    device: torch.device,
    cfg,
    tokenizer,
    sampling_config: dict = None,
    num_samples: int = 3,
) -> float:
    """
    使用多樣本平均的 CER 評估（對同一輸入進行多次採樣取平均）

    參數：
        encoder: 聲學編碼器
        decoder: 去噪解碼器
        scheduler: 擴散排程器
        data_loader: 資料載入器
        device: 計算裝置
        cfg: 配置物件
        tokenizer: 文字編碼器
        sampling_config: 採樣配置參數
        num_samples: 每次輸入的採樣次數

    回傳：
        平均 CER 值
    """
    # 匯入 jumpy sampler
    from sampler.jumpy_sampler import DiffusionJumpySampler

    # 預設採樣配置
    if sampling_config is None:
        sampling_config = {
            'T_infer': cfg.get('inference', {}).get('T_infer', 20),
            'r': cfg.get('inference', {}).get('r', 2),
            'greedy': cfg.get('inference', {}).get('greedy', True),
            'posterior_mode': cfg.get('inference', {}).get('posterior_mode', 'average'),
            'sampling_mode': cfg.get('inference', {}).get('sampling_mode', 'exact'),
            'temperature': cfg.get('inference', {}).get('temperature', 1.0),
        }

    encoder.eval()
    decoder.eval()

    pad_id = cfg.data['pad_id']
    vocab_size = cfg.data['vocab_size']
    T_train = cfg.diffusion['T']
    total_cer = 0.0
    total_samples = 0

    for batch in data_loader:
        wave, x0 = batch
        wave = wave.to(device)
        B, L = x0.shape

        # 取得聲學條件
        c, c_mask, _ = encoder(wave)

        # 對每個樣本進行多次採樣
        batch_predictions = []
        for b in range(B):
            sample_predictions = []
            c_single = c[b:b+1]  # [1, S, D]

            for _ in range(num_samples):
                # 建立針對單一樣本的 sampler
                sampler = DiffusionJumpySampler(
                    scheduler=scheduler.sch if hasattr(scheduler, 'sch') else scheduler,
                    decoder=decoder,
                    K=vocab_size,
                    T_train=T_train,
                    T_infer=sampling_config['T_infer'],
                    r=sampling_config['r'],
                    greedy=False,  # 多樣本需要隨機性
                    posterior_mode=sampling_config['posterior_mode'],
                    sampling_mode=sampling_config['sampling_mode'],
                    temperature=sampling_config['temperature'],
                    device=device,
                )

                # 進行採樣
                x_pred, _ = sampler.sample(cond_c=c_single, seq_len=L)  # [1, L]

                # 轉換為文字
                pred_text = _ids_to_text_one(
                    x_pred[0], tokenizer, pad_id,
                    cfg.data.get('bos_id'), cfg.data.get('eos_id')
                )
                sample_predictions.append(pred_text)

            # 對多次採樣結果進行投票或平均
            # 這裡使用簡單的多數決策略
            batch_predictions.append(sample_predictions[0])  # 臨時使用第一個樣本

        # 還原真實文字
        refs = [
            _ids_to_text_one(x0[i], tokenizer, pad_id, cfg.data.get('bos_id'), cfg.data.get('eos_id'))
            for i in range(B)
        ]

        # 計算 CER
        for ref, hyp in zip(refs, batch_predictions):
            cer = calculate_cer(ref, hyp)
            total_cer += cer
            total_samples += 1

    return (total_cer / total_samples) if total_samples > 0 else 0.0

@torch.no_grad()
def evaluate_cer_with_jumpy_sampling(
    encoder,
    decoder,
    scheduler,
    data_loader,
    device: torch.device,
    cfg,
    tokenizer,
) -> float:
    """
    使用 Jumpy Sampler 的 CER 評估（對齊論文：預設 greedy + MAP）。
    會呼叫上面的 full_sampling 評估函式，並沿用 cfg.inference 的其他參數。
    """
    sampling_config = {
        'T_infer':      cfg.get('inference', {}).get('T_infer', 20),
        'r':            cfg.get('inference', {}).get('r', 5),
        'greedy':       cfg.get('inference', {}).get('greedy', True),
        'posterior_mode': cfg.get('inference', {}).get('posterior_mode', 'map'),  # 預設 'map'
        'sampling_mode':  cfg.get('inference', {}).get('sampling_mode', 'exact'),
        'temperature':    cfg.get('inference', {}).get('temperature', 1.0),
    }
    return evaluate_cer_with_full_sampling(
        encoder, decoder, scheduler, data_loader, device, cfg, tokenizer, sampling_config
    )