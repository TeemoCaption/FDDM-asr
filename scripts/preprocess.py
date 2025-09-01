# -*- coding: utf-8 -*-
"""
preprocess.py - 自動合併多版本數據集前處理腳本

功能特色
--------
✅ 自動發現 data/raw/ 下的所有數據集版本
✅ 智能去重：基於文本相似度和音檔 hash
✅ 統一輸出：生成標準的 train.json, dev.json, test.json
✅ 批次處理：記憶體優化，支援大型數據集
✅ 版本管理：追蹤數據來源，生成詳細報告
✅ 向後兼容：保留原有單一數據集處理功能

數據結構
--------
請將 Common Voice 官方釋出的壓縮檔解壓到專案的 data/raw 目錄下：
    data/raw/
    ├── cv-corpus-22.0-2025-06-20/
    │   └── zh-TW/
    │       ├── clips/
    │       ├── train.tsv
    │       ├── dev.tsv
    │       └── test.tsv
    ├── cv-corpus-23.0-2025-12-20/
    │   └── zh-TW/
    │       ├── clips/
    │       ├── train.tsv
    │       ├── dev.tsv
    │       └── test.tsv
    └── ...

輸出結構
--------
data/processed/
├── clips/                    # 轉檔後的 16kHz WAV
├── train.json               # 訓練集索引（合併後）
├── dev.json                 # 驗證集索引（合併後）
├── test.json                # 測試集索引（合併後）
├── merge_report.json        # 合併報告
└── duplicates_removed.json  # 去重記錄

使用範例
--------
# 自動處理所有版本（推薦）
python preprocess.py --auto_merge

# 指定特定版本
python preprocess.py --dataset_names "cv-corpus-22.0,cv-corpus-23.0" --auto_merge

# 調整去重參數
python preprocess.py --auto_merge --text_similarity_threshold 0.9 --enable_audio_hash

# 傳統單一數據集模式（向後兼容）
python preprocess.py --dataset_name "cv-corpus-22.0-2025-06-20" --splits "train,dev,test"

重要特性
--------
- 自動去重：避免不同版本間的重複音檔
- 文本相似度檢查：可調整閾值（預設 0.95）
- 音檔 hash 檢查：可選啟用，更精確但計算量大
- 批次處理：記憶體友善，支援大型數據集
- 詳細報告：追蹤處理過程和統計資訊
"""

import os
import json
import argparse
import re
import hashlib
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from difflib import SequenceMatcher

import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm


# -----------------------------
# 專案內部目錄（與你的 FDDM-asr 結構相容）
# -----------------------------
HERE = os.path.abspath(os.path.dirname(__file__))              # 當前檔案所在資料夾（scripts/）
ROOT_DIR = os.path.dirname(HERE)                              # 專案根目錄（上一層）
DATA_DIR = os.path.join(ROOT_DIR, "data")                     # data/
RAW_DIR = os.path.join(DATA_DIR, "raw")                       # data/raw/
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")           # data/processed/
PROCESSED_CLIPS = os.path.join(PROCESSED_DIR, "clips")        # data/processed/clips/
# 確保目錄存在
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_CLIPS, exist_ok=True)


# =========================================================
# 文本正規化
# =========================================================
def normalize_text(text: str) -> str:
    """
    文本正規化策略（可依需求調整）：
      - 去除括號內的羅馬字（例如：我愛你(guá ài lí) -> 我愛你）
      - 全部轉成小寫
      - 移除多餘空白
      - 僅保留中文、英數與空白（去掉奇怪符號）
    參考關鍵字：ASR text normalization、繁體中文正規化、台語羅馬字清理
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\([^)]*\)", "", text)             # 去掉括號與內容
    text = text.lower()                                # 轉小寫
    text = re.sub(r"\s+", " ", text).strip()           # 多空白壓一個
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", "", text)   # 只留中英數與空白
    return text


# =========================================================
# 音訊處理
# =========================================================
def to_wav_16k_mono(src_path: str, dst_path: str, target_sr: int = 16000) -> float:
    """
    將來源音檔（多半是 mp3）轉成 16kHz 單聲道 WAV。
    回傳：音檔秒數（float）。若失敗回傳 -1。

    參數
    ----
    src_path : 來源檔案絕對路徑（Common Voice 的 clips/*.mp3）
    dst_path : 轉檔後輸出絕對路徑（data/processed/clips/*.wav）
    target_sr: 目標取樣率（預設 16000）
    """
    try:
        # librosa 讀檔；sr=None 代表保持原始取樣率、mono=True 代表混縮成單聲道
        y, sr = librosa.load(src_path, sr=None, mono=True)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        sf.write(dst_path, y, target_sr, subtype="PCM_16")
        duration = float(len(y)) / float(target_sr)
        return duration
    except Exception as e:
        print(f"[轉檔失敗] {src_path} -> {dst_path} | {e}")
        return -1.0


# =========================================================
# 讀取本地 TSV 並產生索引
# =========================================================
def read_split_tsv(cv_lang_dir: str, split_name: str) -> pd.DataFrame:
    """
    從 <cv_root>/<language>/ 讀取指定分割的 .tsv 檔

    會嘗試對應幾個常見命名：
      - train.tsv / dev.tsv / test.tsv / validated.tsv
      - validation.tsv / other.tsv（若你要自訂也可加）

    回傳：包含至少 ['path','sentence'] 欄位的 DataFrame
    """
    # 能對應的檔名映射
    candidate_files = {
        "train": ["train.tsv"],
        "dev": ["dev.tsv", "validation.tsv", "validated.tsv"],  # dev 也有人叫 validation
        "test": ["test.tsv"],
        "validated": ["validated.tsv"],                         # 額外允許
        "other": ["other.tsv"],
    }

    filenames = candidate_files.get(split_name, [f"{split_name}.tsv"])
    tsv_path = None
    for fn in filenames:
        fp = os.path.join(cv_lang_dir, fn)
        if os.path.isfile(fp):
            tsv_path = fp
            break
    if tsv_path is None:
        raise FileNotFoundError(f"找不到 {split_name} 的 .tsv，已嘗試：{filenames}")

    # 讀檔（Common Voice 為 tab 分隔）
    df = pd.read_csv(tsv_path, sep="\t", quoting=3, dtype=str, keep_default_na=False)
    # 欄位容錯：新版固定是 'path' 和 'sentence'
    if "path" not in df.columns:
        # 有少數版本用 'filename'
        if "filename" in df.columns:
            df = df.rename(columns={"filename": "path"})
        else:
            raise KeyError(f"{tsv_path} 缺少 'path' 欄位。實際欄位：{list(df.columns)}")

    if "sentence" not in df.columns:
        # 有的版本可能用 'text'
        if "text" in df.columns:
            df = df.rename(columns={"text": "sentence"})
        else:
            raise KeyError(f"{tsv_path} 缺少 'sentence' 欄位。實際欄位：{list(df.columns)}")

    return df[["path", "sentence"]]


def detect_datasets(dataset_names: Optional[List[str]] = None) -> List[str]:
    """
    自動發現 data/raw/ 下的所有數據集版本
    
    參數
    ----
    dataset_names : 指定的數據集名稱列表，None 表示自動發現所有
    
    回傳
    ----
    數據集名稱列表
    """
    if dataset_names:
        # 使用指定的數據集名稱
        datasets = []
        for name in dataset_names:
            dataset_path = os.path.join(RAW_DIR, name)
            if os.path.isdir(dataset_path):
                datasets.append(name)
            else:
                print(f"警告：指定的數據集不存在：{dataset_path}")
        return datasets
    
    # 自動發現所有數據集
    datasets = []
    if not os.path.isdir(RAW_DIR):
        print(f"警告：raw 目錄不存在：{RAW_DIR}")
        return datasets
    
    for item in os.listdir(RAW_DIR):
        item_path = os.path.join(RAW_DIR, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            # 檢查是否包含語言子目錄
            has_language_dirs = False
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path):
                    clips_path = os.path.join(subitem_path, "clips")
                    if os.path.isdir(clips_path):
                        has_language_dirs = True
                        break
            
            if has_language_dirs:
                datasets.append(item)
    
    return sorted(datasets)  # 按名稱排序確保一致性


def detect_languages(cv_root: str) -> List[str]:
    """
    自動偵測資料集中的語言資料夾
    
    參數
    ----
    cv_root : Common Voice 資料集根目錄
    
    回傳
    ----
    語言代碼清單，例如 ['zh-TW', 'en', 'ja']
    """
    if not os.path.isdir(cv_root):
        return []
    
    languages = []
    # 掃描所有子目錄
    for item in os.listdir(cv_root):
        item_path = os.path.join(cv_root, item)
        # 檢查是否為目錄且包含 clips 子目錄
        if os.path.isdir(item_path):
            clips_path = os.path.join(item_path, "clips")
            if os.path.isdir(clips_path):
                languages.append(item)
    
    return sorted(languages)  # 排序以確保一致性


def calculate_audio_hash(file_path: str) -> Optional[str]:
    """
    計算音檔的 MD5 hash 值用於去重
    
    參數
    ----
    file_path : 音檔路徑
    
    回傳
    ----
    MD5 hash 字串，失敗時回傳 None
    """
    try:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"計算音檔 hash 失敗：{file_path} | {e}")
        return None


def text_similarity(text1: str, text2: str) -> float:
    """
    計算兩個文本的相似度
    
    參數
    ----
    text1, text2 : 要比較的文本
    
    回傳
    ----
    相似度分數 (0.0 到 1.0)
    """
    return SequenceMatcher(None, text1, text2).ratio()


def is_duplicate_record(record: dict, 
                       seen_texts: Set[str], 
                       text_to_records: Dict[str, List[dict]], 
                       seen_audio_hashes: Set[str],
                       text_similarity_threshold: float = 0.95,
                       enable_audio_hash: bool = False) -> Tuple[bool, str]:
    """
    檢查記錄是否為重複
    
    參數
    ----
    record : 要檢查的記錄
    seen_texts : 已見過的文本集合
    text_to_records : 文本到記錄的映射
    seen_audio_hashes : 已見過的音檔 hash 集合
    text_similarity_threshold : 文本相似度閾值
    enable_audio_hash : 是否啟用音檔 hash 檢查
    
    回傳
    ----
    (是否重複, 重複原因)
    """
    text = record.get('text', '').strip()
    
    # 1. 完全相同的文本
    if text in seen_texts:
        return True, "完全相同的文本"
    
    # 2. 高相似度文本檢查
    for existing_text in seen_texts:
        if text_similarity(text, existing_text) >= text_similarity_threshold:
            return True, f"高相似度文本 (相似度 >= {text_similarity_threshold})"
    
    # 3. 音檔 hash 檢查（如果啟用）
    if enable_audio_hash and 'audio_hash' in record:
        audio_hash = record['audio_hash']
        if audio_hash and audio_hash in seen_audio_hashes:
            return True, "相同的音檔 hash"
    
    return False, ""


def process_language_batch(dataset_name: str,
                           language: str,
                           splits: List[str],
                           batch_size: int = 1000,
                           seen_texts: Optional[Set[str]] = None,
                           seen_audio_hashes: Optional[Set[str]] = None,
                           text_similarity_threshold: float = 0.95,
                           enable_audio_hash: bool = False) -> Tuple[float, Dict]:
    """
    批次處理單一語言的資料集，避免一次性載入所有資料到記憶體，支援去重功能
    
    參數
    ----
    dataset_name : 數據集名稱
    language : 語言代碼
    splits : 要處理的分割列表
    batch_size : 每個批次處理的檔案數量，預設 1000
    seen_texts : 已見過的文本集合（用於去重）
    seen_audio_hashes : 已見過的音檔 hash 集合（用於去重）
    text_similarity_threshold : 文本相似度閾值
    enable_audio_hash : 是否啟用音檔 hash 檢查
    
    回傳：(該語言所有有效音檔的總時長（秒數）, 統計資訊字典)
    """
    # 初始化去重集合（如果未提供）
    if seen_texts is None:
        seen_texts = set()
    if seen_audio_hashes is None:
        seen_audio_hashes = set()
    
    # 初始化該語言的總時長變數
    total_duration_lang = 0.0
    
    # 統計資訊
    stats = {
        'total_found': 0,           # 總發現檔案數
        'total_processed': 0,       # 總處理檔案數
        'duplicates_removed': 0,    # 去重移除數
        'invalid_files': 0,         # 無效檔案數
        'duplicates_detail': []     # 重複記錄詳情
    }
    
    # 固定從 data/raw 讀取資料集
    cv_root = os.path.join(RAW_DIR, dataset_name)              # e.g. data/raw/cv-corpus-22.0-2025-06-20
    cv_lang_dir = os.path.join(cv_root, language)             # e.g. data/raw/cv-corpus-22.0.../zh-TW
    clips_dir = os.path.join(cv_lang_dir, "clips")            # e.g. .../zh-TW/clips
    
    print(f"\n批次處理語言：{language}（批次大小：{batch_size}）")
    print(f"語言目錄：{cv_lang_dir}")

    # 逐個 split 處理
    for split in splits:
        print(f"=== 批次處理 {language} - {split} ===")
        # 初始化該 split 的總時長變數
        total_duration_split = 0.0
        # 初始化該 split 的有效檔案計數
        total_valid_files = 0
        
        try:
            df = read_split_tsv(cv_lang_dir, split)               # 讀 TSV
        except FileNotFoundError:
            print(f"跳過 {language} - {split}：找不到對應的 TSV 檔案")
            continue

        # 新增欄位：正規化文本和語言標記
        df["text"] = df["sentence"].apply(normalize_text)
        df["language"] = language
        
        # 統計總發現檔案數
        stats['total_found'] += len(df)

        # 準備輸出檔案路徑（暫時使用語言_分割格式，稍後會合併）
        jsonl_path = os.path.join(PROCESSED_DIR, f"{language}_{split}_temp.json")
        csv_path = os.path.join(PROCESSED_DIR, f"{language}_{split}_temp.csv")
        
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        
        # 初始化批次資料
        batch_records = []
        batch_start_idx = 0
        
        # 分批處理 DataFrame，使用 tqdm 顯示批次處理進度
        for batch_start_idx in tqdm(range(0, len(df), batch_size), desc=f"處理 {split} 批次"):
            batch_end_idx = min(batch_start_idx + batch_size, len(df))
            batch_df = df.iloc[batch_start_idx:batch_end_idx]
            
            print(f"處理批次 {batch_start_idx}-{batch_end_idx}（總共 {len(df)}）")
            
            # 處理當前批次的每個檔案，使用 tqdm 顯示檔案處理進度
            for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc="處理檔案"):
                rel_path = row["path"]                            # 如 'common_voice_zh-TW_12345.mp3'
                src_fp = os.path.join(clips_dir, rel_path)        # 絕對路徑

                # 為避免不同語言和 split 同名檔案互相覆蓋，輸出檔名前加上語言和 split 前綴
                base = os.path.splitext(os.path.basename(rel_path))[0]
                out_name = f"{language}_{split}_{base}.wav"
                out_rel = os.path.join("data", "processed", "clips", out_name)     # 存索引時使用相對路徑
                out_abs = os.path.join(ROOT_DIR, out_rel)                           # 真正寫檔用絕對路徑

                if not os.path.isfile(src_fp):
                    # 少數 .tsv 可能引用不存在的檔案，略過並提示
                    stats['invalid_files'] += 1
                    continue

                # 計算音檔 hash（如果啟用）
                audio_hash = None
                if enable_audio_hash:
                    audio_hash = calculate_audio_hash(src_fp)

                # 創建臨時記錄用於去重檢查
                temp_record = {
                    'text': row["text"],
                    'sentence': row["sentence"],
                    'audio_hash': audio_hash,
                    'dataset': dataset_name,
                    'language': language,
                    'split': split,
                    'path': rel_path
                }

                # 檢查是否為重複記錄
                is_duplicate, duplicate_reason = is_duplicate_record(
                    temp_record, seen_texts, {}, seen_audio_hashes,
                    text_similarity_threshold, enable_audio_hash
                )

                if is_duplicate:
                    stats['duplicates_removed'] += 1
                    stats['duplicates_detail'].append({
                        'path': rel_path,
                        'text': row["text"],
                        'reason': duplicate_reason,
                        'dataset': dataset_name,
                        'language': language,
                        'split': split
                    })
                    continue

                duration = to_wav_16k_mono(src_fp, out_abs, target_sr=16000)

                # 失敗就跳過
                if duration <= 0:
                    stats['invalid_files'] += 1
                    continue

                # 篩時長（使用固定的合理範圍）
                min_dur, max_dur = 0.1, 30.0  # 0.1秒到30秒的合理範圍
                if duration < min_dur or duration > max_dur:
                    stats['invalid_files'] += 1
                    continue

                # 記錄為已處理（加入去重集合）
                seen_texts.add(row["text"])
                if audio_hash:
                    seen_audio_hashes.add(audio_hash)
                stats['total_processed'] += 1

                # 累加該 split 的總時長
                total_duration_split += duration
                # 累加有效檔案數量
                total_valid_files += 1
                
                # 加入批次記錄
                batch_records.append({
                    "path": rel_path,
                    "sentence": row["sentence"],
                    "text": row["text"],
                    "duration": round(float(duration), 3),
                    "processed_path": out_rel.replace("\\", "/"),
                    "language": language,
                    "dataset": dataset_name,
                    "split": split,
                    "audio_hash": audio_hash if enable_audio_hash else None,
                })
                
                # 如果批次記錄達到一定大小，就寫入檔案並清空
                if len(batch_records) >= 500:
                    # 串流寫入 JSON（追加模式）
                    if not os.path.exists(jsonl_path):
                        # 第一次寫入，建立檔案
                        with open(jsonl_path, "w", encoding="utf-8") as f:
                            json.dump(batch_records, f, ensure_ascii=False, indent=2)
                    else:
                        # 追加寫入
                        with open(jsonl_path, "r+", encoding="utf-8") as f:
                            existing_data = json.load(f)
                            existing_data.extend(batch_records)
                            f.seek(0)
                            json.dump(existing_data, f, ensure_ascii=False, indent=2)
                            f.truncate()
                    
                    # 串流寫入 CSV（追加模式）
                    batch_df_out = pd.DataFrame(batch_records)
                    if not os.path.exists(csv_path):
                        batch_df_out.to_csv(csv_path, index=False, encoding="utf-8")
                    else:
                        batch_df_out.to_csv(csv_path, mode='a', header=False, index=False, encoding="utf-8")
                    
                    # 清空批次記錄以釋放記憶體
                    batch_records = []
                    
                    print(f"已寫入批次資料，釋放記憶體（當前批次記錄數：{len(batch_records)}）")
            
            # 批次處理完成後，立即寫入剩餘記錄
            if batch_records:
                # 串流寫入 JSON
                if not os.path.exists(jsonl_path):
                    with open(jsonl_path, "w", encoding="utf-8") as f:
                        json.dump(batch_records, f, ensure_ascii=False, indent=2)
                else:
                    with open(jsonl_path, "r+", encoding="utf-8") as f:
                        existing_data = json.load(f)
                        existing_data.extend(batch_records)
                        f.seek(0)
                        json.dump(existing_data, f, ensure_ascii=False, indent=2)
                        f.truncate()
                
                # 串流寫入 CSV
                batch_df_out = pd.DataFrame(batch_records)
                if not os.path.exists(csv_path):
                    batch_df_out.to_csv(csv_path, index=False, encoding="utf-8")
                else:
                    batch_df_out.to_csv(csv_path, mode='a', header=False, index=False, encoding="utf-8")
                
                print(f"最終批次寫入完成，釋放記憶體")
        
        # 累加該語言的總時長
        total_duration_lang += total_duration_split
        
        if total_valid_files == 0:
            print(f"警告：{language} - {split} 沒有有效的音檔記錄")
            continue

        # 印出該 split 的總時長
        hours_split = total_duration_split / 3600.0
        print(f"完成 {language} - {split}: 音檔 {total_valid_files} 筆 | 索引：{jsonl_path} / {csv_path} | 總時長：{hours_split:.2f} 小時")
    
    # 印出該語言的總時長和統計資訊
    hours_lang = total_duration_lang / 3600.0
    print(f"完成語言 {language}: 總時長 {hours_lang:.2f} 小時")
    print(f"  - 總發現檔案: {stats['total_found']}")
    print(f"  - 成功處理: {stats['total_processed']}")
    print(f"  - 去重移除: {stats['duplicates_removed']}")
    print(f"  - 無效檔案: {stats['invalid_files']}")
    
    return total_duration_lang, stats

def build_manifests_auto_merge(dataset_names: Optional[List[str]] = None,
                               language: Optional[str] = None,
                               splits: List[str] = ["train", "dev", "test"],
                               batch_size: int = 1000,
                               text_similarity_threshold: float = 0.95,
                               enable_audio_hash: bool = False) -> Dict:
    """
    自動合併多版本數據集的主要函數
    
    參數
    ----
    dataset_names : 指定的數據集名稱列表，None 表示自動發現所有
    language : 指定語言，None 表示自動偵測所有語言
    splits : 要處理的分割列表
    batch_size : 批次處理大小
    text_similarity_threshold : 文本相似度閾值
    enable_audio_hash : 是否啟用音檔 hash 檢查
    
    回傳：合併統計資訊字典
    """
    # 初始化全域去重集合
    global_seen_texts = set()
    global_seen_audio_hashes = set()
    
    # 初始化統計資訊
    merge_stats = {
        'total_duration': 0.0,
        'total_datasets': 0,
        'total_languages': 0,
        'datasets_processed': [],
        'languages_processed': set(),
        'split_stats': {split: {'records': [], 'total_duration': 0.0} for split in splits},
        'global_stats': {
            'total_found': 0,
            'total_processed': 0,
            'duplicates_removed': 0,
            'invalid_files': 0,
            'duplicates_detail': []
        }
    }
    
    # 自動發現數據集
    datasets = detect_datasets(dataset_names)
    if not datasets:
        raise ValueError("找不到任何數據集，請檢查 data/raw/ 目錄")
    
    print(f"發現數據集：{datasets}")
    merge_stats['total_datasets'] = len(datasets)
    
    # 處理每個數據集
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"處理數據集：{dataset_name}")
        print(f"{'='*60}")
        
        cv_root = os.path.join(RAW_DIR, dataset_name)
        
        # 檢查數據集是否存在
        if not os.path.isdir(cv_root):
            print(f"跳過不存在的數據集：{cv_root}")
            continue
        
        # 自動偵測語言或使用指定語言
        if language:
            languages = [language]
            print(f"使用指定語言：{language}")
        else:
            languages = detect_languages(cv_root)
            if not languages:
                print(f"在 {dataset_name} 中找不到任何語言資料夾")
                continue
            print(f"自動偵測到語言：{languages}")
        
        merge_stats['languages_processed'].update(languages)
        
        # 處理每個語言
        for lang in languages:
            try:
                print(f"\n處理 {dataset_name} - {lang}")
                
                # 批次處理單一語言並獲取其總時長和統計
                duration_lang, lang_stats = process_language_batch(
                    dataset_name, lang, splits, batch_size,
                    global_seen_texts, global_seen_audio_hashes,
                    text_similarity_threshold, enable_audio_hash
                )
                
                # 累加到總時長
                merge_stats['total_duration'] += duration_lang
                
                # 累加統計資訊
                merge_stats['global_stats']['total_found'] += lang_stats['total_found']
                merge_stats['global_stats']['total_processed'] += lang_stats['total_processed']
                merge_stats['global_stats']['duplicates_removed'] += lang_stats['duplicates_removed']
                merge_stats['global_stats']['invalid_files'] += lang_stats['invalid_files']
                merge_stats['global_stats']['duplicates_detail'].extend(lang_stats['duplicates_detail'])
                
            except Exception as e:
                print(f"處理語言 {lang} 時發生錯誤：{e}")
                continue
        
        merge_stats['datasets_processed'].append(dataset_name)
    
    merge_stats['total_languages'] = len(merge_stats['languages_processed'])
    
    # 合併臨時檔案為統一的索引檔案
    print(f"\n{'='*60}")
    print("合併臨時檔案為統一索引...")
    print(f"{'='*60}")
    
    merge_temp_files_to_unified(splits, merge_stats)
    
    # 生成合併報告
    generate_merge_report(merge_stats)
    
    return merge_stats


def build_manifests_batch(dataset_name: str,
                         language: str,
                         splits: List[str],
                         batch_size: int = 1000) -> float:
    """
    批次處理主要函數：自動偵測語言並使用批次處理所有資料（向後兼容）
    
    參數
    ----
    batch_size : 批次處理大小，預設 1000
    
    回傳：整個資料集的有效音檔總時長（秒數）
    """
    # 使用新的合併函數
    merge_stats = build_manifests_auto_merge(
        dataset_names=[dataset_name],
        language=language,
        splits=splits,
        batch_size=batch_size
    )
    
    return merge_stats['total_duration']


def merge_temp_files_to_unified(splits: List[str], merge_stats: Dict):
    """
    合併臨時檔案為統一的索引檔案（train.json, dev.json, test.json）
    
    參數
    ----
    splits : 分割列表
    merge_stats : 合併統計資訊
    """
    for split in splits:
        unified_records = []
        total_duration = 0.0
        
        print(f"合併 {split} 分割...")
        
        # 尋找所有相關的臨時檔案
        temp_pattern = f"*_{split}_temp.json"
        temp_files = []
        
        for file in os.listdir(PROCESSED_DIR):
            if file.endswith(f"_{split}_temp.json"):
                temp_files.append(os.path.join(PROCESSED_DIR, file))
        
        # 讀取並合併所有臨時檔案
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    with open(temp_file, "r", encoding="utf-8") as f:
                        temp_records = json.load(f)
                        if isinstance(temp_records, list):
                            unified_records.extend(temp_records)
                            # 計算該檔案的總時長
                            for record in temp_records:
                                total_duration += record.get('duration', 0.0)
                except Exception as e:
                    print(f"讀取臨時檔案失敗：{temp_file} | {e}")
        
        # 寫入統一的索引檔案
        if unified_records:
            unified_json_path = os.path.join(PROCESSED_DIR, f"{split}.json")
            unified_csv_path = os.path.join(PROCESSED_DIR, f"{split}.csv")
            
            # 寫入 JSON 格式
            with open(unified_json_path, "w", encoding="utf-8") as f:
                json.dump(unified_records, f, ensure_ascii=False, indent=2)
            
            # 寫入 CSV 格式
            df_unified = pd.DataFrame(unified_records)
            df_unified.to_csv(unified_csv_path, index=False, encoding="utf-8")
            
            # 更新統計資訊
            merge_stats['split_stats'][split]['records'] = unified_records
            merge_stats['split_stats'][split]['total_duration'] = total_duration
            
            print(f"  - 合併完成：{len(unified_records)} 筆記錄")
            print(f"  - 總時長：{total_duration/3600.0:.2f} 小時")
            print(f"  - 輸出檔案：{unified_json_path}")
            print(f"  - 輸出檔案：{unified_csv_path}")
        else:
            print(f"  - 警告：{split} 分割沒有找到任何記錄")
        
        # 清理臨時檔案
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                # 同時清理對應的 CSV 臨時檔案
                temp_csv = temp_file.replace('.json', '.csv')
                if os.path.exists(temp_csv):
                    os.remove(temp_csv)
            except Exception as e:
                print(f"清理臨時檔案失敗：{temp_file} | {e}")


def generate_merge_report(merge_stats: Dict):
    """
    生成合併報告
    
    參數
    ----
    merge_stats : 合併統計資訊
    """
    # 生成詳細報告
    report = {
        'merge_summary': {
            'total_datasets': merge_stats['total_datasets'],
            'total_languages': merge_stats['total_languages'],
            'datasets_processed': merge_stats['datasets_processed'],
            'languages_processed': list(merge_stats['languages_processed']),
            'total_duration_hours': merge_stats['total_duration'] / 3600.0,
            'processing_timestamp': pd.Timestamp.now().isoformat()
        },
        'global_statistics': merge_stats['global_stats'],
        'split_statistics': {},
        'duplicates_detail': merge_stats['global_stats']['duplicates_detail']
    }
    
    # 添加各分割的統計
    for split, split_data in merge_stats['split_stats'].items():
        if split_data['records']:
            report['split_statistics'][split] = {
                'total_records': len(split_data['records']),
                'total_duration_hours': split_data['total_duration'] / 3600.0,
                'avg_duration_seconds': split_data['total_duration'] / len(split_data['records']) if split_data['records'] else 0
            }
    
    # 寫入合併報告
    report_path = os.path.join(PROCESSED_DIR, "merge_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # 寫入去重記錄
    if merge_stats['global_stats']['duplicates_detail']:
        duplicates_path = os.path.join(PROCESSED_DIR, "duplicates_removed.json")
        with open(duplicates_path, "w", encoding="utf-8") as f:
            json.dump(merge_stats['global_stats']['duplicates_detail'], f, ensure_ascii=False, indent=2)
    
    # 印出總結
    print(f"\n{'='*60}")
    print("合併完成總結")
    print(f"{'='*60}")
    print(f"處理數據集：{merge_stats['total_datasets']} 個")
    print(f"處理語言：{merge_stats['total_languages']} 個")
    print(f"總錄製時長：{merge_stats['total_duration']/3600.0:.2f} 小時")
    print(f"總發現檔案：{merge_stats['global_stats']['total_found']} 個")
    print(f"成功處理：{merge_stats['global_stats']['total_processed']} 個")
    print(f"去重移除：{merge_stats['global_stats']['duplicates_removed']} 個")
    print(f"無效檔案：{merge_stats['global_stats']['invalid_files']} 個")
    print(f"\n各分割統計：")
    for split, split_data in merge_stats['split_stats'].items():
        if split_data['records']:
            print(f"  - {split}: {len(split_data['records'])} 筆，{split_data['total_duration']/3600.0:.2f} 小時")
    print(f"\n報告檔案：{report_path}")
    if merge_stats['global_stats']['duplicates_detail']:
        print(f"去重記錄：{os.path.join(PROCESSED_DIR, 'duplicates_removed.json')}")


# =========================================================
# 參數與進入點
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="將本地 Common Voice 檔案夾轉為訓練可用的 16kHz WAV 與索引檔（支援自動合併多版本數據集）"
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        default="",
        help="指定數據集名稱列表，以逗號分隔。例如：cv-corpus-22.0,cv-corpus-23.0（留空則自動發現所有數據集）"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="",
        help="單一資料集名稱（向後兼容參數，建議使用 --dataset_names）"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="",
        help="語言子資料夾名稱（留空則自動偵測所有語言）"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,dev,test",
        help="要處理的分割清單，以逗號分隔。例如：train,dev,test 或 validated（自動使用0.1s-30s時長過濾）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="批次處理大小，減少記憶體使用量。預設 1000（較小的值使用更少記憶體但處理較慢）"
    )
    parser.add_argument(
        "--text_similarity_threshold",
        type=float,
        default=0.95,
        help="文本相似度閾值，超過此值視為重複。範圍 0.0-1.0，預設 0.95"
    )
    parser.add_argument(
        "--enable_audio_hash",
        action="store_true",
        help="啟用音檔 hash 檢查去重（計算量大但更精確）"
    )
    parser.add_argument(
        "--auto_merge",
        action="store_true",
        help="啟用自動合併模式（自動發現並合併多版本數據集）"
    )
    parser.add_argument(
        "--use_memory_optimized",
        action="store_true",
        help="啟用記憶體優化模式（批次處理 + 串流輸出）"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 解析 splits
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    
    # 決定使用哪種模式
    if args.auto_merge or (not args.dataset_name and not args.dataset_names):
        # 自動合併模式
        print("="*60)
        print("自動合併多版本數據集模式")
        print("="*60)
        
        # 解析數據集名稱
        dataset_names = None
        if args.dataset_names:
            dataset_names = [name.strip() for name in args.dataset_names.split(",") if name.strip()]
        elif args.dataset_name:
            dataset_names = [args.dataset_name]
        
        print(f"分割：{splits}")
        print(f"時長過濾：0.1s - 30.0s（固定範圍）")
        print(f"文本相似度閾值：{args.text_similarity_threshold}")
        print(f"音檔 hash 檢查：{'啟用' if args.enable_audio_hash else '停用'}")
        print(f"批次大小：{args.batch_size}")
        
        # 執行自動合併
        merge_stats = build_manifests_auto_merge(
            dataset_names=dataset_names,
            language=args.language if args.language else None,
            splits=splits,
            batch_size=args.batch_size,
            text_similarity_threshold=args.text_similarity_threshold,
            enable_audio_hash=args.enable_audio_hash
        )
        
        print("\n自動合併完成！")
        print("輸出檔案：")
        for split in splits:
            json_path = os.path.join(PROCESSED_DIR, f"{split}.json")
            if os.path.exists(json_path):
                print(f"  - {json_path}")
        
    else:
        # 傳統單一數據集模式（向後兼容）
        dataset_name = args.dataset_name or (args.dataset_names.split(",")[0] if args.dataset_names else "")
        if not dataset_name:
            print("錯誤：請指定 --dataset_name 或使用 --auto_merge 模式")
            return
        
        print(f"開始處理資料集：{dataset_name}")
        if args.language:
            print(f"指定語言：{args.language}")
            print(f"資料集路徑：data/raw/{dataset_name}/{args.language}/")
        else:
            print(f"自動偵測語言模式")
            print(f"資料集路徑：data/raw/{dataset_name}/")
        print(f"分割：{splits}")
        print(f"時長過濾：0.1s - 30.0s（固定範圍）")
        
        # 根據參數選擇處理模式
        if args.use_memory_optimized:
            print(f"記憶體優化模式：啟用（批次大小：{args.batch_size}）")
            print("="*60)
        else:
            print(f"標準模式（若記憶體不足，請使用 --use_memory_optimized）")
            print("="*50)
        
        # 執行批次處理模式
        total_duration = build_manifests_batch(
            dataset_name=dataset_name,
            language=args.language if args.language else None,
            splits=splits,
            batch_size=args.batch_size
        )
        
        # 印出總錄製時數
        total_hours = total_duration / 3600.0
        print(f"\n總錄製時數：{total_hours:.2f} 小時")
        print("全部完成。你可以在 data/processed/ 找到索引與轉檔後音檔。")


if __name__ == "__main__":
    main()
