# -*- coding: utf-8 -*-
"""
preprocess.py  (本地 Common Voice 版)

說明
----
本腳本僅支援「本地檔案夾」的 Common Voice 資料，不再使用 Hugging Face 下載。
請將 Common Voice 官方釋出的壓縮檔解壓到專案的 data/raw 目錄下，例如：
    data/raw/cv-corpus-22.0-2025-06-20/zh-TW/
其底下應該長這樣：
    clips/
    train.tsv
    dev.tsv
    test.tsv
    validated.tsv
    ...

輸出
----
在專案目錄下建立：
    data/processed/clips/             # 轉檔後的 16kHz WAV
    data/processed/{split}.json       # 索引（每行一筆 JSON 物件）
    data/processed/{split}.csv        # 索引（CSV 版）
其中 split 可能為 train / dev / test / validated 等。

使用範例
--------
python preprocess.py --dataset_name "cv-corpus-22.0-2025-06-20" --splits "train,dev,test"

註：腳本會自動偵測資料集中的所有語言資料夾並處理，使用預設時長過濾（0.1s-30s）

參數說明在 main() 底下有詳細註解。
"""

import os
import json
import argparse
import re
from typing import Dict, List

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


def process_language_batch(dataset_name: str,
                           language: str,
                           splits: List[str],
                           batch_size: int = 1000) -> float:
    """
    批次處理單一語言的資料集，避免一次性載入所有資料到記憶體
    
    參數
    ----
    batch_size : 每個批次處理的檔案數量，預設 1000
    
    回傳：該語言所有有效音檔的總時長（秒數）
    """
    # 初始化該語言的總時長變數
    total_duration_lang = 0.0
    
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

        # 準備輸出檔案路徑
        jsonl_path = os.path.join(PROCESSED_DIR, f"{language}_{split}.json")
        csv_path = os.path.join(PROCESSED_DIR, f"{language}_{split}.csv")
        
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
                    continue

                duration = to_wav_16k_mono(src_fp, out_abs, target_sr=16000)

                # 失敗就跳過
                if duration <= 0:
                    continue

                # 篩時長（使用固定的合理範圍）
                min_dur, max_dur = 0.1, 30.0  # 0.1秒到30秒的合理範圍
                if duration < min_dur or duration > max_dur:
                    continue

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
    
    # 印出該語言的總時長
    hours_lang = total_duration_lang / 3600.0
    print(f"完成語言 {language}: 總時長 {hours_lang:.2f} 小時")
    
    return total_duration_lang

def build_manifests_batch(dataset_name: str,
                         language: str,
                         splits: List[str],
                         batch_size: int = 1000) -> float:
    """
    批次處理主要函數：自動偵測語言並使用批次處理所有資料
    
    參數
    ----
    batch_size : 批次處理大小，預設 1000
    
    回傳：整個資料集的有效音檔總時長（秒數）
    """
    # 初始化整個資料集的總時長變數
    total_duration_dataset = 0.0
    
    # 固定從 data/raw 讀取資料集
    cv_root = os.path.join(RAW_DIR, dataset_name)              # e.g. data/raw/cv-corpus-22.0-2025-06-20
    
    # 檢查資料集是否存在
    if not os.path.isdir(cv_root):
        raise NotADirectoryError(f"找不到資料集目錄：{cv_root}\n請確認已將 Common Voice 資料集解壓到 data/raw/ 目錄下")
    
    print(f"從資料集讀取：{cv_root}（批次大小：{batch_size}）")
    
    # 自動偵測語言或使用指定語言
    if language:
        # 使用指定語言
        languages = [language]
        print(f"使用指定語言：{language}")
    else:
        # 自動偵測所有語言
        languages = detect_languages(cv_root)
        if not languages:
            raise ValueError(f"在 {cv_root} 中找不到任何語言資料夾")
        print(f"自動偵測到語言：{languages}")
    
    # 處理每個語言
    for lang in languages:
        try:
            # 批次處理單一語言並獲取其總時長
            duration_lang = process_language_batch(dataset_name, lang, splits, batch_size)
            # 累加到資料集總時長
            total_duration_dataset += duration_lang
        except Exception as e:
            print(f"處理語言 {lang} 時發生錯誤：{e}")
            continue
    
    # 印出整個資料集的總時長
    hours_dataset = total_duration_dataset / 3600.0
    print(f"\n資料集總時長：{hours_dataset:.2f} 小時")
    
    return total_duration_dataset

# =========================================================
# 參數與進入點
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="將本地 Common Voice 檔案夾轉為訓練可用的 16kHz WAV 與索引檔（記憶體優化版）"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="資料集資料夾名稱，例如 cv-corpus-22.0-2025-06-20（會從 data/raw/{dataset_name} 讀取）"
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
        "--use_memory_optimized",
        action="store_true",
        help="啟用記憶體優化模式（批次處理 + 串流輸出）"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 解析 splits
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    
    print(f"開始處理資料集：{args.dataset_name}")
    if args.language:
        print(f"指定語言：{args.language}")
        print(f"資料集路徑：data/raw/{args.dataset_name}/{args.language}/")
    else:
        print(f"自動偵測語言模式")
        print(f"資料集路徑：data/raw/{args.dataset_name}/")
    print(f"分割：{splits}")
    print(f"時長過濾：0.1s - 30.0s（固定範圍）")
    
    # 根據參數選擇處理模式
    if args.use_memory_optimized:
        print(f"記憶體優化模式：啟用（批次大小：{args.batch_size}）")
        print("="*60)
        
        # 執行批次處理模式
        total_duration = build_manifests_batch(
            dataset_name=args.dataset_name,
            language=args.language if args.language else None,
            splits=splits,
            batch_size=args.batch_size
        )
    else:
        print(f"標準模式（若記憶體不足，請使用 --use_memory_optimized）")
        print("="*50)
        
        # 執行標準處理模式
        total_duration = build_manifests_batch(  # 修正函數名稱為 build_manifests_batch，用於批次處理資料集
            dataset_name=args.dataset_name,
            language=args.language if args.language else None,
            splits=splits,
        )
    
    # 印出總錄製時數
    total_hours = total_duration / 3600.0
    print(f"\n總錄製時數：{total_hours:.2f} 小時")
    print("全部完成。你可以在 data/processed/ 找到索引與轉檔後音檔。")


if __name__ == "__main__":
    main()
