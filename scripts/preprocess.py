# preprocess.py
# 此腳本用於前處理 Common Voice 16.1 zh-TW 資料集
# 包括下載、解壓、音檔處理、文本正規化以及生成資料索引

import os  # 用於檔案路徑操作
import json  # 用於處理JSON格式資料
import pandas as pd  # 用於資料處理和CSV輸出
import librosa  # 用於音檔處理，如resample
import soundfile as sf  # 新增：統一於頂部匯入soundfile，負責WAV等音檔的讀寫，避免在函式內重複匯入
import re  # 用於文本正規化

# 設定資料集版本和語言
DATASET_VERSION = "cv-corpus-16.1-2023-12-06"  # Common Voice 16.1 版本
LANGUAGE = "zh-TW"  # 語言為繁體中文台灣

# 設定路徑
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # 專案根目錄
DATA_DIR = os.path.join(ROOT_DIR, "data")  # 原始與處理後資料存放目錄
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")  # 原始資料存放目錄
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")  # 處理後資料存放目錄

# 確保目錄存在
os.makedirs(RAW_DATA_DIR, exist_ok=True)  # 如果raw目錄不存在，則創建
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)  # 如果processed目錄不存在，則創建

# 使用Hugging Face下載

def download_dataset():
    """
    下載 Common Voice 資料集從 Hugging Face，使用內建的分割
    """
    print("開始下載 zh-TW 資料集從 Hugging Face (使用內建分割)...")
    from datasets import load_dataset

    # 使用 Hugging Face 內建的分割
    dataset = load_dataset(
        "mozilla-foundation/common_voice_16_1",
        "zh-TW",
        split=["train", "validation", "test"],  # 使用內建的 train/validation/test 分割
        cache_dir=RAW_DATA_DIR,
        trust_remote_code=True
    )

    print("下載完成")
    print(f"  訓練集: {len(dataset[0])} 樣本")
    print(f"  開發集: {len(dataset[1])} 樣本")
    print(f"  測試集: {len(dataset[2])} 樣本")

    # 返回字典格式以保持與現有程式碼的兼容性
    return {
        "train": dataset[0],
        "validation": dataset[1],
        "test": dataset[2]
    }

def extract_dataset(dataset):
    """
    處理 Hugging Face 資料集，使用內建分割
    """
    extract_dir = os.path.join(RAW_DATA_DIR, LANGUAGE)
    os.makedirs(extract_dir, exist_ok=True)

    # 為每個分割建立目錄
    splits_data = {}

    for split_name in ["train", "validation", "test"]:
        print(f"處理 {split_name} 分割...")

        # 建立分割專用目錄
        split_dir = os.path.join(extract_dir, split_name)
        clips_dir = os.path.join(split_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)

        # 處理當前分割的資料
        split_data = []
        split_dataset = dataset[split_name]

        for item in split_dataset:
            # 取得來源音檔在快取中的路徑
            audio_path = item['audio']['path']
            # 從來源檔名萃取不含副檔名的基底名稱
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            # 組合WAV輸出完整路徑
            wav_filename = f"{base_name}.wav"
            wav_output_path = os.path.join(clips_dir, wav_filename)

            # 將記憶體中的音訊array以WAV格式寫出
            sf.write(wav_output_path, item['audio']['array'], item['audio']['sampling_rate'])

            # 收集標註資料
            split_data.append({
                'path': wav_filename,
                'sentence': item['sentence'],
                'client_id': item['client_id']
            })

        # 保存分割的 TSV 檔案
        import pandas as pd
        df = pd.DataFrame(split_data)
        tsv_path = os.path.join(split_dir, f"{split_name}.tsv")
        df.to_csv(tsv_path, sep='\t', index=False)

        splits_data[split_name] = {
            'dir': split_dir,
            'tsv': tsv_path,
            'data': split_data
        }

        print(f"{split_name} 分割處理完成: {len(split_data)} 樣本")

    return splits_data

def normalize_text(text):
    """
    正規化文本：轉小寫、移除多餘標點、處理數字等
    """
    # 移除括號內的羅馬字（根據路線圖建議，可選A案）
    text = re.sub(r'\([^)]*\)', '', text)  # 移除括號及其內容
    # 轉小寫
    text = text.lower()
    # 移除多餘空白
    text = re.sub(r'\s+', ' ', text).strip()
    # 正規化標點（簡化處理，可根據需要擴充）
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)  # 只保留中文、英文和數字
    return text

def process_audio(audio_path, output_path):
    """
    處理音檔：resample到16kHz，轉單聲道
    """
    # 載入音檔
    y, sr = librosa.load(audio_path, sr=None)  # 保持原取樣率
    # 如果是多聲道，取第一聲道
    if y.ndim > 1:
        y = y[:, 0]  # 取第一聲道
    # resample到16kHz
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)  # 將音訊重取樣到16kHz，符合ASR常見需求
    # 保存處理後的音檔
    sf.write(output_path, y_resampled, 16000)  # 改用soundfile寫出WAV，避免使用已棄用的librosa.output.write_wav

def generate_index(splits_data):
    """
    生成資料索引（JSON和CSV格式），使用 Hugging Face 內建分割
    """
    for split_name, split_info in splits_data.items():
        print(f"生成 {split_name} 分割的索引...")

        split_dir = split_info['dir']
        tsv_path = split_info['tsv']
        split_data = split_info['data']

        # 讀取分割的 TSV 檔案
        df = pd.read_csv(tsv_path, sep='\t')

        # 正規化文本
        df['normalized_sentence'] = df['sentence'].apply(normalize_text)
        df['len_text'] = df['normalized_sentence'].apply(len)  # 計算文本長度

        # 設定處理後音檔路徑
        clips_dir = os.path.join(split_dir, "clips")
        df['processed_path'] = df['path'].apply(lambda x: os.path.join(PROCESSED_DATA_DIR, "clips", f"{split_name}_{x}"))

        # 處理每個音檔
        for idx, row in df.iterrows():
            original_path = os.path.join(clips_dir, row['path'])
            processed_path = row['processed_path']

            if os.path.exists(original_path):
                process_audio(original_path, processed_path)
            else:
                print(f"音檔不存在：{original_path}")

        # 保存為 JSON
        index_data = df.to_dict('records')
        json_path = os.path.join(PROCESSED_DATA_DIR, f"{split_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=4)

        # 保存為 CSV
        csv_path = os.path.join(PROCESSED_DATA_DIR, f"{split_name}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')

        print(f"{split_name} 索引檔案生成完成: {json_path}, {csv_path}")

    return list(splits_data.keys())

def main():
    """
    主函數：執行整個前處理流程
    """
    print("開始前處理 Common Voice 16.1 zh-TW 資料集...")
    # 1. 下載資料集（使用內建分割）
    dataset = download_dataset()
    # 2. 處理資料集
    splits_data = extract_dataset(dataset)
    # 3. 生成索引並處理資料
    generate_index(splits_data)
    print("前處理完成！")

if __name__ == "__main__":
    main()
