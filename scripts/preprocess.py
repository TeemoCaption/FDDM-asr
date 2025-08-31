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
    下載 Common Voice 資料集從Hugging Face
    """
    print("開始下載 zh-TW 資料集從Hugging Face...")  # 提示使用者目前正在開始下載資料集，便於追蹤流程
    from datasets import load_dataset  # 從datasets套件匯入load_dataset函式，用於載入Hugging Face上的資料集
    
    # 使用新的方式載入資料集，避免使用舊版script-based載入
    dataset = load_dataset(
        "mozilla-foundation/common_voice_16_1",  # 指定資料集倉庫ID，對應Common Voice 16.1版本
        "zh-TW",  # 指定語言子集為繁體中文台灣
        split="train",  # 明確指定載入train split
        cache_dir=RAW_DATA_DIR,  # 指定快取目錄為專案的raw資料夾，方便管理與重複利用
        trust_remote_code=True  # 信任遠端代碼，避免因代碼問題而無法載入
    )
    print("下載完成")  # 提示下載動作已完成
    return {"train": dataset}  # 為了與extract_dataset()中使用dataset['train']的介面相容，將Dataset包成字典回傳

def extract_dataset(dataset):
    """
    處理Hugging Face資料集
    """
    extract_dir = os.path.join(RAW_DATA_DIR, LANGUAGE)
    os.makedirs(extract_dir, exist_ok=True)
    # 創建clips目錄
    clips_dir = os.path.join(extract_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    # 處理train set
    train_data = []
    for item in dataset['train']:
        # 取得來源音檔在快取中的路徑（通常為.mp3），後續僅用其檔名基底來生成WAV檔名
        audio_path = item['audio']['path']  # 從Hugging Face項目中取得音訊的原始檔案路徑（多半是快取中的mp3）
        # 從來源檔名萃取不含副檔名的基底名稱，準備生成對應的WAV檔名
        base_name = os.path.splitext(os.path.basename(audio_path))[0]  # 取得例如"xyz"而非"xyz.mp3"，以便後續改存為WAV
        # 組合WAV輸出完整路徑，確保以.wav為副檔名，避免以.mp3副檔名寫入導致失敗
        wav_filename = f"{base_name}.wav"  # 明確指定輸出檔名為WAV格式
        wav_output_path = os.path.join(clips_dir, wav_filename)  # 建立WAV輸出完整路徑，放在extract的clips資料夾
        # 將記憶體中的音訊array以WAV格式寫出，取樣率採用資料集中提供的sampling_rate
        sf.write(wav_output_path, item['audio']['array'], item['audio']['sampling_rate'])  # 使用soundfile寫出WAV，避免使用棄用API
        # 收集標註資料，path欄位改為WAV檔名，確保後續流程一致
        train_data.append({  # 建立一筆資料列，包含WAV檔名與文本與client_id等資訊
            'path': wav_filename,  # 使用剛輸出的WAV檔名，後續不需要再做副檔名替換
            'sentence': item['sentence'],  # 原始語句內容
            'client_id': item['client_id']  # 說話者ID
        })
    # 保存tsv
    import pandas as pd
    df = pd.DataFrame(train_data)
    df.to_csv(os.path.join(extract_dir, "train.tsv"), sep='\t', index=False)
    print("處理完成")
    return extract_dir

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

def generate_index(extract_dir):
    """
    生成資料索引（JSON和CSV格式）
    """
    # 讀取tsv檔案（Common Voice的標註檔案）
    tsv_path = os.path.join(extract_dir, "train.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"找不到標註檔案：{tsv_path}")
    # 讀取TSV（使用pandas）
    df = pd.read_csv(tsv_path, sep='\t')
    # 不再需要將.mp3替換為.wav，因為在extract階段已直接輸出為WAV檔名
    # 過濾訓練資料（根據路線圖，只取path, sentence, client_id等）
    df = df[['path', 'sentence', 'client_id']]  # 選擇必要欄位
    df = df.dropna()  # 移除空值
    # 正規化文本
    df['normalized_sentence'] = df['sentence'].apply(normalize_text)
    df['len_text'] = df['normalized_sentence'].apply(len)  # 計算文本長度
    # 過濾長度<300的樣本（根據路線圖）
    df = df[df['len_text'] < 300]
    # 設定處理後音檔路徑
    df['processed_path'] = df['path'].apply(lambda x: os.path.join(PROCESSED_DATA_DIR, "clips", x))  # 直接使用WAV檔名建立處理後路徑
    # 創建clips目錄
    os.makedirs(os.path.join(PROCESSED_DATA_DIR, "clips"), exist_ok=True)
    # 處理每個音檔
    for idx, row in df.iterrows():
        original_path = os.path.join(extract_dir, "clips", row['path'])  # 組合原始WAV音檔的完整路徑
        processed_path = row['processed_path']  # 取出處理後輸出目的路徑
        if os.path.exists(original_path):  # 檢查原始檔案是否存在，避免因缺檔而拋出例外
            process_audio(original_path, processed_path)  # 呼叫音檔處理流程（重取樣等）並寫出至處理後路徑
        else:
            print(f"音檔不存在：{original_path}")  # 若檔案不存在，提示以利除錯
    # 保存索引為JSON
    index_data = df.to_dict('records')
    json_path = os.path.join(PROCESSED_DATA_DIR, "index.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, ensure_ascii=False, indent=4)
    # 保存為CSV
    csv_path = os.path.join(PROCESSED_DATA_DIR, "index.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"索引文件生成完成：{json_path} 和 {csv_path}")
    return json_path, csv_path

def main():
    """
    主函數：執行整個前處理流程
    """
    print("開始前處理 Common Voice 16.1 zh-TW 資料集...")
    # 1. 下載資料集
    dataset = download_dataset()
    # 2. 處理資料集
    extract_dir = extract_dataset(dataset)
    # 3. 生成索引並處理資料
    generate_index(extract_dir)
    print("前處理完成！")

if __name__ == "__main__":
    main()
