# scripts/tokenizer_train.py
# 功能：依據 configs/tokenizer_zhTW.yaml 設定，讀取 data/processed/index.csv (由 preprocess.py 產生)，
#      使用 SentencePiece 訓練 tokenizer，並將產物輸出到 data/tokenizer/ 下的對應目錄。
#
# 重要修正：
#   1) 只用 --unk_id/--bos_id/--eos_id/--pad_id 指定四個基本特殊符號的 id
#   2) 不再把 <unk>/<s></s>/<pad> 放進 --user_defined_symbols，避免與 control/special 衝突
#   3) 若未來需要額外特殊符號（例如 <noise>、<lang=tw>），再另外用 config.extra_user_symbols 指定
#
# 使用方式：
#   python scripts/tokenizer_train.py --config configs/tokenizer_zhTW.yaml
#
# 產出：
#   data/tokenizer/zh-TW_A/
#     ├─ spm_zhTW_A.model
#     ├─ spm_zhTW_A.vocab
#     └─ vocab.json  # 方便後續程式/設定檔讀取 K 與 token→id 對照
#
# 依賴：
#   pip install sentencepiece pyyaml pandas

import os
import io
import json
import argparse
import tempfile
import pandas as pd
import sentencepiece as spm
import yaml

def load_config(path: str) -> dict:
    """讀取 YAML 設定檔"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def prepare_corpus_text(index_csv: str,
                        text_field: str,
                        min_len: int,
                        max_len: int) -> list:
    """
    從 index.csv 讀出語料文字，並做長度篩選。
    參數：
        index_csv:  preprocess 產生的 CSV 路徑
        text_field: 欲用來訓練 tokenizer 的欄位（此專案為 normalized_sentence）
        min_len:    最短長度
        max_len:    最長長度
    回傳：
        List[str]：每個元素是一個句子
    """
    if not os.path.exists(index_csv):
        raise FileNotFoundError(f"找不到語料 CSV：{index_csv}\n請先執行 preprocess.py 產生 data/processed/index.csv")

    df = pd.read_csv(index_csv)
    if text_field not in df.columns:
        raise KeyError(f"CSV 中找不到欄位 `{text_field}`。可檢查 preprocess.py 的輸出欄位。")

    texts = []
    for s in df[text_field].astype(str).tolist():
        s = s.strip()
        if min_len is not None and len(s) < min_len:
            continue
        if max_len is not None and len(s) > max_len:
            s = s[:max_len]
        if s:
            texts.append(s)
    if len(texts) == 0:
        raise ValueError("語料為空，請檢查 index_csv / text_field / 長度門檻。")
    return texts

def write_corpus_to_tempfile(texts: list) -> str:
    """
    將語料寫到臨時檔，提供給 SentencePiece 訓練。
    回傳：臨時檔路徑
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
    with io.open(tmp.name, "w", encoding="utf-8") as f:
        for line in texts:
            f.write(line + "\n")
    return tmp.name

def build_spm_training_cmd(input_txt: str,
                           output_prefix: str,
                           model_type: str,
                           vocab_size: int,
                           character_coverage: float,
                           input_sentence_size: int,
                           shuffle_input_sentence: bool,
                           # 注意：這裡的 user_defined_symbols 只放「額外」符號，不包含 <unk>/<s></s>/<pad>
                           user_defined_symbols: list):
    """
    組 SentencePiece 的訓練指令字串（spm.SentencePieceTrainer.Train 的 arg 格式）。
    - 四個基本特殊符號交由 *_id 參數指定：unk_id/bos_id/eos_id/pad_id
    - user_defined_symbols 僅用於「額外」特殊符號（例如 <noise> 等）
    """
    # 濾掉不該進 user_defined 的基本符號（保險起見，即使 config 傳了也忽略）
    basic = {"<unk>", "<s>", "</s>", "<pad>"}
    uds = [s for s in (user_defined_symbols or []) if s not in basic]

    args = {
        "input": input_txt,
        "model_prefix": output_prefix,
        "model_type": model_type,
        "vocab_size": vocab_size,
        "character_coverage": character_coverage,
        "input_sentence_size": input_sentence_size,
        "shuffle_input_sentence": "true" if shuffle_input_sentence else "false",

        # 指定四個基本特殊符號的固定 id（避免和 user_defined 衝突）
        "unk_id": 0,   # <unk> = 0
        "bos_id": 1,   # <s> = 1
        "eos_id": 2,   # </s> = 2
        "pad_id": 3,   # <pad> = 3

        # 額外符號才放進來
        "user_defined_symbols": ",".join(uds) if len(uds) > 0 else ""
    }
    # 轉成 spm 接受的 CLI-style 字串
    arg_str = " ".join([f"--{k}={v}" for k, v in args.items() if v != "" and v is not None])
    return arg_str

def export_vocab_json(sp_model_path: str, save_json_path: str):
    """
    將 SentencePiece 的詞表輸出成 JSON 格式，包含：
      - id2token / token2id 對照
      - vocab_size
      - special_token_ids
    方便下游（擴散的 K、以及 token ↔ id）直接讀用。
    """
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)
    vocab_size = sp.get_piece_size()

    id2token = [sp.id_to_piece(i) for i in range(vocab_size)]
    token2id = {tok: i for i, tok in enumerate(id2token)}

    special = {
        "unk_id": sp.unk_id(),
        "bos_id": sp.bos_id(),
        "eos_id": sp.eos_id(),
        "pad_id": sp.pad_id(),
    }

    payload = {
        "vocab_size": vocab_size,
        "id2token": id2token,
        "token2id": token2id,
        "special_token_ids": special,
        "sp_model_path": sp_model_path
    }
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] 輸出 vocab JSON：{save_json_path}")

def main():
    parser = argparse.ArgumentParser(description="訓練 SentencePiece tokenizer（適用 zh-TW A 案）")
    parser.add_argument("--config", type=str, required=True, help="YAML 設定檔路徑（例：configs/tokenizer_zhTW.yaml）")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 讀取語料
    texts = prepare_corpus_text(
        index_csv=cfg["corpus"]["index_csv"],
        text_field=cfg["corpus"]["text_field"],
        min_len=cfg["corpus"].get("min_len", 1),
        max_len=cfg["corpus"].get("max_len", None)
    )

    # 準備輸出資料夾
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    # 產出臨時語料檔
    tmp_corpus = write_corpus_to_tempfile(texts)

    try:
        # 組訓練參數
        spm_prefix = os.path.join(out_dir, cfg["tokenizer"]["model_prefix"])

        # 從 config 讀「額外」特殊符號（選填）
        extra_user_syms = cfg.get("tokenizer", {}).get("extra_user_symbols", None)
        if extra_user_syms and not isinstance(extra_user_syms, list):
            raise TypeError("config.tokenizer.extra_user_symbols 必須是 list")

        spm_args = build_spm_training_cmd(
            input_txt=tmp_corpus,
            output_prefix=spm_prefix,
            model_type=cfg["tokenizer"]["model_type"],
            vocab_size=cfg["tokenizer"]["vocab_size"],
            character_coverage=cfg["tokenizer"]["character_coverage"],
            input_sentence_size=cfg["tokenizer"]["input_sentence_size"],
            shuffle_input_sentence=cfg["tokenizer"]["shuffle_input_sentence"],
            user_defined_symbols=extra_user_syms  # 只給「額外」符號
        )

        # 執行訓練
        print("[INFO] 開始訓練 SentencePiece...")
        spm.SentencePieceTrainer.train(spm_args)
        print("[OK] 訓練完成")

        # 匯出 vocab.json
        sp_model_path = f"{spm_prefix}.model"
        vocab_json_path = os.path.join(out_dir, cfg["output"]["export_vocab_json"])
        export_vocab_json(sp_model_path, vocab_json_path)

        print(f"[DONE] 產出：\n  - {sp_model_path}\n  - {spm_prefix}.vocab\n  - {vocab_json_path}")

    finally:
        # 清理臨時語料檔
        if os.path.exists(tmp_corpus):
            os.remove(tmp_corpus)

if __name__ == "__main__":
    main()
