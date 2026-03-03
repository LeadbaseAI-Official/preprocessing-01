"""
FineWeb → Tokenize → Push to HuggingFace

Requires:
    HF_TOKEN environment variable

Compatible with:
    datasets==4.6.1
    huggingface_hub==1.5.0
"""

import os
import tempfile
import numpy as np
import pyarrow as pa
import sentencepiece as spm
from tqdm import tqdm
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi


# ==========================================================
# CONFIG
# ==========================================================

HF_DATASET = "HuggingFaceFW/fineweb"
HF_TEXT_KEY = "text"
ROWS_TO_DOWNLOAD = 5

RAW_FILE = "raw.txt"
TOKENIZER_PATH = "tokenizer.model"

HF_REPO_ID = "anisoleai/fineweb-tokenized"

TOKEN_DTYPE = np.uint16
MIN_DOC_CHARS = 200
SPLIT_DOC = True


# ==========================================================
# DOWNLOAD
# ==========================================================

def download_data():

    print("STEP 1 — DOWNLOAD")

    ds = load_dataset(
        HF_DATASET,
        split="train",
        streaming=True,
        token=os.getenv("HF_TOKEN"),
    )

    with open(RAW_FILE, "w", encoding="utf-8") as f:
        for sample in tqdm(ds.take(ROWS_TO_DOWNLOAD), total=ROWS_TO_DOWNLOAD):
            text = sample.get(HF_TEXT_KEY, "")
            if text:
                f.write(text.strip() + "\n\n")

    print("✔ Download complete\n")


# ==========================================================
# LOAD TOKENIZER
# ==========================================================

def load_tokenizer():

    print("STEP 2 — LOAD TOKENIZER")

    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError("tokenizer.model not found")

    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    print(f"✔ Loaded tokenizer (vocab size: {sp.get_piece_size():,})\n")
    return sp


# ==========================================================
# ENCODE
# ==========================================================

def encode_corpus(sp):

    print("STEP 3 — ENCODE")

    bos = sp.bos_id()
    eos = sp.eos_id()

    all_tokens = []

    with open(RAW_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    docs = text.split("\n\n") if SPLIT_DOC else [text]

    for doc in tqdm(docs):
        doc = doc.strip()
        if len(doc) < MIN_DOC_CHARS:
            continue

        tokens = []

        if SPLIT_DOC and bos >= 0:
            tokens.append(bos)

        tokens.extend(sp.encode(doc, out_type=int))

        if SPLIT_DOC and eos >= 0:
            tokens.append(eos)

        all_tokens.extend(tokens)

    arr = np.array(all_tokens, dtype=TOKEN_DTYPE)

    print(f"✔ Encoded {len(arr):,} tokens\n")
    return arr


# ==========================================================
# PUSH TO HUB
# ==========================================================

def push_dataset(arr):

    print("STEP 4 — PUSH")

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN not set")

    # Create dataset
    pa_array = pa.array(arr, type=pa.uint16())
    table = pa.table({"token_ids": pa_array})
    ds = Dataset(table)

    print("Uploading dataset split 'shard'...")
    ds.push_to_hub(
        HF_REPO_ID,
        split="shard",
        token=hf_token,
        private=False,
    )

    print("✔ Dataset uploaded")

    # Upload tokenizer + README
    api = HfApi()

    if os.path.exists(TOKENIZER_PATH):
        api.upload_file(
            path_or_fileobj=TOKENIZER_PATH,
            path_in_repo="tokenizer.model",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=hf_token,
        )
        print("✔ tokenizer.model uploaded")

    if os.path.exists("README.md"):
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=hf_token,
        )
        print("✔ README uploaded")

    print("\n🚀 Push complete\n")


# ==========================================================
# MAIN
# ==========================================================

def main():

    if not os.getenv("HF_TOKEN"):
        raise EnvironmentError(
            "HF_TOKEN missing.\n"
            "Set it locally or via GitHub Actions secrets."
        )

    download_data()
    sp = load_tokenizer()
    arr = encode_corpus(sp)
    push_dataset(arr)


if __name__ == "__main__":
    main()
