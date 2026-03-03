"""
Incremental FineWeb → Tokenize → Push (CI Safe)

State stored on HuggingFace:
    progress.json

Each run:
    1. Reads progress.json
    2. Downloads next chunk
    3. Tokenizes
    4. Uploads new shard parquet
    5. Updates progress.json
"""

import os
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sentencepiece as spm
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download


# ==========================================================
# CONFIG
# ==========================================================

HF_DATASET = "HuggingFaceFW/fineweb"
HF_TEXT_KEY = "text"

ROWS_PER_RUN = 350_000   # Safe for CI

TOKENIZER_PATH = "tokenizer.model"
HF_REPO_ID = "anisoleai/fineweb-tokenized"

TOKEN_DTYPE = np.uint16
MIN_DOC_CHARS = 200
SPLIT_DOC = True


# ==========================================================
# PROGRESS HANDLING
# ==========================================================

def load_progress(api, token):

    try:
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="progress.json",
            repo_type="dataset",
            token=token,
        )
        with open(path, "r") as f:
            data = json.load(f)
            return data.get("last_index", 0), data.get("shard_index", 1)
    except Exception:
        return 0, 1


def save_progress(api, token, last_index, shard_index):

    progress = {
        "last_index": last_index,
        "shard_index": shard_index
    }

    with open("progress.json", "w") as f:
        json.dump(progress, f)

    api.upload_file(
        path_or_fileobj="progress.json",
        path_in_repo="progress.json",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=token,
    )


# ==========================================================
# DOWNLOAD NEXT CHUNK
# ==========================================================

def download_chunk(start_index, token):

    ds = load_dataset(
        HF_DATASET,
        split="train",
        streaming=True,
        token=token,
    )

    ds = ds.skip(start_index)

    texts = []
    count = 0

    for sample in tqdm(ds.take(ROWS_PER_RUN), total=ROWS_PER_RUN):
        text = sample.get(HF_TEXT_KEY, "")
        if text:
            texts.append(text.strip())
            count += 1

    return texts, count


# ==========================================================
# TOKENIZE
# ==========================================================

def tokenize_texts(texts):

    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError("tokenizer.model missing")

    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)

    bos = sp.bos_id()
    eos = sp.eos_id()

    all_tokens = []

    for doc in tqdm(texts):
        if len(doc) < MIN_DOC_CHARS:
            continue

        tokens = []

        if SPLIT_DOC and bos >= 0:
            tokens.append(bos)

        tokens.extend(sp.encode(doc, out_type=int))

        if SPLIT_DOC and eos >= 0:
            tokens.append(eos)

        all_tokens.extend(tokens)

    return np.array(all_tokens, dtype=TOKEN_DTYPE)


# ==========================================================
# UPLOAD SHARD
# ==========================================================

def upload_shard(arr, shard_index, api, token):

    if len(arr) == 0:
        print("No tokens generated. Skipping upload.")
        return

    table = pa.table({"token_ids": pa.array(arr, type=pa.uint16())})

    shard_name = f"data/shard-{shard_index:05d}.parquet"

    pq.write_table(table, "temp.parquet")

    api.upload_file(
        path_or_fileobj="temp.parquet",
        path_in_repo=shard_name,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=token,
    )

    print(f"Uploaded shard {shard_index}")


# ==========================================================
# MAIN
# ==========================================================

def main():

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN not set")

    api = HfApi()

    print("Loading progress...")
    last_index, shard_index = load_progress(api, hf_token)

    print(f"Resuming from index: {last_index}")
    print(f"Next shard index: {shard_index}")

    print("Downloading next chunk...")
    texts, downloaded = download_chunk(last_index, hf_token)

    if downloaded == 0:
        print("No new data available.")
        return

    print("Tokenizing...")
    arr = tokenize_texts(texts)

    print("Uploading shard...")
    upload_shard(arr, shard_index, api, hf_token)

    print("Updating progress...")
    save_progress(
        api,
        hf_token,
        last_index + downloaded,
        shard_index + 1
    )

    print("Run complete.")


if __name__ == "__main__":
    main()



