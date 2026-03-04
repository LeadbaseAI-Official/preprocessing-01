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

ROWS_PER_RUN = 350_000
SHARDS_PER_WORKER = 1500

ROWS_PER_WORKER = ROWS_PER_RUN * SHARDS_PER_WORKER  # 525M rows per worker

TOKENIZER_PATH = "tokenizer.model"
HF_REPO_ID = "anisoleai/fineweb-tokenized"

TOKEN_DTYPE = np.uint16
MIN_DOC_CHARS = 200
SPLIT_DOC = True


# ==========================================================
# WORKER CONFIG
# ==========================================================

WORKER_ID = int(os.getenv("WORKER_ID", "1"))

DATA_FOLDER = f"data_{WORKER_ID}"
PROGRESS_FILE = f"progress/worker_{WORKER_ID}.json"


# ==========================================================
# WORKER RANGE
# ==========================================================

def worker_range():

    start = (WORKER_ID - 1) * ROWS_PER_WORKER
    end = WORKER_ID * ROWS_PER_WORKER

    return start, end


# ==========================================================
# PROGRESS
# ==========================================================

def load_progress(api, token):

    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=PROGRESS_FILE,
        repo_type="dataset",
        token=token,
    )

    with open(path) as f:
        data = json.load(f)

    return data["last_index"], data["shard_index"]


def save_progress(api, token, last_index, shard_index):

    progress = {
        "last_index": last_index,
        "shard_index": shard_index
    }

    local = f"worker_{WORKER_ID}.json"

    with open(local, "w") as f:
        json.dump(progress, f)

    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=PROGRESS_FILE,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=token,
    )


# ==========================================================
# DOWNLOAD
# ==========================================================

def download_chunk(start_index, token, limit):

    ds = load_dataset(
        HF_DATASET,
        split="train",
        streaming=True,
        token=token,
    )

    ds = ds.skip(start_index)

    texts = []
    count = 0

    for sample in tqdm(ds.take(limit), total=limit):

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
# UPLOAD
# ==========================================================

def upload_shard(arr, shard_index, api, token):

    if len(arr) == 0:
        print("No tokens generated. Skipping upload.")
        return

    table = pa.table({"token_ids": pa.array(arr, type=pa.uint16())})

    shard_name = f"{DATA_FOLDER}/shard-{shard_index:05d}.parquet"

    pq.write_table(table, "temp.parquet")

    api.upload_file(
        path_or_fileobj="temp.parquet",
        path_in_repo=shard_name,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        token=token,
    )

    print(f"Uploaded {shard_name}")


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

    worker_start, worker_end = worker_range()

    print(f"Worker {WORKER_ID}")
    print(f"Worker range: {worker_start} → {worker_end}")
    print(f"Current index: {last_index}")

    if last_index >= worker_end:

        print("Worker completed assigned range. Exiting.")
        return

    remaining = worker_end - last_index

    limit = min(ROWS_PER_RUN, remaining)

    print("Downloading chunk...")

    texts, downloaded = download_chunk(last_index, hf_token, limit)

    if downloaded == 0:

        print("No new data.")
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
