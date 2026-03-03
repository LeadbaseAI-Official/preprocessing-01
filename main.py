"""
preprocessing.py  —  Standalone FineWeb Preprocessing Pipeline
================================================================

Environment variable required:
    HF_TOKEN=<your_huggingface_token>

Install deps:
    pip install datasets sentencepiece tqdm numpy pyarrow huggingface_hub

Run:
    export HF_TOKEN=your_token
    python preprocessing.py
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
import sys
import tempfile
import numpy as np
import pyarrow as pa
import sentencepiece as spm
from tqdm import tqdm
from datasets import load_dataset as hf_load_dataset, Dataset
from huggingface_hub import HfApi

# ==============================================================================
# CONFIG
# ==============================================================================

CONFIG = {
    "hf_dataset":       "HuggingFaceFW/fineweb",
    "hf_text_key":      "text",
    "rows_to_download": 5,

    "raw_dir":          "raw",
    "raw_file":         "raw/fineweb.txt",
    "tokenizer_path":   "tokenizer.model",

    "token_dtype":      np.uint16,
    "shard_size":       1_000_000_000,
    "split_doc":        True,
    "min_doc_chars":    200,

    "hf_repo_id":       "anisoleai/fineweb-tokenized",
    "push_to_hub":      True,
}

# ==============================================================================
# UTIL
# ==============================================================================

def _banner(title: str):
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}\n")


# ==============================================================================
# STEP 1 — DOWNLOAD
# ==============================================================================

def download_fineweb(hf_dataset, text_key, output_path, rows_to_download, hf_token):

    print(f"Streaming from: {hf_dataset}")
    ds = hf_load_dataset(hf_dataset, split="train", streaming=True, token=hf_token)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in tqdm(ds.take(rows_to_download), total=rows_to_download):
            text = sample.get(text_key, "")
            if text:
                f.write(text.strip() + "\n\n")
                written += 1

    print(f"✔ Downloaded {written} rows")


# ==============================================================================
# STEP 2 — TOKENIZER
# ==============================================================================

def load_tokenizer(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Tokenizer not found at {path}. "
            "Place tokenizer.model in project root."
        )

    sp = spm.SentencePieceProcessor(model_file=path)
    print(f"✔ Tokenizer loaded (vocab size: {sp.get_piece_size():,})")
    return sp


# ==============================================================================
# STEP 3 — ENCODE TO BINARY
# ==============================================================================

def encode_to_bin(corpus_path, tok, dtype, tmp_path, split_doc=True, min_doc_chars=200):

    total = 0
    bos = tok.bos_id() if split_doc else None
    eos = tok.eos_id() if split_doc else None

    with open(corpus_path, "r", encoding="utf-8") as src, \
         open(tmp_path, "wb") as dst:

        buffer = src.read()

        docs = buffer.split("\n\n") if split_doc else [buffer]

        for doc in tqdm(docs):
            doc = doc.strip()
            if len(doc) < min_doc_chars:
                continue

            tokens = []
            if split_doc and bos is not None:
                tokens.append(bos)

            tokens.extend(tok.encode(doc, out_type=int))

            if split_doc and eos is not None:
                tokens.append(eos)

            arr = np.array(tokens, dtype=dtype)
            arr.tofile(dst)
            total += len(arr)

    return total


# ==============================================================================
# STEP 4 — PUSH SHARDS
# ==============================================================================

def push_from_bin(tmp_path, total_tokens, shard_size, dtype, repo_id, hf_token):

    shard_index = 1
    pushed = 0

    with open(tmp_path, "rb") as f:

        while pushed < total_tokens:

            count = min(shard_size, total_tokens - pushed)
            arr = np.fromfile(f, dtype=dtype, count=count)

            if arr.size == 0:
                break

            pa_col = pa.array(arr, type=pa.uint16())
            table = pa.table({"token_ids": pa_col})
            ds = Dataset(table)

            print(f"Pushing shard {shard_index} ({len(arr):,} tokens)")

            ds.push_to_hub(
                repo_id,
                split="shard",
                token=hf_token,
                private=False,
                append=True if shard_index > 1 else False,
            )

            pushed += arr.size
            shard_index += 1

    print(f"✔ Pushed {total_tokens:,} tokens")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_pipeline(cfg=CONFIG):

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token and cfg["push_to_hub"]:
        raise EnvironmentError(
            "HF_TOKEN not found in environment variables.\n"
            "Set it before running:\n"
            "  export HF_TOKEN=your_token (Linux/macOS)\n"
            "  set HF_TOKEN=your_token (Windows)\n"
            "Or use GitHub Actions secrets."
        )

    _banner("STEP 1 — DOWNLOAD")
    download_fineweb(
        cfg["hf_dataset"],
        cfg["hf_text_key"],
        cfg["raw_file"],
        cfg["rows_to_download"],
        hf_token,
    )

    _banner("STEP 2 — LOAD TOKENIZER")
    tok = load_tokenizer(cfg["tokenizer_path"])

    _banner("STEP 3 — ENCODE")
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".bin")
    os.close(tmp_fd)

    try:
        total_tokens = encode_to_bin(
            cfg["raw_file"],
            tok,
            cfg["token_dtype"],
            tmp_path,
            cfg["split_doc"],
            cfg["min_doc_chars"],
        )

        _banner("STEP 4 — PUSH")
        if cfg["push_to_hub"]:
            push_from_bin(
                tmp_path,
                total_tokens,
                cfg["shard_size"],
                cfg["token_dtype"],
                cfg["hf_repo_id"],
                hf_token,
            )

        # Upload tokenizer + README
        api = HfApi()

        if os.path.exists(cfg["tokenizer_path"]):
            api.upload_file(
                path_or_fileobj=cfg["tokenizer_path"],
                path_in_repo="tokenizer.model",
                repo_id=cfg["hf_repo_id"],
                repo_type="dataset",
                token=hf_token,
            )

        if os.path.exists("README.md"):
            api.upload_file(
                path_or_fileobj="README.md",
                path_in_repo="README.md",
                repo_id=cfg["hf_repo_id"],
                repo_type="dataset",
                token=hf_token,
            )

        print("✔ Metadata uploaded")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    _banner("PIPELINE COMPLETE")


if __name__ == "__main__":
    run_pipeline()
