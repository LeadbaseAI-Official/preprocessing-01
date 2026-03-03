"""
preprocessing.py  —  Standalone FineWeb Preprocessing Pipeline
================================================================
Self-contained: no project imports, all logic lives in this file.

Tokenizer: SentencePiece (.model format) — fast C++ trainer, trains in
minutes on a full corpus with no manual sampling step required.

Key design: uses a temp binary file instead of a Python list buffer.
Tokens are written with arr.tofile() and read back with np.fromfile(),
then pushed via PyArrow — zero Python-int overhead.
1B token shards cost only ~2 GB RAM.

Expected directory layout (relative to this script)
----------------------------------------------------
    ./tokenizer.model      SentencePiece model (train with slm.py first)
    ./.env                 must contain HF_TOKEN=<your_token>
    ./raw/                 raw .txt corpus files downloaded here

Install deps
------------
    pip install datasets sentencepiece python-dotenv tqdm numpy pyarrow

Run
---
    python preprocessing.py
"""


# ==============================================================================
# I M P O R T S
# ==============================================================================

import os
import sys
import tempfile
import numpy as np
import pyarrow as pa
import sentencepiece as spm
from tqdm import tqdm
from dotenv import load_dotenv, set_key
from datasets import load_dataset as hf_load_dataset, Dataset
from huggingface_hub import HfApi

# ==============================================================================
# C O N F I G
# ==============================================================================

CONFIG = {
    # ── FineWeb dataset ───────────────────────────────────────────────────────
    "hf_dataset":       "HuggingFaceFW/fineweb",
    "hf_text_key":      "text",
    "rows_to_download": 5,

    # ── Paths (flat, relative to this file) ───────────────────────────────────
    "raw_dir":          "raw",
    "raw_file":         "raw/fineweb.txt",
    "tokenizer_path":   "tokenizer.model",   # SentencePiece .model file
    "env_path":         ".env",

    "token_dtype":      np.uint16,
    "shard_size":       1_000_000_000,
    "split_doc":        True,  # Add BOS/EOS around documents
    "min_doc_chars":    200,

    # ── HuggingFace Hub ───────────────────────────────────────────────────────
    "hf_repo_id":       "anisoleai/fineweb-tokenized",
    "push_to_hub":      True,
}

# ==============================================================================
# U T I L I T I E S
# ==============================================================================

def _banner(title: str) -> None:
    """Print a clearly visible section banner."""
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}\n")


# ==============================================================================
# F U N C T I O N S
# ==============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# 1. DOWNLOAD
# ──────────────────────────────────────────────────────────────────────────────

def download_fineweb(
    hf_dataset:       str,
    text_key:         str,
    output_path:      str,
    rows_to_download: int,
    hf_token:         str | None,
    env_path:         str,
) -> None:
    """
    Stream rows from HuggingFace and append them to a .txt file.
    Progress is stored in progress.txt so runs are resumable.
    """
    progress_file = "progress.txt"
    last_index = 0
    if os.path.exists(progress_file):
        with open(progress_file, "r") as pf:
            try:
                last_index = int(pf.read().strip() or "0")
            except ValueError:
                last_index = 0

    print(f"  Streaming from : {hf_dataset}")
    print(f"  Rows requested : {rows_to_download:,}")
    print(f"  Resume offset  : {last_index:,}")
    print()

    ds = hf_load_dataset(hf_dataset, split="train", streaming=True, token=hf_token)
    ds = ds.skip(last_index)

    mode    = "a" if os.path.exists(output_path) else "w"
    written = 0

    with open(output_path, mode, encoding="utf-8") as f:
        pbar = tqdm(
            total=rows_to_download,
            desc="  Downloading ",
            unit=" rows",
            colour="blue",
            dynamic_ncols=True,
            file=sys.stdout,
        )
        for sample in ds.take(rows_to_download):
            text = sample.get(text_key, "")
            if not text:
                continue
            f.write(text.strip() + "\n\n")
            written += 1
            pbar.update(1)
        pbar.close()

    with open(progress_file, "w") as pf:
        pf.write(str(last_index + written))

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"\n  ✔  {written:,} rows  →  {output_path}  ({size_mb:.1f} MB)")


# ──────────────────────────────────────────────────────────────────────────────
# 2. TOKENIZER LOAD
# ──────────────────────────────────────────────────────────────────────────────

def load_tokenizer(tokenizer_path: str) -> spm.SentencePieceProcessor:
    """
    Load a SentencePiece model from a .model file.
    Raises FileNotFoundError clearly if it does not exist.
    """
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}\n"
            "tokenizer.model must exist before running this pipeline."
        )
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    print(f"  ✔  Tokenizer loaded  (vocab: {sp.get_piece_size():,} tokens)")
    return sp


# ──────────────────────────────────────────────────────────────────────────────
# 3. ENCODE → BINARY TEMP FILE  (same strategy as chunk_tokenize)
# ──────────────────────────────────────────────────────────────────────────────

def encode_text(tok: spm.SentencePieceProcessor, text: str) -> list[int]:
    """Encode a string → list of integer token ids."""
    return tok.encode(text, out_type=int)


def encode_to_bin(
    corpus_path:   str,
    tok:           spm.SentencePieceProcessor,
    dtype:         np.dtype,
    tmp_path:      str,
    split_doc:     bool = True,
    min_doc_chars: int = 50,
) -> int:
    """
    Stream-encode the corpus into a raw binary temp file.
    If split_doc is True, adds BOS/EOS tokens around \n\n separated documents.
    """
    import re
    file_size = os.path.getsize(corpus_path)
    sub_chunk = 10 * 1024 * 1024
    total     = 0

    bos_id = tok.bos_id() if split_doc else -1
    eos_id = tok.eos_id() if split_doc else -1

    pbar = tqdm(
        total=file_size, desc="  Encoding    ", unit="B",
        unit_scale=True, unit_divisor=1000, colour="green",
        dynamic_ncols=True, file=sys.stdout,
    )

    with open(corpus_path, "rb") as src, open(tmp_path, "wb") as dst:
        if not split_doc:
            while True:
                raw = src.read(sub_chunk)
                if not raw: break
                text = raw.decode("utf-8", errors="replace")
                tokens = tok.encode(text, out_type=int)
                arr = np.array(tokens, dtype=dtype)
                arr.tofile(dst)
                total += len(arr)
                pbar.update(len(raw))
                pbar.set_postfix(tokens=f"{total:,}")
        else:
            buffer = ""
            while True:
                raw = src.read(sub_chunk)
                if not raw: break
                text = raw.decode("utf-8", errors="replace")
                buffer += text
                parts = re.split(r"\n{2,}", buffer)
                buffer = parts.pop()

                for doc in parts:
                    doc = doc.strip()
                    if len(doc) < min_doc_chars: continue
                    toks = []
                    if bos_id >= 0: toks.append(bos_id)
                    toks.extend(tok.encode(doc, out_type=int))
                    if eos_id >= 0: toks.append(eos_id)
                    
                    if toks:
                        arr = np.array(toks, dtype=dtype)
                        arr.tofile(dst)
                        total += len(arr)
                
                pbar.update(len(raw))
                pbar.set_postfix(tokens=f"{total:,}")
            
            # Final document
            remainder = buffer.strip()
            if len(remainder) >= min_doc_chars:
                toks = []
                if bos_id >= 0: toks.append(bos_id)
                toks.extend(tok.encode(remainder, out_type=int))
                if eos_id >= 0: toks.append(eos_id)
                if toks:
                    arr = np.array(toks, dtype=dtype)
                    arr.tofile(dst)
                    total += len(arr)

    pbar.close()
    return total


# ──────────────────────────────────────────────────────────────────────────────
# 4. PUSH ONE SHARD VIA PYARROW  (zero Python-int overhead)
# ──────────────────────────────────────────────────────────────────────────────

def push_shard(
    arr:       np.ndarray,
    shard_idx: int,
    repo_id:   str,
    hf_token:  str,
    pbar:      tqdm,
) -> None:
    """
    Convert a numpy array → PyArrow table → HF Dataset → push_to_hub.

    PyArrow reads the numpy buffer directly (zero-copy) so no Python list of
    ints is ever created.  1B uint16 tokens costs only ~2 GB with this path.
    """
    pa_col = pa.array(arr, type=pa.uint16())        # zero-copy from numpy
    table  = pa.table({"token_ids": pa_col})
    ds     = Dataset(table)

    pbar.write(
        f"\n  → Pushing shard {shard_idx:04d}  "
        f"({len(arr):,} tokens  /  {arr.nbytes / 1e9:.2f} GB)  →  {repo_id} …"
    )

    ds.push_to_hub(
        repo_id,
        split="shard",
        token=hf_token,
        private=False,
        append=True if shard_idx > 1 else False
    )

    pbar.write(f"  ✔  Shard {shard_idx:04d} pushed.\n")


# ──────────────────────────────────────────────────────────────────────────────
# 5. READ BINARY FILE → PUSH SHARDS  (same strategy as chunk_tokenize Step 2)
# ──────────────────────────────────────────────────────────────────────────────

def push_from_bin(
    tmp_path:    str,
    total_tokens: int,
    shard_size:  int,
    dtype:       np.dtype,
    repo_id:     str,
    hf_token:    str,
) -> None:
    """
    Read back the temp binary file in shard_size chunks using np.fromfile(),
    then push each chunk to HF Hub via PyArrow.

    RAM cost: one shard at a time.
      1B tokens × uint16 → np.fromfile returns 2 GB numpy array.
      PyArrow conversion is zero-copy → ~2 GB total for 1B shard.
    """
    shard_index = 1
    pushed      = 0

    pbar = tqdm(
        total=total_tokens,
        desc="  Pushing     ",
        unit=" tok",
        unit_scale=True,
        colour="cyan",
        dynamic_ncols=True,
        file=sys.stdout,
    )

    with open(tmp_path, "rb") as f:
        while pushed < total_tokens:
            count = min(shard_size, total_tokens - pushed)
            arr   = np.fromfile(f, dtype=dtype, count=count)   # pure numpy, no Python ints
            if arr.size == 0:
                break

            push_shard(arr, shard_index, repo_id, hf_token, pbar)

            pbar.update(arr.size)
            pushed      += arr.size
            shard_index += 1

    pbar.close()
    print(f"\n  ✔  {total_tokens:,} tokens  across  {shard_index - 1} shard(s)")
    print(f"  ✔  Live at: https://huggingface.co/datasets/{repo_id}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. TOKENIZE AND PUSH
# ──────────────────────────────────────────────────────────────────────────────

def step_tokenize_and_push(cfg: dict, tok: spm.SentencePieceProcessor) -> None:
    """
    Encodes the corpus to a temp binary file and then pushes shards to HF Hub.
    """
    hf_token = os.getenv("HF_TOKEN") # Re-fetch token as it might be needed here

    # ── STEP 3: Encode → temp binary (arr.tofile) ─────────────────────────────
    _banner("STEP 3 · Encoding Corpus → Binary Temp File")
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".bin", prefix="tokens_")
    os.close(tmp_fd)

    try:
        total_tokens = encode_to_bin(
            corpus_path   = cfg["raw_file"],
            tok           = tok,
            dtype         = cfg["token_dtype"],
            tmp_path      = tmp_path,
            split_doc     = cfg.get("split_doc", True),
            min_doc_chars = cfg.get("min_doc_chars", 50),
        )

        # ── STEP 4: Read chunks → push via PyArrow ────────────────────────────
        _banner("STEP 4 · Pushing Shards to HuggingFace Hub")
        if not cfg["push_to_hub"]:
            print("  push_to_hub is False — skipping upload.")
        else:
            if not hf_token:
                raise EnvironmentError(
                    "HF_TOKEN not found in .env\n"
                    "Add HF_TOKEN=<your_token> to .env before running."
                )
            push_from_bin(
                tmp_path     = tmp_path,
                total_tokens = total_tokens,
                shard_size   = cfg["shard_size"],
                dtype        = cfg["token_dtype"],
                repo_id      = cfg["hf_repo_id"],
                hf_token     = hf_token,
            )

    finally:
        # Always clean up the temp binary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            print(f"  Temp binary cleaned up.")

    # ── STEP 5: Push Metadata (Tokenizer + README) ────────────────────────────
    if cfg["push_to_hub"] and hf_token:
        _banner("STEP 5 · Uploading Metadata to Hub")
        api = HfApi()
        
        # Upload tokenizer.model
        if os.path.exists(cfg["tokenizer_path"]):
            print(f"  → Uploading {cfg['tokenizer_path']} …")
            api.upload_file(
                path_or_fileobj=cfg["tokenizer_path"],
                path_in_repo="tokenizer.model",
                repo_id=cfg["hf_repo_id"],
                repo_type="dataset",
                token=hf_token,
            )
        
        # Upload README.md
        readme_path = "README.md"
        if os.path.exists(readme_path):
            print(f"  → Uploading {readme_path} …")
            api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=cfg["hf_repo_id"],
                repo_type="dataset",
                token=hf_token,
            )
        print("  ✔  Metadata uploaded.")

    _banner("PIPELINE COMPLETE ✔")
    print("  All steps finished successfully.\n")



# ==============================================================================
# M A I N   P I P E L I N E
# ==============================================================================

def run_pipeline(cfg: dict = CONFIG) -> None:
    """
    Full preprocessing pipeline:
        STEP 1  Download FineWeb rows from HuggingFace → raw/fineweb.txt
        STEP 2  Load SentencePiece tokenizer from tokenizer.model
        STEP 3  Encode corpus → temp binary file  (arr.tofile — no Python list)
        STEP 4  Read binary file in shard_size chunks → push via PyArrow
                1B token shard costs only ~2 GB RAM.
    """
    load_dotenv(cfg["env_path"])
    hf_token = os.getenv("HF_TOKEN")

    _banner("PREPROCESSING PIPELINE — FineWeb")
    print(f"  Dataset    : {cfg['hf_dataset']}")
    print(f"  Rows       : {cfg['rows_to_download']:,}")
    print(f"  Tokenizer  : {cfg['tokenizer_path']}")
    print(f"  Shard size : {cfg['shard_size']:,} tokens  "
          f"({cfg['shard_size'] * np.dtype(cfg['token_dtype']).itemsize / 1e9:.1f} GB RAM per shard)")
    print(f"  HF repo    : {cfg['hf_repo_id']}")
    print(f"  Push       : {cfg['push_to_hub']}")

    # ── STEP 1: Download ───────────────────────────────────────────────────────────────
    _banner("STEP 1 · Downloading FineWeb")
    os.makedirs(cfg["raw_dir"], exist_ok=True)
    download_fineweb(
        hf_dataset       = cfg["hf_dataset"],
        text_key         = cfg["hf_text_key"],
        output_path      = cfg["raw_file"],
        rows_to_download = cfg["rows_to_download"],
        hf_token         = hf_token,
        env_path         = cfg["env_path"],
    )

    # ── STEP 2: Load tokenizer ──────────────────────────────────────────────────────────
    _banner("STEP 2 · Loading SentencePiece Tokenizer")
    tok = load_tokenizer(cfg["tokenizer_path"])

    # ── STEP 3 & 4: Encode + push ───────────────────────────────────────────────────────
    step_tokenize_and_push(cfg, tok)

    _banner("PIPELINE COMPLETE ✔")
    print("  All steps finished successfully.\n")


if __name__ == "__main__":
    run_pipeline()
