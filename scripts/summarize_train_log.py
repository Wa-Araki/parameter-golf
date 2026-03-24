#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def last_match(pattern: str, text: str):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return matches[-1] if matches else None


def first_match(pattern: str, text: str):
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.groupdict() if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize baseline-relevant metrics from train.log")
    parser.add_argument("--log", required=True, help="Path to train.log")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    text = log_path.read_text(encoding="utf-8", errors="replace")

    final_exact = first_match(
        r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[0-9.]+) val_bpb:(?P<val_bpb>[0-9.]+)",
        text,
    )
    if final_exact is None:
        final_fallback = last_match(r"step:\d+/\d+ val_loss:([0-9.]+) val_bpb:([0-9.]+)", text)
        if final_fallback is not None:
            final_exact = {"val_loss": final_fallback[0], "val_bpb": final_fallback[1]}

    compressed = first_match(
        r"Serialized model int8\+zlib: (?P<bytes>\d+) bytes",
        text,
    )
    total_submission = first_match(
        r"Total submission size int8\+zlib: (?P<bytes>\d+) bytes",
        text,
    )

    train_loader = first_match(
        r"train_loader:dataset:(?P<dataset>\S+) train_shards:(?P<train_shards>\d+)",
        text,
    )
    world = first_match(r"world_size:(?P<world_size>\d+) grad_accum_steps:(?P<grad_accum_steps>\d+)", text)
    train_cfg = first_match(
        r"train_batch_tokens:(?P<train_batch_tokens>\d+) train_seq_len:(?P<train_seq_len>\d+) "
        r"iterations:(?P<iterations>\d+) warmup_steps:(?P<warmup_steps>\d+) "
        r"max_wallclock_seconds:(?P<max_wallclock_seconds>[0-9.]+)",
        text,
    )
    seed = first_match(r"seed:(?P<seed>\d+)", text)
    tokenizer = first_match(
        r"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path=(?P<tokenizer_path>\S+)",
        text,
    )

    summary = {
        "log_path": str(log_path),
        "final": final_exact,
        "artifact": {
            "compressed_model_bytes": int(compressed["bytes"]) if compressed else None,
            "total_submission_bytes": int(total_submission["bytes"]) if total_submission else None,
        },
        "run_conditions": {
            "dataset": train_loader,
            "distribution": world,
            "train": train_cfg,
            "seed": seed["seed"] if seed else None,
            "tokenizer": tokenizer,
        },
    }

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
