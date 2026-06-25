from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer


def count_records(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def sample_indices(total: int, sample_size: int) -> list[int]:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if sample_size > total:
        raise ValueError(f"sample_size {sample_size} exceeds record count {total}")
    if sample_size == 1:
        return [0]
    return [round(index * (total - 1) / (sample_size - 1)) for index in range(sample_size)]


def load_sample(path: Path, indices: list[int]) -> list[dict[str, Any]]:
    wanted = set(indices)
    records: list[dict[str, Any]] = []
    record_index = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            if record_index in wanted:
                records.append(json.loads(line))
            record_index += 1
    if len(records) != len(indices):
        raise RuntimeError(f"loaded {len(records)} records, expected {len(indices)}")
    return records


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Benchmark Title 12 embeddings on CUDA")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=root / "data" / "canonical" / "title12_2025-09-01" / "chunks.jsonl",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "models" / "bge-large-en-v1.5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "title12_gpu_benchmark.json",
    )
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment")
    if not args.chunks.is_file():
        raise FileNotFoundError(args.chunks)
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)

    total = count_records(args.chunks)
    indices = sample_indices(total, args.sample_size)
    records = load_sample(args.chunks, indices)
    texts = [record["embedding_text"] for record in records]

    load_started = time.perf_counter()
    model = SentenceTransformer(str(args.model), device="cuda", local_files_only=True)
    model_load_seconds = time.perf_counter() - load_started

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    encode_started = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    torch.cuda.synchronize()
    encode_seconds = time.perf_counter() - encode_started

    properties = torch.cuda.get_device_properties(0)
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_mib": round(properties.total_memory / 1024**2, 2),
        "model_path": str(args.model.resolve()),
        "chunks_path": str(args.chunks.resolve()),
        "total_chunks": total,
        "sample_method": "inclusive evenly spaced indices",
        "sample_size": args.sample_size,
        "first_sample_index": indices[0],
        "last_sample_index": indices[-1],
        "batch_size": args.batch_size,
        "model_load_seconds": round(model_load_seconds, 6),
        "encode_seconds": round(encode_seconds, 6),
        "chunks_per_second": round(args.sample_size / encode_seconds, 6),
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "normalized_embeddings": True,
        "peak_allocated_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "peak_reserved_memory_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 2),
        "cuda_oom": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
