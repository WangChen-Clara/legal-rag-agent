from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    runtime_cache = root / ".runtime_cache" / "huggingface"
    runtime_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(runtime_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(runtime_cache / "hub"))
    model_path = root / "models" / "bge-large-en-v1.5"
    index_path = root / "data" / "legacy_indexes" / "bge-large-embeddings" / "vector_db.index"
    metadata_path = root / "data" / "legacy_indexes" / "bge-large-embeddings" / "metadata.npy"

    started = time.perf_counter()
    import faiss
    import networkx
    import numpy as np
    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer

    load_started = time.perf_counter()
    model = SentenceTransformer(str(model_path), device="cpu")
    model_load_seconds = time.perf_counter() - load_started

    query = "What is the minimum common equity tier 1 capital ratio?"
    encode_started = time.perf_counter()
    vector = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    encode_seconds = time.perf_counter() - encode_started

    index = faiss.read_index(str(index_path))
    metadata = np.load(metadata_path, allow_pickle=True).tolist()
    search_started = time.perf_counter()
    distances, indices = index.search(vector, 5)
    search_seconds = time.perf_counter() - search_started

    hits = []
    for rank, (distance, item_index) in enumerate(
        zip(distances[0], indices[0]), start=1
    ):
        item = metadata[int(item_index)]
        hits.append(
            {
                "rank": rank,
                "distance": float(distance),
                "index": int(item_index),
                "doc_id": item.get("doc_id") if isinstance(item, dict) else None,
                "source_file": item.get("source_file") if isinstance(item, dict) else None,
                "text_preview": (
                    str(item.get("chunk", ""))[:300]
                    if isinstance(item, dict)
                    else str(item)[:300]
                ),
            }
        )

    vector_norm = float(np.linalg.norm(vector[0]))
    checks = {
        "python_310": sys.version_info[:2] == (3, 10),
        "model_dimension_matches_index": vector.shape[1] == index.d,
        "index_metadata_count_matches": index.ntotal == len(metadata),
        "query_vector_is_normalized": abs(vector_norm - 1.0) < 1e-5,
        "top_5_returned": len(hits) == 5,
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "runtime": {
            "python": platform.python_version(),
            "executable": sys.executable,
            "packages": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "sentence_transformers": sentence_transformers.__version__,
                "networkx": networkx.__version__,
                "numpy": np.__version__,
                "faiss": getattr(faiss, "__version__", "unknown"),
            },
        },
        "assets": {
            "model_path": str(model_path),
            "index_path": str(index_path),
            "metadata_path": str(metadata_path),
            "vectors": index.ntotal,
            "metadata_records": len(metadata),
            "dimension": index.d,
        },
        "checks": checks,
        "timing_seconds": {
            "model_load": round(model_load_seconds, 4),
            "query_encode": round(encode_seconds, 4),
            "faiss_search": round(search_seconds, 6),
            "total": round(time.perf_counter() - started, 4),
        },
        "query": query,
        "query_vector_norm": vector_norm,
        "hits": hits,
    }

    report_path = root / "reports" / "environment_validation.json"
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
