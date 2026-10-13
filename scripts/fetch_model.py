"""One-time download of the local embedding model into data/models/ (M0 only).

Run once with network access. Afterwards the model is loaded strictly offline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = os.environ.get("CODERAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CACHE_DIR = ROOT / "data" / "models"


def main() -> int:
    """Download the model, then report its embedding dimension."""
    from fastembed import TextEmbedding

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(CACHE_DIR))
    vec = next(iter(model.embed(["hello world"])))
    print(f"downloaded {MODEL_NAME} into {CACHE_DIR} (dim={len(vec)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
