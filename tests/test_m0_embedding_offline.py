"""M0: the cached embedding model loads with network disabled (marker: eval)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from conftest import MODELS_DIR, PROJECT_ROOT

pytestmark = pytest.mark.eval

_SNIPPET = """
import sys
from fastembed import TextEmbedding
m = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=sys.argv[1], local_files_only=True)
vecs = list(m.embed(["def parse_config(path): ...", "load settings from disk", "x"]))
print(len(vecs), *sorted({len(v) for v in vecs}))
"""


def test_cached_model_loads_offline():
    env = dict(os.environ)
    env.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        # Any attempted download is routed to a closed port and fails fast.
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "HTTP_PROXY": "http://127.0.0.1:9",
    })
    proc = subprocess.run(
        [sys.executable, "-c", _SNIPPET, str(MODELS_DIR)],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    count, *dims = proc.stdout.split()
    assert count == "3"
    assert len(dims) == 1 and int(dims[0]) > 0
