"""ColBERTv2 retriever client (two-process architecture).

colbert-ai==0.2.22 pins transformers==4.44.2, which requires tokenizers<0.20,
while litellm/dspy require tokenizers>=0.21. These are mutually exclusive, so
ColBERT cannot live in the same Python process as the DSPy/GEPA runner. Instead
ColBERT runs behind a tiny local HTTP service (``colbert_service.py``, launched in
the ``.colbert_probe_venv``), and this module is a thin client. Point
gepa-artifact's ``HotpotMultiHop`` retriever import at its ``search`` function --
a drop-in replacement for the in-process ``hover_program.search_colbert``.

Interface (identical to the in-process version):
    search(query: str, k: int = 7) -> DotDict   # .passages -> list[str]

Each passage is shaped exactly like ``f"{title} | {' '.join(text)}"``.
The expensive ColBERT call + its diskcache live in the service; this client
additionally memoizes HTTP responses locally in the shared cache dir so repeated
queries across runs never even hit the network.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request


class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


COLBERT_SERVICE_URL = os.environ.get("COLBERT_SERVICE_URL", "http://127.0.0.1:8899").rstrip("/")
_HTTP_TIMEOUT = float(os.environ.get("COLBERT_HTTP_TIMEOUT", "180"))
_HTTP_RETRIES = int(os.environ.get("COLBERT_HTTP_RETRIES", "6"))


def _post_search(query: str, k: int) -> list[str]:
    body = json.dumps({"query": query, "k": int(k)}).encode("utf-8")
    req = urllib.request.Request(
        COLBERT_SERVICE_URL + "/search",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["passages"]


def search(query: str, k: int = 7) -> DotDict:
    """Retrieve top-k passages from the ColBERT service. Retries on transient errors."""
    last_err: Exception | None = None
    for attempt in range(_HTTP_RETRIES):
        try:
            passages = _post_search(query, k)
            return DotDict({"passages": passages[:k]})
        except Exception as e:  # noqa: BLE001 - service may still be warming up
            last_err = e
            time.sleep(min(2.0 * (attempt + 1), 15.0))
    raise RuntimeError(
        f"ColBERT service at {COLBERT_SERVICE_URL} unreachable after {_HTTP_RETRIES} tries: {last_err!r}"
    )


def health() -> dict:
    with urllib.request.urlopen(COLBERT_SERVICE_URL + "/health", timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_until_ready(timeout: float = 900.0, poll: float = 3.0) -> dict:
    """Block until the service reports ready (index + corpus loaded)."""
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        try:
            h = health()
            if h.get("ready"):
                return h
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(poll)
    raise TimeoutError(f"ColBERT service not ready within {timeout}s (last: {last!r})")
