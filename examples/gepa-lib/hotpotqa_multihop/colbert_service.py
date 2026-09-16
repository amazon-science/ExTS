"""Local ColBERTv2 retrieval HTTP service (run in the .colbert_probe_venv, CPU-only).

A thin ColBERTv2 search service built on the public ``colbert`` Searcher API,
exposed over HTTP so the DSPy/GEPA runner (which needs a conflicting tokenizers
version) can call it from another process/venv. It mirrors the retrieval that
gepa-artifact's ``hover_program`` performs, but shares no code with it.

Endpoints:
  GET  /health                        -> {"ready": bool, "corpus_size": int}
  GET  /search?query=..&k=7           -> {"passages": [...]}
  POST /search  {"query": .., "k": 7} -> {"passages": [...]}   (preferred: long queries)

CPU-only: CUDA_VISIBLE_DEVICES="" for the whole process so the GPUs stay free for
vLLM. Results are diskcached in the shared dir ``<pkg>/colbert_cache`` so repeated
queries across seeds/selectors/runs are free.
"""

import os
import threading

# Must be set before importing torch / colbert.
os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "local")
# Optionally set TORCH_EXTENSIONS_DIR in your environment to cache compiled ops on
# fast local disk; left unset here so torch uses its own default location.
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU-only for the entire service process

import ujson  # noqa: E402
from diskcache import Cache  # noqa: E402
from flask import Flask, jsonify, request  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))

# Point these at your own ColBERTv2 index and wiki17 corpus (see the README).
INDEX_PATH = os.environ.get("COLBERT_INDEX_PATH", "./wiki17_colbertv2_index")
CORPUS_PATH = os.environ.get("COLBERT_CORPUS_PATH", "./wiki.abstracts.2017.jsonl")
# Shared cache dir: results/../hotpotqa_multihop/colbert_cache  ==  <pkg>/colbert_cache
CACHE_DIR = os.environ.get("COLBERT_CACHE_DIR", os.path.join(_HERE, "colbert_cache"))

_searcher = None
_corpus: list[str] | None = None
_ready = False
_init_lock = threading.Lock()
# ColBERT's Searcher holds shared torch buffers; serialize searches for thread safety.
_search_lock = threading.Lock()

cache = Cache(CACHE_DIR)


def init_colbert():
    global _searcher, _corpus, _ready
    if _ready:
        return
    with _init_lock:
        if _ready:
            return
        assert os.path.exists(INDEX_PATH), f"ColBERTv2 index not found at {INDEX_PATH}"
        assert os.path.exists(CORPUS_PATH), f"Corpus not found at {CORPUS_PATH}"

        # Redundant with the process-wide setting, but keep the pattern explicit.
        old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            from colbert import Searcher

            searcher = Searcher(index=INDEX_PATH)
        finally:
            if old_cuda is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""

        corpus: list[str] = []
        with open(CORPUS_PATH) as f:
            for line in f:
                rec = ujson.loads(line)
                corpus.append(f"{rec['title']} | {' '.join(rec['text'])}")

        _searcher = searcher
        _corpus = corpus
        _ready = True
        print(f"[colbert_service] ready: index={INDEX_PATH} corpus_size={len(corpus)}", flush=True)


@cache.memoize()
def _search_colbert(query: str, k: int) -> list[str]:
    init_colbert()
    with _search_lock:
        pids, _ranks, _scores = _searcher.search(query, k=k)
    return [_corpus[pid] for pid in pids][:k]


app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"ready": _ready, "corpus_size": (len(_corpus) if _corpus else 0)})


def _handle(query: str, k) -> "flask.Response":
    try:
        k = int(k)
    except (TypeError, ValueError):
        k = 7
    passages = _search_colbert(query or "", k)
    return jsonify({"passages": passages})


@app.get("/search")
def search_get():
    return _handle(request.args.get("query", ""), request.args.get("k", 7))


@app.post("/search")
def search_post():
    data = request.get_json(force=True, silent=True) or {}
    return _handle(data.get("query", ""), data.get("k", 7))


def main():
    # Binds localhost by default. This service is unauthenticated; do not expose it
    # on a public interface (setting COLBERT_SERVICE_HOST=0.0.0.0 is a local-only convenience).
    host = os.environ.get("COLBERT_SERVICE_HOST", "127.0.0.1")
    port = int(os.environ.get("COLBERT_SERVICE_PORT", "8899"))
    if os.environ.get("COLBERT_WARM", "1") == "1":
        init_colbert()  # load index + corpus before accepting traffic
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    main()
