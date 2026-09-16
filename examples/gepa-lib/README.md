# GEPA integration example

GEPA (`gepa-ai/gepa`) is a production, pip-installable library for optimizing
prompts and other text artifacts through reflective evolution. This example
integrates ExTS with **zero changes to GEPA's core**: it passes an
`ExTSCandidateSelector` (implementing GEPA's `CandidateSelector` protocol) to
`gepa.optimize(...)` via `candidate_selection_strategy`, the injected-selector
form of the controller pattern. GEPA's default selector (`"pareto"`) stays the default.

```python
result = gepa.optimize(
    ...,
    candidate_selection_strategy=ExTSCandidateSelector(exploration_type="puct"),
)
```

## Setup

```bash
bash setup.sh                 # clone the pinned upstream and apply exts.patch
cd upstream && pip install -e ".[full]"
```

## Run

> The benchmark here is **sample code**: an illustrative reference for this example, not a maintained product. Adapt it as needed.

`benchmark_hotpotqa_multihop.py` (driven by `run_matched_multihop.sh`) follows the paper's HotpotQA fullwiki multi-hop protocol on the production library: `gepa.optimize` optimizes the four module instructions of a DSPy `HotpotMultiHop` pipeline (`summarize1 -> create_query_hop2 -> summarize2 -> final_answer`) over a ColBERTv2 retriever, with ExTS injected as the candidate selector. `run_matched_multihop.sh` runs original (Pareto) vs ExTS across seeds 0/42/1024 and reports the best candidate's held-out test EM and F1. The `HotpotMultiHop` program and its per-module feedback functions are **not included here**; the benchmark loads them from a [gepa-artifact](https://github.com/gepa-ai/gepa-artifact) checkout (see below).

It is NOT runnable as-is. It needs:
- The HotpotQA multi-hop program (`HotpotMultiHop` + its per-module feedback functions), which is **not included here**. Clone [gepa-artifact](https://github.com/gepa-ai/gepa-artifact) (MIT), set `GEPA_ARTIFACT_PATH` to the checkout, and point that program's retriever import at `hotpotqa_multihop.colbert_retriever.search` (replacing `from ..hover.hover_program import search_colbert as search`).
- An LLM endpoint (vLLM or any OpenAI-compatible server) via `OPENAI_API_BASE` + `OPENAI_API_KEY` (or `GEPA_MODEL` / `GEPA_API_BASE`), plus `dspy`.
- A ColBERTv2 index over the 2017 Wikipedia abstracts corpus, served CPU-only by `hotpotqa_multihop/colbert_service.py`. Point `COLBERT_INDEX_PATH` and `COLBERT_CORPUS_PATH` at the index and corpus. Because `colbert-ai` pins `tokenizers<0.20` (conflicting with dspy/litellm), `run_matched_multihop.sh` uses two virtualenvs: `.colbert_probe_venv` (colbert-ai) and `.runner_venv` (dspy + gepa + litellm).
- Real fullwiki HotpotQA via the HuggingFace `datasets` package.

```bash
export GEPA_ARTIFACT_PATH=/path/to/gepa-artifact    # provides HotpotMultiHop (MIT, not shipped here)
export OPENAI_API_BASE=http://localhost:8001/v1    # your LLM endpoint (+ OPENAI_API_KEY)
export COLBERT_INDEX_PATH=/path/to/wiki17_colbertv2_index
export COLBERT_CORPUS_PATH=/path/to/wiki.abstracts.2017.jsonl
bash run_matched_multihop.sh                       # original vs ExTS, seeds 0/42/1024
```

## Upstream and license

- Upstream: `https://github.com/gepa-ai/gepa.git`, pinned to commit
  `8b0ce6cd99a234f6b74daf37558a2ac0ce18f975` (tag `v0.1.4`).
- Upstream GEPA is MIT. The HotpotQA multi-hop program used by the benchmark comes from [gepa-artifact](https://github.com/gepa-ai/gepa-artifact) (MIT, (c) 2025 Lakshya A Agrawal) and is **not redistributed here**; obtain it from the upstream project. Only the ExTS additions authored here are CC BY-NC 4.0 (see [../../LICENSE](../../LICENSE)).
