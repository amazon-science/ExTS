#!/usr/bin/env python
# Sample code: an illustrative benchmark for this example, not a maintained product. Adapt as needed.
"""GEPA x ExTS -- HotpotQA fullwiki MULTI-HOP RAG optimization (or aggregation).

Optimizes the per-module instructions of a 4-module DSPy ChainOfThought pipeline
(``HotpotMultiHop``: summarize1 -> create_query_hop2 -> summarize2 -> final_answer)
over a ColBERTv2 retriever on 2017 Wikipedia abstracts, using ``gepa.optimize``
with the gepa-artifact HotpotQA protocol (EM metric + per-module textual feedback).
ColBERT runs in a separate process behind a local HTTP service (see
``hotpotqa_multihop/colbert_service.py``) because its tokenizers pin conflicts with
litellm/dspy.

The only difference between the two arms is ``candidate_selection_strategy``:
  * ``original`` -> GEPA's built-in ``"pareto"`` selector.
  * ``exts``     -> ``ExTSCandidateSelector(exploration_type="puct")``.

After optimization, the single best candidate is evaluated ONCE on a held-out TEST
split (seed-independent) and both EM and F1 are reported for the same predictions.

Model + credentials come from the environment (vLLM / LiteLLM):

    export OPENAI_API_BASE=http://localhost:8001/v1
    export OPENAI_API_KEY=dummy
    export COLBERT_SERVICE_URL=http://127.0.0.1:8899
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import dspy
import gepa
from gepa.adapters.dspy_adapter.dspy_adapter import DspyAdapter
from gepa.strategies.exts_candidate_selector import ExTSCandidateSelector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hotpotqa_multihop import _dspy_compat

_dspy_compat.apply()  # make dspy 3.0.4 Evaluate tolerate gepa DspyAdapter dead kwargs

from hotpotqa_multihop.hotpot_data import load_hotpotqa_fullwiki

# This example does NOT redistribute gepa-artifact's HotpotQA program. Obtain the
# 4-module HotpotMultiHop program and its per-module feedback functions from
# gepa-artifact (MIT, (c) 2025 Lakshya A Agrawal):
#   https://github.com/gepa-ai/gepa-artifact  ->  benchmarks/hotpotQA/hotpot_program.py
# Set GEPA_ARTIFACT_PATH to your checkout, and point that program's retriever
# import at this package's two-process HTTP client, i.e. replace
#     from ..hover.hover_program import search_colbert as search
# with
#     from hotpotqa_multihop.colbert_retriever import search
_artifact_path = os.environ.get("GEPA_ARTIFACT_PATH")
if _artifact_path:
    sys.path.insert(0, _artifact_path)
try:
    from gepa_artifact.benchmarks.hotpotQA.hotpot_program import (  # noqa: E402
        HotpotMultiHop,
        answer_exact_match_with_feedback,
        feedback_fn_map,
    )
except ImportError as _e:
    raise SystemExit(
        "Could not import gepa-artifact's HotpotQA program (HotpotMultiHop / "
        "feedback_fn_map). This example does not redistribute it: clone "
        "https://github.com/gepa-ai/gepa-artifact (MIT), set GEPA_ARTIFACT_PATH to "
        "the checkout, and point its retriever import at "
        "hotpotqa_multihop.colbert_retriever.search. Original error: " + repr(_e)
    )

try:
    from dspy.dsp.utils import EM, F1
except ImportError:  # dspy 3.x moved these
    from dspy.evaluate.metrics import EM, F1


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _build_lm() -> "dspy.LM":
    api_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("GEPA_API_BASE")
    if not api_base:
        raise SystemExit("Set OPENAI_API_BASE to the vLLM /v1 endpoint (e.g. http://localhost:8001/v1).")
    model = os.environ.get("GEPA_MODEL", "openai/qwen/qwen3-8b")
    return dspy.LM(
        model,
        api_base=api_base,
        api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
        temperature=float(os.environ.get("GEPA_TASK_TEMPERATURE", "0.6")),
        top_p=float(os.environ.get("GEPA_TASK_TOP_P", "0.95")),
        max_tokens=_env_int("GEPA_MAX_TOKENS", 2048),
        extra_body={"top_k": _env_int("GEPA_TOP_K", 20)},
    )


def _test_metric_em(example, pred, trace=None):
    return EM(getattr(pred, "answer", "") or "", [example.answer])


def _evaluate_test(best_program, testset, workers: int) -> tuple[float, float, int]:
    """Run the best program ONCE on the held-out test split; return (EM, F1, n)."""
    evaluator = dspy.Evaluate(
        devset=testset,
        metric=_test_metric_em,
        num_threads=workers,
        failure_score=0.0,
        provide_traceback=True,
        max_errors=len(testset) * 100,
        display_progress=True,
    )
    res = evaluator(best_program)
    ems, f1s = [], []
    for ex, pred, _score in res.results:
        ans = getattr(pred, "answer", "") or ""
        ems.append(float(EM(ans, [ex.answer])))
        f1s.append(float(F1(ans, [ex.answer])))
    n = max(1, len(ems))
    return sum(ems) / n, sum(f1s) / n, len(ems)


def run_one(args: argparse.Namespace) -> dict:
    lm = _build_lm()
    dspy.configure(lm=lm)

    train, val, test = load_hotpotqa_fullwiki(
        seed=args.seed, train=args.train_size, val=args.val_size, test=args.test_size
    )
    print(
        f"[multihop] data: fullwiki  train={len(train)} val={len(val)} test={len(test)}  "
        f"(held-out test is seed-independent)"
    )
    print(f"[multihop] sample train question: {train[0].question!r}  gold={train[0].answer!r}")

    program = HotpotMultiHop()
    program.set_lm(lm)
    seed_candidate = {name: p.signature.instructions for name, p in program.named_predictors()}
    assert set(seed_candidate.keys()) == set(feedback_fn_map.keys()), (
        f"predictor names {sorted(seed_candidate)} != feedback keys {sorted(feedback_fn_map)}"
    )
    print(f"[multihop] modules: {sorted(seed_candidate.keys())}")

    workers = _env_int("GEPA_WORKERS", 32)
    adapter = DspyAdapter(
        student_module=program,
        metric_fn=answer_exact_match_with_feedback,
        feedback_map=feedback_fn_map,
        reflection_lm=lm,
        num_threads=workers,
    )

    if args.selector == "exts":
        selection_strategy = ExTSCandidateSelector(rng=random.Random(args.seed), exploration_type="puct")
    elif args.selector == "original":
        selection_strategy = "pareto"
    else:
        raise SystemExit(f"Unknown selector: {args.selector!r}")

    print(
        f"[multihop] selector={args.selector} seed={args.seed} budget={args.max_metric_calls} "
        f"workers={workers} model={os.environ.get('GEPA_MODEL','openai/qwen/qwen3-8b')}"
    )

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=train,
        valset=val,
        adapter=adapter,
        candidate_selection_strategy=selection_strategy,
        module_selector="round_robin",
        reflection_minibatch_size=3,
        max_metric_calls=args.max_metric_calls,
        seed=args.seed,
        display_progress_bar=True,
        raise_on_exception=False,
    )

    val_select = result.val_aggregate_scores[result.best_idx]
    print(
        f"[multihop] optimization done: #candidates={result.num_candidates} "
        f"best_idx={result.best_idx} val_select={val_select:.4f} "
        f"total_metric_calls={result.total_metric_calls}"
    )

    best = adapter.build_program(result.best_candidate)
    test_em, test_f1, n_test = _evaluate_test(best, test, workers)

    record = {
        "seed": args.seed,
        "selector": args.selector,
        "test_em": test_em,
        "test_f1": test_f1,
        "val_select": val_select,
        "num_candidates": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "budget": args.max_metric_calls,
        "train_size": len(train),
        "val_size": len(val),
        "test_size": n_test,
        "model": os.environ.get("GEPA_MODEL", "openai/qwen/qwen3-8b"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{args.selector}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    print(
        f"[multihop] DONE selector={args.selector} seed={args.seed} "
        f"test_EM={test_em:.4f} test_F1={test_f1:.4f} val_select={val_select:.4f} "
        f"#cand={result.num_candidates} calls={result.total_metric_calls} -> {out_path}"
    )
    return record


def aggregate(results_dir: str) -> None:
    files = sorted(Path(results_dir).glob("*.json"))
    by_selector: dict[str, list[dict]] = {}
    for fp in files:
        try:
            rec = json.loads(fp.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "selector" in rec and "test_em" in rec:
            by_selector.setdefault(rec["selector"], []).append(rec)

    if not by_selector:
        print(f"No result files found in {results_dir}. Run the benchmark first.")
        return

    print("\n===== HotpotQA MULTI-HOP (fullwiki): ORIGINAL vs ExTS -- held-out TEST, best candidate =====")
    print(f"{'selector':<11}{'n':>3}   {'EM mean':>8}{'EM std':>8}    {'F1 mean':>8}{'F1 std':>8}    {'seeds':>14}")
    print("-" * 84)
    for selector in sorted(by_selector):
        recs = sorted(by_selector[selector], key=lambda r: r["seed"])
        em = [r["test_em"] for r in recs]
        f1 = [r["test_f1"] for r in recs]
        seeds = ",".join(str(r["seed"]) for r in recs)
        em_m = statistics.mean(em)
        em_s = statistics.stdev(em) if len(em) > 1 else 0.0
        f1_m = statistics.mean(f1)
        f1_s = statistics.stdev(f1) if len(f1) > 1 else 0.0
        print(f"{selector:<11}{len(em):>3}   {em_m:>8.4f}{em_s:>8.4f}    {f1_m:>8.4f}{f1_s:>8.4f}    {seeds:>14}")
    print("=" * 84)
    print("(Primary optimization metric = exact match; F1 is the same best candidate. Higher is better.)\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selector", choices=["original", "exts"], default="original")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-size", type=int, default=_env_int("GEPA_TRAIN_SIZE", 150))
    p.add_argument("--val-size", type=int, default=_env_int("GEPA_VAL_SIZE", 300))
    p.add_argument("--test-size", type=int, default=_env_int("GEPA_TEST_SIZE", 300))
    p.add_argument("--max-metric-calls", type=int, default=_env_int("GEPA_MAX_METRIC_CALLS", 6871))
    p.add_argument(
        "--results-dir",
        default=os.environ.get("GEPA_RESULTS_DIR", str(Path(__file__).with_name("results") / "matched_multihop")),
    )
    p.add_argument("--aggregate", action="store_true", help="Aggregate result files and print a table, then exit.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.aggregate:
        aggregate(args.results_dir)
        return
    run_one(args)


if __name__ == "__main__":
    sys.exit(main())
