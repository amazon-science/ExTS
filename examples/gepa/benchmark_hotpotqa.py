#!/usr/bin/env python
# Sample code: an illustrative benchmark for this example, not a maintained product. Adapt as needed.
"""Minimal HotpotQA benchmark for GEPA with ExTS candidate selection.

This runs stock GEPA (from the patched upstream artifact) on HotpotQA, but with
`candidate_selection_strategy='exts'` so the ExTS tree drives which candidate
program is refined next instead of GEPA's default Pareto/greedy sampler.

Everything configurable (model, credentials, budget, dataset sizes) is read
from CLI flags / environment variables — no secrets or machine paths are
hardcoded.

Prerequisites (see ../README.md and setup.sh):
  * `bash setup.sh` has cloned + patched the artifact into ./gepa-artifact
  * upstream deps installed (`bash setup_gepa_repo.sh && uv sync`)
  * GEPA_MODEL + credentials exported (OPENAI_API_KEY / GEPA_API_KEY, or a
    local GEPA_API_BASE)

Note: HotpotMultiHop retrieves over the wiki.abstracts.2017 corpus using the
upstream artifact's retriever; make sure the artifact's retrieval setup is
complete (see the upstream repo). This is a small quick run; scale it up with
--train-size / --val-size / --max-metric-calls.
"""

import argparse
import os

import dspy

# HotpotMultiHop configures a retriever at import time; importing the benchmark
# meta gives us the program, metric and per-predictor feedback functions.
from gepa_artifact.benchmarks.hotpotQA import benchmark as hotpotqa_benchmark_metas
from gepa_artifact.gepa.gepa import GEPA
from gepa_artifact.utils.capture_stream_logger import Logger


def build_lm() -> dspy.LM:
    """Construct a DSPy LM from environment variables."""
    model = os.environ.get("GEPA_MODEL", "openai/gpt-4.1-mini")
    api_key = os.environ.get("GEPA_API_KEY") or os.environ.get("OPENAI_API_KEY")
    api_base = os.environ.get("GEPA_API_BASE")  # e.g. a local vLLM/arbor server

    if not api_key and not api_base:
        raise SystemExit(
            "No model credentials found. Set GEPA_API_KEY (or OPENAI_API_KEY), "
            "or point GEPA_API_BASE at a local OpenAI-compatible server."
        )

    kwargs = dict(model=model, max_tokens=16384, num_retries=0)
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base
    return dspy.LM(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=int(os.environ.get("GEPA_MAX_METRIC_CALLS", "200")),
        help="Optimization budget in metric calls (small by default).",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=int(os.environ.get("GEPA_TRAIN_SIZE", "20")),
        help="Number of HotpotQA training examples to use.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=int(os.environ.get("GEPA_VAL_SIZE", "20")),
        help="Number of HotpotQA validation examples to use.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=int(os.environ.get("GEPA_NUM_THREADS", "8")))
    parser.add_argument(
        "--run-dir",
        default=os.environ.get("GEPA_RUN_DIR", "./exts_hotpotqa_run"),
        help="Directory for logs, checkpoints and exts_tree.json.",
    )
    args = parser.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)

    dspy.configure(lm=build_lm())

    # HotpotQA benchmark meta: benchmark class, program(s), metric, feedback map.
    meta = hotpotqa_benchmark_metas[0]
    bench = meta.benchmark()             # loads HotpotQA (HuggingFace datasets)
    program = meta.program[0]            # HotpotMultiHop()
    feedback_fn_map = meta.feedback_fn_maps[0]

    trainset = bench.train_set[: args.train_size]
    valset = bench.val_set[: args.val_size]

    # ExTS hyperparameters forwarded to ExTSTree(**exts_kwargs). The defaults in
    # gepa_artifact/gepa/exts.py are sensible; tweak per-dataset here if desired.
    exts_kwargs = dict(
        exploration_type="puct",
        # exploration_constant=1.0,
        # success_rate_alpha=1.414,
        # score_temperature=0.3,
        # max_children=3,
    )

    logger = Logger(os.path.join(args.run_dir, "run_log.txt"))

    optimizer = GEPA(
        named_predictor_to_feedback_fn_map=feedback_fn_map,
        knowledgebase_qe=None,
        metric=meta.metric,
        logger=logger,
        run_dir=args.run_dir,
        run_linearized_gepa=False,               # keep full GEPA search loop
        candidate_selection_strategy="exts",     # <-- ExTS instead of pareto/greedy
        exts_kwargs=exts_kwargs,
        use_merge=False,
        track_scores_on="val",
        set_for_merge_minibatch="val",
        max_metric_calls=args.max_metric_calls,
        num_threads=args.num_threads,
        seed=args.seed,
        use_wandb=False,
    )

    print(
        f"Running GEPA+ExTS on HotpotQA: train={len(trainset)} val={len(valset)} "
        f"budget={args.max_metric_calls} metric calls, seed={args.seed}"
    )
    optimized_program = optimizer.compile(program, trainset=trainset, valset=valset)

    optimized_program.save(os.path.join(args.run_dir, "optimized_program"), save_program=True)
    print("Done.")
    print(f"  Optimized program : {os.path.join(args.run_dir, 'optimized_program')}")
    print(f"  ExTS search tree  : {os.path.join(args.run_dir, 'exts_tree.json')}")


if __name__ == "__main__":
    main()
