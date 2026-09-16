#!/usr/bin/env python3
# Sample code: an illustrative benchmark for this example, not a maintained product. Adapt as needed.
"""LiveCodeBench: ExTS vs AB-MCTS-A vs Standard MCTS (real harness, minimal).

The actual LiveCodeBench harness used to benchmark ExTS (arXiv:2608.23848) on
TreeQuest (arXiv:2503.04412), trimmed to a minimal runnable script. ExTS is added by
exts.patch as `tq.ExTS`. Public test cases drive the tree-search reward; the
final pass@1 is judged on the hidden test cases (a solution counts only if it
passes all of them).

NOT runnable as-is. It requires:
  1. The LiveCodeBench dataset + sandboxed checker (`lcb_runner.*`). Clone
     https://github.com/LiveCodeBench/LiveCodeBench and set LCB_RUNNER_PATH to the
     checkout (defaults to a sibling ./LiveCodeBench). lcb_runner fetches the
     dataset from the Hugging Face Hub (needs network).
  2. AWS Bedrock credentials. boto3 reads them from the environment. Override the
     model / region with BEDROCK_MODEL_ID / AWS_REGION (default us-east-1).

Quick run (scales up via --num-problems / --budget; the paper used all
post-cutoff problems at budget 128):
    export LCB_RUNNER_PATH=/path/to/LiveCodeBench   # plus AWS credentials in env
    python run_livecodebench.py --num-problems 5 --budget 32
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# LiveCodeBench provides the dataset loader + sandboxed checker (lcb_runner.*).
_LCB = os.environ.get("LCB_RUNNER_PATH", str(Path(__file__).resolve().parent / "LiveCodeBench"))
if _LCB and _LCB not in sys.path:
    sys.path.insert(0, _LCB)

import boto3
import treequest as tq
from lcb_runner.benchmarks.code_generation import (
    CodeGenerationProblem,
    load_code_generation_dataset,
)
from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness
from lcb_runner.prompts.code_generation import (
    PromptConstants,
    get_generic_question_template_answer,
)

MODEL = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")
REGION = os.environ.get("AWS_REGION") or os.environ.get("BEDROCK_REGION", "us-east-1")
_client = None


def call_llm(system: str, prompt: str, temperature: float = 0.6, max_tokens: int = 4096) -> str:
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=REGION)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    })
    for attempt in range(5):
        try:
            resp = _client.invoke_model(modelId=MODEL, body=body)
            return json.loads(resp["body"].read())["content"][0]["text"]
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt + random.random())


def extract_code(text: str) -> str:
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if "```" in l]
    return "\n".join(lines[idx[-2] + 1: idx[-1]]) if len(idx) >= 2 else ""


def public_score(problem: CodeGenerationProblem, code: str) -> float:
    if not code.strip() or not problem.public_test_cases:
        return 0.0
    try:
        sample = {"input_output": json.dumps({
            "inputs": [t.input for t in problem.public_test_cases],
            "outputs": [t.output for t in problem.public_test_cases],
            "fn_name": problem.metadata.get("func_name"),
        })}
        res, _ = check_correctness(sample, code, timeout=6, debug=False)
        return sum(1 for r in res if r is True) / len(res)
    except Exception:
        return 0.0


def private_pass(problem: CodeGenerationProblem, code: str) -> bool:
    if not code.strip():
        return False
    try:
        res, _ = check_correctness(problem.get_evaluation_sample(), code, timeout=6, debug=False)
        return all(r is True for r in res)
    except Exception:
        return False


class State:
    __slots__ = ("code",)

    def __init__(self, code: str):
        self.code = code


def make_generate_fn(problem: CodeGenerationProblem):
    system = PromptConstants.SYSTEM_MESSAGE_GENERIC
    base = get_generic_question_template_answer(problem)

    def generate(parent):
        if parent is None:
            prompt = base
        else:
            prompt = base + (
                f"\n\nA previous attempt passed {public_score(problem, parent.code) * 100:.0f}% "
                f"of the visible tests:\n```python\n{parent.code}\n```\n\n"
                "Provide an improved solution in the same backticked format."
            )
        code = extract_code(call_llm(system, prompt))
        return State(code), public_score(problem, code)

    return generate


def make_algorithms(seed: int):
    return {
        "ExTS": tq.ExTS(
            exploration_type="puct", score_temperature=0.3, success_rate_alpha=1.414,
            max_children=3, expansion_gate_quantile=0.25, use_gated_expansion=True,
            use_progressive_widening=True, widening_base=2.0, widening_visit_threshold=32,
            widening_score_quantile=0.75, puct_parent_exp=0.5, seed=seed,
        ),
        "ABMCTSA": tq.ABMCTSA(dist_type="gaussian", model_selection_strategy="multiarm_bandit_thompson"),
        "StandardMCTS": tq.StandardMCTS(samples_per_action=5),
    }


def run_algorithm(algo, problem: CodeGenerationProblem, budget: int) -> bool:
    state = algo.init_tree()
    generate_fn = make_generate_fn(problem)
    best = 0.0
    for _ in range(budget):
        state = algo.step(state, {"generate": generate_fn})
        pairs = algo.get_state_score_pairs(state)
        if pairs:
            best = max(best, max(s for _, s in pairs))
        if best >= 1.0:
            break
    pairs = algo.get_state_score_pairs(state)
    best_code = tq.top_k(state, algo, k=1)[0][0].code if pairs else ""
    return private_pass(problem, best_code)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-problems", type=int, default=5,
                   help="Problems to run (default 5; the paper used all post-cutoff problems).")
    p.add_argument("--budget", type=int, default=32,
                   help="generate() calls per problem (default 32; the paper used 128).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--algos", nargs="+", default=["ExTS", "ABMCTSA", "StandardMCTS"],
                   choices=["ExTS", "ABMCTSA", "StandardMCTS"])
    p.add_argument("--release", default=os.environ.get("LCB_RELEASE", "release_v6"))
    p.add_argument("--start-date", default=os.environ.get("LCB_START_DATE", "2025-01-01"),
                   help="Keep problems released on/after this date (post model training cutoff).")
    args = p.parse_args()
    random.seed(args.seed)

    problems = load_code_generation_dataset(
        release_version=args.release, start_date=args.start_date)[: args.num_problems]
    print(f"Running {args.algos} on {len(problems)} LiveCodeBench problems "
          f"(budget={args.budget}, seed={args.seed})\n")

    passes = {a: 0 for a in args.algos}
    for i, problem in enumerate(problems):
        print(f"[{i + 1}/{len(problems)}] {problem.question_title} ({problem.difficulty.value})")
        algos = make_algorithms(args.seed)
        for a in args.algos:
            t0 = time.time()
            ok = run_algorithm(algos[a], problem, args.budget)
            passes[a] += int(ok)
            print(f"    {a:14s} {'PASS' if ok else 'fail'}  ({time.time() - t0:.0f}s)")

    n = len(problems) or 1
    print(f"\npass@1 over {len(problems)} problems:")
    for a in args.algos:
        print(f"  {a:14s} {passes[a]}/{len(problems)}  ({100 * passes[a] / n:.1f}%)")


if __name__ == "__main__":
    main()
