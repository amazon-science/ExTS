# TreeQuest + ExTS

TreeQuest is Sakana AI's adaptive-branching Monte Carlo Tree Search (AB-MCTS) library for inference-time search. Because it ships its own tree-search framework, this example uses the framework-port pattern: ExTS is added as a sibling `Algorithm`, usable exactly like the built-in ones (`tq.ExTS()` alongside `tq.ABMCTSA()`).

```python
import treequest as tq
algo = tq.ExTS()
state = algo.init_tree()
state = algo.step(state, {...})
```

## Setup
```bash
bash setup.sh          # clones the pinned upstream and applies exts.patch
cd treequest
pip install -e .
pip install boto3      # AWS Bedrock client used by the benchmark
```

## Run

> The benchmark script(s) here are **sample code**: an illustrative reference for this example, not a maintained product. Adapt them as needed.
`benchmarks/run_livecodebench.py` is the actual LiveCodeBench harness used to benchmark ExTS (arXiv:2608.23848) on TreeQuest (arXiv:2503.04412), adapted only to import the algorithm as `tq.ExTS` (renamed from the fork's `ExTS0516`). It compares `tq.ExTS()` against TreeQuest's built-in `tq.ABMCTSA()` (Gaussian AB-MCTS-A) and `tq.StandardMCTS()`; public tests drive the search reward and hidden tests give the final pass@1.

It is NOT runnable as-is. It requires:
- The LiveCodeBench dataset and sandboxed checker (`lcb_runner.*`). Clone https://github.com/LiveCodeBench/LiveCodeBench and set `LCB_RUNNER_PATH` to the checkout (defaults to a sibling `./LiveCodeBench`). lcb_runner fetches the dataset from the Hugging Face Hub.
- AWS Bedrock credentials, which `boto3` reads from the environment (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`, or an instance profile). Override the model and region with `BEDROCK_MODEL_ID` and `AWS_REGION` (default `us-east-1`). Nothing is hardcoded.

```bash
export LCB_RUNNER_PATH=/path/to/LiveCodeBench   # plus AWS credentials in the env
python benchmarks/run_livecodebench.py --num-problems 5 --budget 32
```
It is a quick run by default and scales up via `--num-problems` and `--budget` (the paper used all post-cutoff problems at budget 128).

The ExTS unit tests need no data or credentials:
```bash
python -m pytest tests/test_exts.py
```

## Upstream
- URL: https://github.com/SakanaAI/treequest.git
- Commit: 96047d712d66bbbf4dcc86dcd3e2eaab98c35f83

## License
Upstream TreeQuest is Apache-2.0. The ExTS additions here are CC BY-NC 4.0 (see [../../LICENSE](../../LICENSE)).
