# ExTS x K-MSE example

K-MSE is an LLM tree-search method for molecular structure elucidation (ACL 2025). This example integrates ExTS as a controller-pattern drop-in: it is added as a sibling MCTS driver selected by a `--mcts_variant` flag, so K-MSE's original selector stays the default.

## Setup

`bash setup.sh` clones the pinned upstream repo and applies `exts.patch`.

Running the benchmark needs assets from the upstream repo (the neural scorer checkpoint and the dataset) and a CUDA GPU, all obtained from upstream.

## Run

> The benchmark script(s) here are **sample code**: an illustrative reference for this example, not a maintained product. Adapt them as needed.

The API key is read from the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY=...
bash run_benchmark.sh
```

## Upstream

Pinned to [`HICAI-ZJU/K-MSE`](https://github.com/HICAI-ZJU/K-MSE.git) at commit `94d4cb307f205454dbd8bd875df927b79c575498`.

## License

Upstream K-MSE is MIT; the ExTS additions here are CC BY-NC 4.0 (see [`../../LICENSE`](../../LICENSE)).
