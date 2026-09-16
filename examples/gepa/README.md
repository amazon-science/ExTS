# GEPA integration example

GEPA is the prompt-optimizer artifact from the GEPA paper (https://github.com/gepa-ai/gepa-artifact). This example integrates ExTS into GEPA as a controller-pattern drop-in candidate selector. It is enabled by a flag, so GEPA's original selector stays the default.

## Setup

Run `bash setup.sh`. It clones the pinned upstream artifact and applies `exts.patch`. The upstream repository has its own dependency setup, which you run inside the cloned directory.

## Run

> The benchmark script(s) here are **sample code**: an illustrative reference for this example, not a maintained product. Adapt them as needed.

Run `python benchmark_hotpotqa.py`. The model and credentials are read from environment variables.

## Pinned upstream

- URL: https://github.com/gepa-ai/gepa-artifact.git
- Commit: cbefbc1aa0f43dd39874ec4bf42211365dbda42e

## License

Upstream GEPA is licensed under MIT. The ExTS additions in this example are licensed under CC BY-NC 4.0 (see ../../LICENSE).
