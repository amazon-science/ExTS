"""HotpotQA fullwiki data loader for the multi-hop GEPA x ExTS benchmark.

Loads the **fullwiki** HotpotQA config from the HuggingFace Hub
(``hotpot_qa``/``fullwiki``; each example carries ``question``, ``answer``,
``supporting_facts`` = {title, sent_id}, ``context`` = {title, sentences}) as
``dspy.Example(...).with_inputs("question")`` -- the format gepa-artifact's
``HotpotMultiHop`` program and its feedback functions expect.

Splitting protocol (follows the gepa-artifact split convention):
  * Over the full ``train`` split, ordered: test = first 40%, val = 40-80%,
    train = last 20%.
  * A fixed ``random.Random(1)`` samples the requested sizes from each pool, so
    the TEST split is deterministic and **seed-independent** (a true held-out set).
  * For ``seed != 0``, only the (train + val) subsets are pooled, reshuffled with
    ``random.Random(seed)`` and re-split into train/val; the test split never moves.
"""

from __future__ import annotations

import random


def load_hotpotqa_fullwiki(seed: int = 0, train: int = 150, val: int = 300, test: int = 300):
    """Return ``(trainset, valset, testset)`` of ``dspy.Example`` objects.

    Args:
        seed: reshuffles the train/val pools (test is always held out and fixed).
        train, val, test: subset sizes.
    """
    import dspy
    from datasets import load_dataset

    raw = load_dataset("hotpot_qa", "fullwiki", trust_remote_code=True)["train"]
    examples = [dspy.Example(**x).with_inputs("question") for x in raw]

    n = len(examples)
    test_pool = examples[: int(0.4 * n)]           # first 40%  -> held-out test
    val_pool = examples[int(0.4 * n): int(0.8 * n)]  # 40-80%    -> val
    train_pool = examples[int(0.8 * n):]           # last 20%   -> train

    # Fixed RNG -> deterministic, seed-independent subsets.
    rng = random.Random(1)
    train_set = rng.sample(train_pool, train)
    val_set = rng.sample(val_pool, val)
    test_set = rng.sample(test_pool, test)  # held out; identical across seeds

    if seed != 0:
        pool = train_set + val_set
        random.Random(seed).shuffle(pool)
        train_set = pool[:train]
        val_set = pool[train: train + val]

    return train_set, val_set, test_set
