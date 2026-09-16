"""Compatibility shim: gepa's bundled DspyAdapter vs dspy 3.0.x.

The production gepa ``DspyAdapter`` (``gepa/adapters/dspy_adapter/dspy_adapter.py``)
is a *stale* snapshot -- its own header says the maintained copy lives in the DSPy
repo. It calls ``dspy.Evaluate(..., return_outputs=True, return_all_scores=True)``,
but dspy removed those kwargs (``return_outputs`` is hard-rejected since 3.0.2;
results always come back in ``EvaluationResult.results``). Meanwhile the adapter's
trace path needs ``bootstrap_trace_data(..., callback_metadata=...)``, which only
exists in dspy 3.0.4. So no released dspy 3.0.x works with the bundled adapter
unmodified.

Rather than edit the production gepa checkout, we make dspy's ``Evaluate`` tolerant:
strip the two dead kwargs before ``__init__`` validates them. Behavior is
unchanged -- the adapter only reads ``res.results`` (list of (example, prediction,
score) tuples), which dspy 3.0.4 always returns.
"""

from __future__ import annotations


def apply() -> None:
    from dspy.evaluate.evaluate import Evaluate

    if getattr(Evaluate, "_exts_return_outputs_compat", False):
        return

    _orig_init = Evaluate.__init__

    def __init__(self, *args, **kwargs):  # noqa: N807
        kwargs.pop("return_outputs", None)      # removed in dspy 3.0.2+
        kwargs.pop("return_all_scores", None)   # results always carry scores now
        _orig_init(self, *args, **kwargs)

    Evaluate.__init__ = __init__
    Evaluate._exts_return_outputs_compat = True
