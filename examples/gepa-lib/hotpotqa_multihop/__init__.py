"""HotpotQA fullwiki MULTI-HOP RAG benchmark for GEPA x ExTS.

First-party glue for optimizing, with ``gepa.optimize``, the per-module
instructions of a 4-module DSPy ChainOfThought pipeline over a ColBERTv2
retriever on the 2017 Wikipedia abstracts. The multi-hop program itself
(``HotpotMultiHop`` and its per-module feedback functions) is NOT included here;
the benchmark loads it from a gepa-artifact checkout (MIT, see the example
README). This package provides only the data loader, the two-process ColBERT
retrieval client/service, and a dspy compatibility shim.

Two-process architecture (see ``colbert_service.py`` / ``colbert_retriever.py``):
ColBERT's transformers==4.44.2 pin (tokenizers<0.20) is mutually exclusive with
litellm/dspy (tokenizers>=0.21), so ColBERT runs behind a local CPU HTTP service
and the DSPy/GEPA runner talks to it over HTTP.
"""
