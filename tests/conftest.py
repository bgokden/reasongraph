"""Shared pytest configuration.

Force models onto CPU for the test session. The suite loads several real
transformer models (NER, GLiNER2, GLiNER v2.5, cross-encoders, ONNX extractors);
on a machine whose GPU VRAM is largely used by other work, auto-loading them to
CUDA causes flaky ``torch.OutOfMemory`` errors that also take down unrelated
fake-embedder tests sharing the process. System RAM is far larger, so pinning
the tests to CPU makes the suite deterministic and portable.

Set ``CUDA_VISIBLE_DEVICES`` explicitly before running pytest to override (e.g.
``CUDA_VISIBLE_DEVICES=0 pytest`` to exercise the GPU path).
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
