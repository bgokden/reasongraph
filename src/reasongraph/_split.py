"""Sentence splitting for ingest.

Every model in the pipeline (entity extractor, causal span pointer, gate, embedder) is
trained on single sentences, so a paragraph pushed as one fact hurts all of them. With a
splitter set, :meth:`ReasonGraph.add_texts` stores one fact per sentence.

Two implementations:

* :class:`SaTSplitter` -- Segment-any-Text (``wtpsplit``, model ``sat-3l-sm`` by default):
  85 languages, punctuation-agnostic, handles abbreviations and unpunctuated text. Optional
  dependency (``pip install reasongraph[split]``); ~1 ms per paragraph on CPU after load.
* :class:`RegexSplitter` -- dependency-free fallback on terminal punctuation and newlines.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class SentenceSplitter(Protocol):
    def split(self, text: str) -> list[str]: ...


class RegexSplitter:
    """Split on newlines and on ``. ! ? …`` followed by whitespace and an opening
    character. Keeps common abbreviations (``e.g.``, ``z.B.``, ``Dr.``) together."""

    _ABBR = re.compile(r"\b(?:e\.g|i\.e|z\.B|d\.h|u\.a|vs|Dr|Mr|Mrs|Ms|Prof|St|No|Nr|ca|approx)\.$", re.I)
    _END = re.compile(r"(?<=[.!?…])\s+(?=[\"'(\[\w])", re.U)

    def split(self, text: str) -> list[str]:
        out: list[str] = []
        for para in re.split(r"\n+", text):
            buf = ""
            for piece in self._END.split(para.strip()):
                if not piece:
                    continue
                if buf and self._ABBR.search(buf):
                    buf = buf + " " + piece
                elif buf:
                    out.append(buf)
                    buf = piece
                else:
                    buf = piece
            if buf:
                out.append(buf)
        return [s.strip() for s in out if s.strip()]


class SaTSplitter:
    """Segment-any-Text splitter (lazy-loaded)."""

    def __init__(self, model: str = "sat-3l-sm", *, ort_providers: list[str] | None = None) -> None:
        self.model = model
        self.ort_providers = ort_providers
        self._sat = None

    def _load(self) -> None:
        if self._sat is not None:
            return
        try:
            from wtpsplit import SaT
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise ImportError("SaTSplitter needs wtpsplit: pip install 'reasongraph[split]'") from exc
        self._sat = SaT(self.model, ort_providers=self.ort_providers) if self.ort_providers else SaT(self.model)

    def split(self, text: str) -> list[str]:
        self._load()
        return [s.strip() for s in self._sat.split(text) if s and s.strip()]


def resolve_splitter(name: str | SentenceSplitter | None) -> SentenceSplitter | None:
    """``None``/``"off"`` -> None; ``"sat"`` -> SaTSplitter; ``"regex"`` -> RegexSplitter;
    ``"sat:<model>"`` picks a SaT model; a splitter object passes through."""
    if name is None or name is False:
        return None
    if not isinstance(name, str):
        return name
    key = name.strip().lower()
    if key in ("", "off", "none", "0", "false"):
        return None
    if key in ("regex", "simple"):
        return RegexSplitter()
    if key.startswith("sat"):
        model = name.split(":", 1)[1] if ":" in name else "sat-3l-sm"
        return SaTSplitter(model)
    raise ValueError(f"unknown sentence splitter {name!r} (use 'sat', 'sat:<model>', 'regex' or 'off')")
