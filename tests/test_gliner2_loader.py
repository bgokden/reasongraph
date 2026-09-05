"""GLiNER2Extractor loads through gliner2.AutoExtractor when available (GLiNER 2.5
boundary checkpoints), else the legacy GLiNER2 span loader. Uses a stub module so
no model download is needed."""

import sys
import types

from reasongraph._extraction import GLiNER2Extractor


class _Model:
    def __init__(self, tag):
        self.tag = tag

    def extract_entities(self, text, types):
        return {"entities": {"organization": ["Redis"], "location": []}}


def _stub(with_auto: bool):
    mod = types.ModuleType("gliner2")
    calls = {}
    if with_auto:
        class AutoExtractor:
            @staticmethod
            def from_pretrained(name, **kw):
                calls["auto"] = (name, kw)
                return _Model("auto")
        mod.AutoExtractor = AutoExtractor

    class GLiNER2:
        @staticmethod
        def from_pretrained(name):
            calls["legacy"] = name
            return _Model("legacy")
    mod.GLiNER2 = GLiNER2
    return mod, calls


def test_prefers_autoextractor_when_present(monkeypatch):
    mod, calls = _stub(with_auto=True)
    monkeypatch.setitem(sys.modules, "gliner2", mod)
    ext = GLiNER2Extractor("fastino/gliner2.5-small-v1")
    assert ext("Redis is down") == ["Redis"]
    assert calls["auto"][0] == "fastino/gliner2.5-small-v1" and "legacy" not in calls


def test_falls_back_to_legacy_loader(monkeypatch):
    mod, calls = _stub(with_auto=False)
    monkeypatch.setitem(sys.modules, "gliner2", mod)
    ext = GLiNER2Extractor("fastino/gliner2-large-v1")
    assert ext("Redis is down") == ["Redis"]
    assert calls["legacy"] == "fastino/gliner2-large-v1"
