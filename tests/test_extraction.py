import os

import pytest

from reasongraph._extraction import OnnxTokenClassifierExtractor

# Typed-BIO scheme used by the place-entity model.
PLACE_LABELS = {
    0: "O", 1: "B-CITY", 2: "I-CITY", 3: "B-COUNTRY",
    4: "I-COUNTRY", 5: "B-ENTITY", 6: "I-ENTITY",
}

_decode = OnnxTokenClassifierExtractor._decode_bio


def test_decode_bio_multi_token_span():
    text = "New York France"
    offsets = [(0, 0), (0, 3), (4, 8), (9, 15), (0, 0)]  # CLS, New, York, France, SEP
    tags = [0, 1, 2, 3, 0]  # O, B-CITY, I-CITY, B-COUNTRY, O
    assert _decode(text, offsets, tags, PLACE_LABELS, None) == ["New York", "France"]


def test_decode_bio_keep_types_filter():
    text = "New York France"
    offsets = [(0, 0), (0, 3), (4, 8), (9, 15), (0, 0)]
    tags = [0, 1, 2, 3, 0]
    assert _decode(text, offsets, tags, PLACE_LABELS, {"COUNTRY"}) == ["France"]


def test_decode_bio_b_tag_splits_same_type():
    # Two adjacent B-CITY tokens must stay two separate entities, not merge.
    text = "Paris Rome"
    offsets = [(0, 0), (0, 5), (6, 10), (0, 0)]
    tags = [0, 1, 1, 0]  # O, B-CITY, B-CITY, O
    assert _decode(text, offsets, tags, PLACE_LABELS, None) == ["Paris", "Rome"]


def test_decode_bio_dedups_repeats():
    text = "Rome and Rome"
    offsets = [(0, 0), (0, 4), (5, 8), (9, 13), (0, 0)]
    tags = [0, 1, 0, 1, 0]  # Rome, O, Rome
    assert _decode(text, offsets, tags, PLACE_LABELS, None) == ["Rome"]


def test_decode_bio_untyped_scheme():
    # A bare B/I scheme (no type suffix) falls back to a generic 'ENT' type.
    labels = {0: "O", 1: "B", 2: "I"}
    text = "hello world"
    offsets = [(0, 0), (0, 5), (6, 11), (0, 0)]
    tags = [0, 1, 2, 0]
    assert _decode(text, offsets, tags, labels, None) == ["hello world"]


def test_decode_bio_empty_when_all_O():
    text = "nothing here"
    offsets = [(0, 0), (0, 7), (8, 12), (0, 0)]
    tags = [0, 0, 0, 0]
    assert _decode(text, offsets, tags, PLACE_LABELS, None) == []


# -- Optional integration test against a locally-exported ONNX model --

PLACE_MODEL_DIR = os.environ.get(
    "REASONGRAPH_PLACE_MODEL_DIR",
    "/home/berk/repos/trainner/archive/web_release",
)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(PLACE_MODEL_DIR, "config.json")),
    reason="local place-extractor ONNX model not available",
)
def test_onnx_place_extractor_runs_locally():
    pytest.importorskip("onnxruntime")
    onnx_path = os.path.join(PLACE_MODEL_DIR, "onnx", "model_fp16.onnx")
    ext = OnnxTokenClassifierExtractor(PLACE_MODEL_DIR, onnx_path=onnx_path)

    ents = ext("TSMC announced a plant in Phoenix, Arizona.")
    assert isinstance(ents, list) and all(isinstance(e, str) for e in ents)
    assert "Phoenix" in ents  # the city is a bridge entity the model must catch
