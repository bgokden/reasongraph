import pytest

from reasongraph.datasets import load_dataset, AVAILABLE_DATASETS


def test_available_datasets():
    assert "syllogisms" in AVAILABLE_DATASETS
    assert "causal" in AVAILABLE_DATASETS
    assert "taxonomy" in AVAILABLE_DATASETS
    assert "financial" in AVAILABLE_DATASETS


@pytest.mark.parametrize("name", AVAILABLE_DATASETS)
def test_load_dataset_structure(name):
    data = load_dataset(name)
    assert "name" in data
    assert "nodes" in data
    assert "edges" in data
    assert data["name"] == name
    assert len(data["nodes"]) > 0
    assert len(data["edges"]) > 0


@pytest.mark.parametrize("name", AVAILABLE_DATASETS)
def test_dataset_node_types(name):
    data = load_dataset(name)
    for node in data["nodes"]:
        assert "content" in node
        assert "type" in node
        assert node["type"] in ("text", "entity")


@pytest.mark.parametrize("name", AVAILABLE_DATASETS)
def test_dataset_edge_references(name):
    data = load_dataset(name)
    node_contents = {n["content"] for n in data["nodes"]}
    for edge in data["edges"]:
        assert "from" in edge
        assert "to" in edge
        assert edge["from"] in node_contents, f"Edge 'from' references missing node: {edge['from']}"
        assert edge["to"] in node_contents, f"Edge 'to' references missing node: {edge['to']}"


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("nonexistent")
