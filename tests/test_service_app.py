import pytest

from reasongraph.service import app as service_app
from reasongraph.backends._memory import MemoryBackend
from reasongraph.backends._sqlite import SqliteBackend
from reasongraph._synthesizers import TemplateSynthesizer


# -- backend selection --

def test_build_backend_memory_default():
    backend = service_app.build_backend({})
    assert isinstance(backend, MemoryBackend)
    assert backend.file_path is None


def test_build_backend_memory_with_file():
    backend = service_app.build_backend(
        {"REASONGRAPH_BACKEND": "memory", "REASONGRAPH_DATABASE_URL": "/tmp/mem.json"}
    )
    assert isinstance(backend, MemoryBackend)
    assert backend.file_path == "/tmp/mem.json"


def test_build_backend_sqlite():
    backend = service_app.build_backend(
        {"REASONGRAPH_BACKEND": "sqlite", "REASONGRAPH_DATABASE_URL": "/tmp/g.db"}
    )
    assert isinstance(backend, SqliteBackend)
    assert backend.db_path == "/tmp/g.db"


def test_build_backend_postgres_requires_url():
    with pytest.raises(ValueError, match="DATABASE_URL"):
        service_app.build_backend({"REASONGRAPH_BACKEND": "postgres"})


def test_build_backend_unknown():
    with pytest.raises(ValueError, match="Unknown REASONGRAPH_BACKEND"):
        service_app.build_backend({"REASONGRAPH_BACKEND": "cassandra"})


# -- embedder selection --

def test_build_embed_model_default_none():
    assert service_app.build_embed_model({}) is None


def test_build_embed_model_plain_name_passthrough():
    assert service_app.build_embed_model(
        {"REASONGRAPH_EMBED_MODEL": "all-MiniLM-L6-v2"}
    ) == "all-MiniLM-L6-v2"


# -- synthesizer selection --

def test_build_synthesizer_default_template():
    assert isinstance(service_app.build_synthesizer({}), TemplateSynthesizer)


def test_build_synthesizer_none():
    assert service_app.build_synthesizer({"REASONGRAPH_SYNTHESIZER": "none"}) is None


def test_build_synthesizer_unknown():
    with pytest.raises(ValueError, match="Unknown REASONGRAPH_SYNTHESIZER"):
        service_app.build_synthesizer({"REASONGRAPH_SYNTHESIZER": "gpt5"})


# -- misc env parsing --

def test_int_env_parsing():
    assert service_app._int_env({"K": "45"}, "K", 30) == 45
    assert service_app._int_env({}, "K", 30) == 30
    assert service_app._int_env({"K": ""}, "K", None) is None
