from langchain_reasongraph import (ReasonGraphChatMessageHistory, ReasonGraphMemory,
                                   ReasonGraphRetriever, memory_tools, with_memory)


def test_public_names():
    assert ReasonGraphRetriever.model_fields["k"].default == 4
    assert callable(with_memory) and callable(memory_tools)
    assert ReasonGraphMemory and ReasonGraphChatMessageHistory
