"""Unit tests for the search pipeline. Here we use real LLM calls to test that they do not introduce errors. But
tha tests do not rely on the results of these calls in any way to ensure that the test results are deterministic."""
import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from EnhancedDatasetSearch.search_platform.search_pipeline.result_postprocessing import PostProcessor
from EnhancedDatasetSearch.search_platform.search_pipeline.retrieve import Retriever

pytest_plugins = ('pytest_asyncio',)


@pytest.mark.asyncio(loop_scope="session")
async def test_pipeline(search_pipeline):
    """Test that search_pipeline.run() doesn't throw any errors and returns a list."""
    results = await search_pipeline.run("test query")
    assert isinstance(results, list)


@pytest.mark.asyncio(loop_scope="session")
async def test_search_pipeline_second_run(search_pipeline):
    """Test that running search pipeline multiple times is possible."""
    await search_pipeline.run("different query")


def test_extend_user_query(query_preprocessor):
    """Test that extend_user_query doesn't throw any errors and returns a string."""
    query = "test query"
    empty_intent = {
        "categories": [],
        "spatial_coverage": [],
        "temporal_coverage": []
    }
    extended_query = query_preprocessor.extend_user_query(query, applied_filters=None, intent=empty_intent)
    assert isinstance(extended_query, str)


def test_detect_user_intent(config, query_preprocessor):
    """Test that detect_user_intent doesn't throw any errors and returns a string."""
    query = "test query"
    intent = query_preprocessor.detect_user_intent(
        query,
        applied_filters=None,
        categories=config["data_processing"]["categories"],
        other_category=config["data_processing"]["other_category"]
    )
    assert isinstance(intent, dict)
    for intent_category in ["categories", "spatial_coverage", "temporal_coverage"]:
        assert intent_category in intent and isinstance(intent[intent_category], list)


@pytest.mark.asyncio(loop_scope="session")
async def test_retriever(config, database):
    """Test that retriever doesn't throw any errors and returns a list with no duplicates."""
    bm25_dir = config["state_dir"] + "/" + config["search_platform"]["retriever"]["bm25_retriever_persist_dir"]
    retriever = Retriever(config["search_platform"]["retriever"], database, bm25_dir)
    results = await retriever.run(user_query="Desc", extended_query="", applied_filters=None)
    assert isinstance(results, list)

    # test also that the results do not contain any duplicates by comparing id_ and ref_doc_id of each node
    for node_i in results:
        for node_j in results:
            assert node_i.id_ != node_j.node.ref_doc_id


def test_reranking(config, data_catalog):
    """Test that reranking returns a list sorted by score and that  nodes with low score are removed."""
    # create test nodes with scores 0, 0.1, ..., 0.9
    example_search_results = [NodeWithScore(node=TextNode(text=f"node{i}"), score=i/10) for i in range(10)]
    postprocessor = PostProcessor(config["search_platform"]["postprocessing"], data_catalog)
    reranked_results = postprocessor.rerank(
        user_query="test query",
        results=example_search_results
    )
    assert isinstance(reranked_results, list)
    for i in range(len(reranked_results)):
        for j in range(i, len(reranked_results)):
            assert reranked_results[i].score >= reranked_results[j].score

    # cutoff score is 0.2 and nodes have scores 0, 0.1, ..., 0.9, so two scores are below threshold - 8 should remain
    assert len(reranked_results) == 8
