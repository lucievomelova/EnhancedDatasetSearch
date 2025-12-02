import ollama
import pandas as pd
from utils import setup_logger
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

logger = setup_logger(__name__)


def _create_fusion_retriever(index) -> QueryFusionRetriever:
    vector_retriever = index.as_retriever(similarity_top_k=10)
    bm25_retriever = BM25Retriever.from_defaults(docstore=index.docstore, similarity_top_k=10)

    retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=10,
        num_queries=1,  # set this to 1 to disable query generation
        mode="reciprocal_rerank",
        use_async=True,
        verbose=True,
    )
    return retriever


def search(index, expanded_query: list, data_df: pd.DataFrame, k: int = 100) -> pd.DataFrame | None:
    """Search the data for the given query."""

    system_query_intro = """
    You are a helpful AI assistant for a dataset catalog search engine. The user typed in a search query, which was 
    expanded into multiple related queries to improve search results:
    """

    system_query_task = f"""
    Your task is to search for relevant datasets based on the expanded queries. There is a RAG database containing 
    information about all datasets. For each expanded query, search the RAG database and retrieve text chunks that match 
    the expanded search queries. Return {k} most relevant results."""

    logger.info("Searching...")

    retriever = index.as_retriever(similarity_top_k=10)
    nodes = retriever.retrieve(f'{system_query_intro} + {expanded_query} + {system_query_task}')
    node = nodes[0]
    logger.info(f"Retrieved chunks: {[node.metadata for node in nodes]}")
