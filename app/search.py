from typing import List, Dict

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.storage.docstore.postgres import PostgresDocumentStore

from custom_ollama_embedding import CustomOllamaEmbedding
from utils import setup_logger
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.llms.ollama import Ollama

logger = setup_logger(__name__)

def _create_fusion_retriever(index: VectorStoreIndex, docstore: PostgresDocumentStore) -> QueryFusionRetriever:
    vector_retriever = index.as_retriever(similarity_top_k=10)
    bm25_retriever = BM25Retriever.from_defaults(docstore=docstore, similarity_top_k=10)

    retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=10,
        num_queries=1,  # set this to 1 to disable query generation
        mode="reciprocal_rerank",
        use_async=True,
        verbose=True,
    )
    return retriever


def search(index: VectorStoreIndex, docstore: PostgresDocumentStore, expanded_query: list, k: int = 100) -> List[Dict]:
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

    retriever = _create_fusion_retriever(index, docstore)
    nodes = retriever.retrieve(f'{system_query_intro}{expanded_query}{system_query_task}')


    formatted_nodes = []
    for node in nodes:
        text = _remove_title_from_text(node.text)
        formatted_node = {
            "title": node.metadata["title"],
            "url": node.metadata["url"],
            "text": text,
        }
        formatted_nodes.append(formatted_node)
    logger.info("Retrieved chunks:\n")
    for item in formatted_nodes:
        logger.info(f"{item["title"]} - {item["url"]}:\n{item["text"]}\n")
    return formatted_nodes


def _remove_title_from_text(text: str) -> str:
    """Remove dataset title from the text chunk."""
    without_title = text.split('\n')[1:]  # title is on the first line
    joined_string = '\n'.join(without_title)  # join the split string back into one
    return joined_string