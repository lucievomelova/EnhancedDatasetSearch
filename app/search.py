from typing import List, Dict

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.storage.docstore.postgres import PostgresDocumentStore

from utils import setup_logger
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

logger = setup_logger(__name__)


class SearchEngine:
    def __init__(self, index: VectorStoreIndex, docstore: PostgresDocumentStore):
        self.index = index
        self.docstore = docstore
        self.retriever = self._create_fusion_retriever()

    def _create_fusion_retriever(self) -> QueryFusionRetriever:
        vector_retriever = self.index.as_retriever(similarity_top_k=10)
        bm25_retriever = BM25Retriever.from_defaults(docstore=self.docstore, similarity_top_k=10)

        retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=20,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
        )
        return retriever

    async def _retrieve(self, query: str) -> List[Dict[str, str]]:
        """Retrieve relevant chunks from the RAG database for the given query."""
        nodes = await self.retriever.aretrieve(f'{query}')

        formatted_nodes = []
        for node in nodes:
            text = self._remove_title_from_text(node.text)
            formatted_node = {
                "title": node.metadata["title"],
                "url": node.metadata["url"],
                "text": text,
            }
            formatted_nodes.append(formatted_node)
        # logger.info("Retrieved chunks:\n")
        # for item in formatted_nodes:
        #     logger.info(f"{item["title"]} - {item["url"]}:\n{item["text"]}\n")
        return formatted_nodes

    async def _search_all_queries(self, user_query: str, alternative_queries: list, k: int = 10):
        """Search the data using all queries - the original and the alternative."""

        query = f"""
            You are an AI assistant for a dataset catalog search engine. The user typed in a search query:
            {user_query}.

            This query was expanded into multiple related queries to improve search results: {alternative_queries}

            Your task is to search for relevant datasets based on the original and the alternative queries. 
            There is a RAG database containing information about all datasets. For each expanded query, 
            search the RAG database and retrieve text chunks that match the expanded search queries."""

        logger.info("Searching - all queries used.")
        # return await self._retrieve(query)
        return await self.retriever.aretrieve(f'{query}')

    async def _search_one_query(self, query: str) -> list[NodeWithScore]:
        """Search the data using a single query."""
        logger.info(f"Searching - query: {query}")
        nodes = await self.retriever.aretrieve(f'{query}')
        return nodes

    async def search(self, user_query: str, alternative_queries: list = None) -> Dict[str, list[NodeWithScore]]:
        """Search for relevant datasets."""

        logger.info(f"Searching - query: {user_query}")
        nodes = await self.retriever.aretrieve(f'{user_query}')
        return nodes

        # results = {user_query: await self._search_all_queries(user_query, alternative_queries)}
        # for alt_query in alternative_queries:
        #     results[alt_query] = await self._search_one_query(alt_query)
        # return results
