from pathlib import Path

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.storage.docstore.postgres import PostgresDocumentStore

from EnhancedDatasetSearch.data_processing.database import Database
from EnhancedDatasetSearch.utils import setup_logger

logger = setup_logger(__name__)


class Retriever:
    """Retriever - searches the document and vector store based on the provided search query.

    For retrieval, QueryFusionRetriever is used. It combines a VectorIndexRetriever and BM25Retriever."""
    def __init__(self, search_config: dict, database: Database, bm25_persist_dir: str):
        self.search_config: dict = search_config
        self.index: VectorStoreIndex = database.index
        self.document_store: PostgresDocumentStore = database.document_store
        self.bm25_persist_dir: Path = Path(bm25_persist_dir)
        self.retriever: QueryFusionRetriever = self._create_retriever()

    def _create_retriever(self) -> QueryFusionRetriever:
        """Create fusion retriever that combines vector search and BM25 search."""
        vector_retriever = self.index.as_retriever(
            similarity_top_k=self.search_config["vector_top_k"]
        )

        if self.bm25_persist_dir.exists():
            bm25_retriever = BM25Retriever.from_persist_dir(str(self.bm25_persist_dir))
        else:
            bm25_retriever = BM25Retriever.from_defaults(
                docstore=self.document_store,
                similarity_top_k=self.search_config["bm25_top_k"]
            )
            bm25_retriever.persist(str(self.bm25_persist_dir))

        retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=self.search_config["top_k"],
            num_queries=1,
            mode=self.search_config["mode"],
            use_async=True,
            verbose=True,
            retriever_weights=self.search_config["retriever_weights"],
        )
        return retriever

    async def run(self, user_query: str, extended_query: str | None, applied_filters: dict | None) -> list[NodeWithScore]:
        """Search for relevant datasets."""
        query = user_query
        if extended_query:
            query += ", " + extended_query
        if applied_filters:
            filter_list = [item for lst in applied_filters.values() for item in lst]
            query += f", {".".join(filter_list)}"
        logger.info(f"Searching - query: {query}")
        nodes = await self.retriever.aretrieve(query)
        node_ids = [n.id_ for n in nodes]
        logger.info(f"Retrieved {len(node_ids)} nodes.")

        # merge scores for duplicated results -  sum, because a node is more important if it was returned by both retrievers
        duplicates = [n for n in nodes if n.node.ref_doc_id in node_ids]
        for duplicate_node in duplicates:
            node = nodes[node_ids.index(duplicate_node.node.ref_doc_id)]
            node.score += duplicate_node.score
        nodes_without_duplicates = [n for n in nodes if n.node.ref_doc_id not in node_ids]

        logger.info(f"Removed duplicates, {len(nodes_without_duplicates)} nodes left.")
        return nodes_without_duplicates
