from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.storage.docstore.postgres import PostgresDocumentStore

from utils import setup_logger
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter


logger = setup_logger(__name__)


class Retriever:
    def __init__(self, search_config: dict, index: VectorStoreIndex, docstore: PostgresDocumentStore):
        self.search_config = search_config
        self.index = index
        self.docstore = docstore
        self.retriever = self._create_retriever()

    def _create_retriever(self, filters: MetadataFilters | None = None) -> QueryFusionRetriever:
        """Create fusion retriever that combines vector search and BM25 search."""
        vector_retriever = self.index.as_retriever(similarity_top_k=self.search_config["vector_top_k"],
                                                   filters=filters)

        bm25_retriever = BM25Retriever.from_defaults(docstore=self.docstore,
                                                     similarity_top_k=self.search_config["bm25_top_k"],
                                                     filters=filters)
        retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=self.search_config["top_k"],
            num_queries=1,
            mode=self.search_config["mode"],
            use_async=True,
            verbose=True,
            retriever_weights=self.search_config["retriever_weights"],
        )

        # we need to force the retriever to be initialized, otherwise the first user request will be slow, because
        # by default bm25 uses lazy initialization, so it would be initialized only when the first search is performed
        # retriever.retrieve("warmup")

        return retriever

    async def run(self, user_query: str, extended_query: str | None = None, filters: dict | None = None) -> list[NodeWithScore]:
        """Search for relevant datasets."""

        logger.info(f"Searching - query: {user_query} + extended query: {extended_query}")
        nodes = await self.retriever.aretrieve(f'{user_query}, {extended_query}')
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
