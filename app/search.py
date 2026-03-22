from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.storage.docstore.postgres import PostgresDocumentStore

from utils import setup_logger
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

logger = setup_logger(__name__)


class SearchEngine:
    def __init__(self, search_config: dict, index: VectorStoreIndex, docstore: PostgresDocumentStore):
        self.search_config = search_config
        self.index = index
        self.docstore = docstore
        self.retriever = self._create_fusion_retriever()

    def _create_fusion_retriever(self) -> QueryFusionRetriever:
        vector_retriever = self.index.as_retriever(similarity_top_k=self.search_config["vector_top_k"])
        bm25_retriever = BM25Retriever.from_defaults(index=self.index, similarity_top_k=self.search_config["bm25_top_k"])

        retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=self.search_config["top_k"],
            num_queries=1,
            mode=self.search_config["mode"],
            use_async=True,
            verbose=True,
        )
        return retriever


    async def search(self, user_query: str) -> list[NodeWithScore]:
        """Search for relevant datasets."""

        logger.info(f"Searching - query: {user_query}")
        nodes = await self.retriever.aretrieve(f'{user_query}')
        node_ids = [n.id_ for n in nodes]
        nodes_without_duplicates = [n for n in nodes if n.node.ref_doc_id not in node_ids]
        logger.info(f"Retrieved {len(nodes_without_duplicates)} nodes.")
        return nodes_without_duplicates
