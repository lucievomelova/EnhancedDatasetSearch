import json
import os
from llama_index.core import Document, StorageContext
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from dotenv import load_dotenv

from utils import setup_logger


load_dotenv()
logger = setup_logger(__name__)


class Database:
    """Class for handling the RAG database - Postgres with PGVector extension."""
    def __init__(self, config: dict, state_dir: str):
        db_config = config["db"]
        embedding_config = config["embedding"]
        self.state_dir = state_dir
        self.vector_store: PGVectorStore = PGVectorStore.from_params(
            database=os.environ['POSTGRES_DB'],
            host=db_config['host'],
            password=os.environ['POSTGRES_PASSWORD'],
            port=db_config['port'],
            user=os.environ['POSTGRES_USER'],
            table_name=db_config['vector_table'],
            embed_dim=db_config['embed_dim'],
        )
        self.document_store = PostgresDocumentStore.from_params(
            database=os.environ['POSTGRES_DB'],
            host=db_config['host'],
            password=os.environ['POSTGRES_PASSWORD'],
            port=db_config['port'],
            user=os.environ['POSTGRES_USER'],
            table_name=db_config['document_table'],
        )

        self.embedding_model: BaseEmbedding = OllamaEmbedding(
            model_name=embedding_config['model_name'],
            base_url=embedding_config['base_url'],
            embed_batch_size=embedding_config['embed_batch_size'],
        )
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            docstore=self.document_store
        )
        self.index: VectorStoreIndex = VectorStoreIndex(
            nodes=[],
            embed_model=self.embedding_model,
            storage_context=self.storage_context
        )

        self.pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=embedding_config['chunk_size'],
                                 chunk_overlap=embedding_config['chunk_overlap']),
                self.embedding_model,
            ],
            vector_store=self.vector_store,
            docstore=self.document_store,
        )

        self.url_to_node_id_mapping = {}
        """Mapping from dataset url (which is the doc_id in document store) to node_id in vector store."""


    def load_documents(self, new_documents: list[Document]) -> None:
        """Load documents from data_df into the RAG database."""
        logger.info("Ingestion pipeline created. Loading %d new documents to RAG DB.", len(new_documents))
        nodes = self.pipeline.run(documents=new_documents, show_progress=True)
        for node in nodes:
            if node.ref_doc_id not in self.url_to_node_id_mapping:
                self.url_to_node_id_mapping[node.ref_doc_id] = [node.node_id]
            else:
                self.url_to_node_id_mapping[node.ref_doc_id].append(node.node_id)

        # save mapping to state
        with open(os.path.join(self.state_dir, "url_to_node_id_mapping.json"), "w") as f:
            json.dump(self.url_to_node_id_mapping, f)

    def get_similar_datasets_by_embedding(self, dataset_url: str, similarity_threshold: int = 0.8, k: int = 10) -> dict[str, float]:
        """Get datasets similar to the given dataset based on their embedding similarity.

        Returns:
            a dict, where keys are similar dataset urls and values are max similarity scores."""

        node_ids = self.url_to_node_id_mapping.get(dataset_url, [])
        nodes = self.vector_store.get_nodes(node_ids=node_ids)
        similar_nodes = {}

        # retrieve k + len(node_ids) to make sure we retrieve at least k chunks corresponding to other datasets
        vector_retriever = self.index.as_retriever(similarity_top_k=k+len(node_ids))
        for node in nodes:
            text = node.get_content()
            result = vector_retriever.retrieve(text)
            for r in result:
                if r.node_id not in node_ids:  # exclude chunks from the same dataset
                    if r.score < similarity_threshold:
                        continue
                    if r.node.ref_doc_id not in similar_nodes:
                        similar_nodes[r.node.ref_doc_id] = r.score
                    else:
                        similar_nodes[r.node.ref_doc_id] = max(similar_nodes[r.node.ref_doc_id], r.score)

        # get k nodes with max score
        sorted_similar_nodes = {ref_doc_id: similar_nodes[ref_doc_id] for ref_doc_id in sorted(similar_nodes, key=similar_nodes.get, reverse=True)}
        if len(sorted_similar_nodes) > k:
            sorted_similar_nodes = dict(list(sorted_similar_nodes.items())[:k])  # keep only top k
        return sorted_similar_nodes

