import os
from llama_index.core import Document
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
    def __init__(self, config: dict):
        db_config = config["db"]
        embedding_config = config["embedding"]
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
        self.index: VectorStoreIndex = VectorStoreIndex.from_vector_store(self.vector_store, self.embedding_model)

        self.pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=embedding_config['chunk_size'],
                                 chunk_overlap=embedding_config['chunk_overlap']),
                self.embedding_model,
            ],
            vector_store=self.vector_store,
            docstore=self.document_store,
        )

    def load_documents(self, new_documents: list[Document]) -> None:
        """Load documents from data_df into the RAG database."""
        logger.info("Ingestion pipeline created. Loading %d new documents to RAG DB.", len(new_documents))
        self.pipeline.run(documents=new_documents, show_progress=True)
