import os
import pandas as pd
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.embeddings.ollama import OllamaEmbedding

from utils import setup_logger
from nkod_datasets import create_document_from_row

logger = setup_logger(__name__)


def create_ingestion_pipeline(config: dict) -> IngestionPipeline:
    """Create an ingestion pipeline to process and store documents in the rag store."""
    db_config = config['db']
    embedding_config = config['embedding']
    vector_store = PGVectorStore.from_params(
        database=os.environ['POSTGRES_DB'],
        host=db_config['host'],
        password=os.environ['POSTGRES_PASSWORD'],
        port=db_config['port'],
        user=os.environ['POSTGRES_USER'],
        table_name=db_config['vector_table'],
        embed_dim=db_config['embed_dim'],
    )
    document_store = PostgresDocumentStore.from_params(
        database=os.environ['POSTGRES_DB'],
        host=db_config['host'],
        password=os.environ['POSTGRES_PASSWORD'],
        port=db_config['port'],
        user=os.environ['POSTGRES_USER'],
        table_name=db_config['document_table'],
    )

    ollama_embedding = OllamaEmbedding(
        model_name=embedding_config['model_name'],
        base_url=embedding_config['base_url'],
    )

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=embedding_config['chunk_size'],
                             chunk_overlap=embedding_config['chunk_overlap']),
            ollama_embedding,
        ],
        vector_store=vector_store,
        docstore=document_store,
    )
    return pipeline


def load_documents_to_rag_db(config: dict, data_df: pd.DataFrame) -> None:
    """Load documents from data_df into the RAG database."""

    documents = [create_document_from_row(row) for _, row in data_df.iterrows()]
    pipeline = create_ingestion_pipeline(config)
    pipeline.run(documents=documents)
