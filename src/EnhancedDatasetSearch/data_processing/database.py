import json
import os
from datetime import datetime

import psycopg2
from dotenv import load_dotenv
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import QueryBundle
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.vector_stores.postgres import PGVectorStore

from EnhancedDatasetSearch.utils import setup_logger

load_dotenv()
logger = setup_logger(__name__)


class Database:
    """Class for handling the database - Postgres with PGVector extension."""
    def __init__(self, config: dict):
        self.db_config = config["db"]
        self.embedding_config = config["embedding"]
        self.state_dir = config["state_dir"]

        self.vector_store: PGVectorStore
        """Table storing embedding vectors of datasets' documents."""

        self.document_store: PostgresDocumentStore
        """Table storing datasets' documents."""

        self.embedding_model: BaseEmbedding = OllamaEmbedding(
            model_name=self.embedding_config['model_name'],
            base_url=self.embedding_config['base_url'],
            embed_batch_size=self.embedding_config['embed_batch_size'],
        )
        self.storage_context: StorageContext
        self.index: VectorStoreIndex
        self.pipeline: IngestionPipeline
        """Ingestion pipeline - for loading documents into the database."""

        self._setup_vector_index()

        self.url_to_node_id_mapping_file = os.path.join(self.state_dir, config["embedding"]["url_to_node_id_mapping_file"])
        self.url_to_node_id_mapping = {}
        """Mapping from dataset url (which is the doc_id in document store) to node_id in vector store. This is used
        for getting the dataset embedding fast when creating similarity edges in the knowledge graph."""

    def _setup_vector_index(self):
        """Setup vector store, document store, storage context and vector store index."""
        self.vector_store = PGVectorStore.from_params(
            database=os.environ['POSTGRES_DB'],
            host=self.db_config['host'],
            password=os.environ['POSTGRES_PASSWORD'],
            port=self.db_config['port'],
            user=os.environ['POSTGRES_USER'],
            table_name=self.db_config['vector_table'],
            embed_dim=self.db_config['embed_dim'],
        )

        self.document_store = PostgresDocumentStore.from_params(
            database=os.environ['POSTGRES_DB'],
            host=self.db_config['host'],
            password=os.environ['POSTGRES_PASSWORD'],
            port=self.db_config['port'],
            user=os.environ['POSTGRES_USER'],
            table_name=self.db_config['document_table'],
        )

        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            docstore=self.document_store
        )
        self.index = VectorStoreIndex(
            nodes=[],
            embed_model=self.embedding_model,
            storage_context=self.storage_context
        )

    def _setup_ingestion_pipeline(self, vector_store: PGVectorStore, document_store: PostgresDocumentStore) -> None:
        """Set up the ingestion pipeline with the new vector and document stores."""
        self.pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=self.embedding_config['chunk_size'],
                                 chunk_overlap=self.embedding_config['chunk_overlap']),
                self.embedding_model,
            ],
            vector_store=vector_store,
            docstore=document_store,
        )

    def _create_staging_vector_and_doc_store(self, staging_suffix : str) -> (PGVectorStore, PostgresDocumentStore):
        """Creating staging vector store and document store.

         Staging stores will be used for loading new documents and once everything is ready, they will be swapped
         to live. This way, there is no downtime for users."""
        vector_store_staging: PGVectorStore = PGVectorStore.from_params(
            database=os.environ['POSTGRES_DB'],
            host=self.db_config['host'],
            password=os.environ['POSTGRES_PASSWORD'],
            port=self.db_config['port'],
            user=os.environ['POSTGRES_USER'],
            table_name=self.db_config['vector_table'] + staging_suffix,
            embed_dim=self.db_config['embed_dim'],
        )

        document_store_staging = PostgresDocumentStore.from_params(
            database=os.environ['POSTGRES_DB'],
            host=self.db_config['host'],
            password=os.environ['POSTGRES_PASSWORD'],
            port=self.db_config['port'],
            user=os.environ['POSTGRES_USER'],
            table_name=self.db_config['document_table'] + staging_suffix,
        )
        return vector_store_staging, document_store_staging

    def _swap_staging_tables_to_live(self, staging_suffix: str):
        """Rename current live tables to old, then rename staging tables to live."""
        conn = psycopg2.connect(
            database=os.environ['POSTGRES_DB'],
            host=self.db_config['host'],
            password=os.environ['POSTGRES_PASSWORD'],
            port=self.db_config['port'],
            user=os.environ['POSTGRES_USER']
        )

        # real vector and document table names - with data_ prefix and suffixes
        vector_table = "data_" + self.db_config['vector_table']
        doc_table = "data_" + self.db_config['document_table']
        vector_table_old = "data_" + self.db_config['vector_table'] + "_old"
        doc_table_old = "data_" + self.db_config['document_table'] + "_old"
        staging_vector_table = "data_" + self.db_config['vector_table'] + staging_suffix
        staging_document_table = "data_" + self.db_config['document_table'] + staging_suffix

        conn.autocommit = False
        try:
            with conn.cursor() as cur:
                # drop existing old tables
                cur.execute(f'DROP TABLE IF EXISTS "{vector_table_old}" CASCADE;')
                cur.execute(f'DROP TABLE IF EXISTS "{doc_table_old}" CASCADE;')
                # rename current live -> old
                cur.execute(f"""
                    DO $$
                    BEGIN
                        IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{vector_table}') THEN
                            ALTER TABLE "{vector_table}" RENAME TO "{vector_table_old}";
                        END IF;
                        IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{doc_table}') THEN
                            ALTER TABLE "{doc_table}" RENAME TO "{doc_table_old}";
                        END IF;
                    END $$;
                """)

                # rename staging -> live
                cur.execute(f'ALTER TABLE "{staging_vector_table}" RENAME TO "{vector_table}";')
                cur.execute(f'ALTER TABLE "{staging_document_table}" RENAME TO "{doc_table}";')

            conn.commit()
            logger.info("Staging tables swapped to live.")
        except Exception:
            conn.rollback()
            logger.exception("Swap failed, rolled back.")
            raise

    def load_documents(self, new_documents: list[Document]) -> None:
        """Load datasets' documents from data_catalog.datasets into the database.

        Documents are loaded into the staging tables. Once loading is complete, we rename the live tables to old
        and then the staging tables are renamed and used as new live tables."""
        logger.info("Ingestion pipeline created. Loading %d new documents to RAG DB.", len(new_documents))
        time = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        staging_suffix = f"_{time}"
        vector_store, document_store = self._create_staging_vector_and_doc_store(staging_suffix)
        self._setup_ingestion_pipeline(vector_store, document_store)

        nodes = self.pipeline.run(documents=new_documents, show_progress=True, num_workers=8)
        self.url_to_node_id_mapping = {}
        for node in nodes:
            if node.ref_doc_id not in self.url_to_node_id_mapping:
                self.url_to_node_id_mapping[node.ref_doc_id] = [node.node_id]
            else:
                self.url_to_node_id_mapping[node.ref_doc_id].append(node.node_id)

        # save mapping to state
        with open(self.url_to_node_id_mapping_file, "w") as f:
            logger.info("Storing url_to_node_mapping to state.")
            json.dump(self.url_to_node_id_mapping, f)

        self._swap_staging_tables_to_live(staging_suffix)
        logger.info("Loading complete.")

    def get_similar_datasets_by_embedding(self, dataset_url: str, similarity_threshold: int, k: int) -> dict[str, float]:
        """Get datasets similar to the given dataset based on their embedding similarity.

        Returns:
            a dict, where keys are similar dataset urls and values are max similarity scores."""
        if self.url_to_node_id_mapping == {}:
            logger.info("Loading url_to_node_mapping from state.")
            with open(self.url_to_node_id_mapping_file, "r") as f:
                self.url_to_node_id_mapping = json.load(f)
        node_ids = self.url_to_node_id_mapping.get(dataset_url, [])
        nodes = self.vector_store.get_nodes(node_ids=node_ids)
        similar_nodes = {}

        # retrieve k + len(node_ids) to make sure we retrieve at least k chunks corresponding to other datasets
        vector_retriever = self.index.as_retriever(similarity_top_k=k+len(node_ids))
        for node in nodes:
            # use QueryBundle for retrieval - we can pass already computed embedding instead of recomputing it
            result = vector_retriever.retrieve(QueryBundle(embedding=node.embedding, query_str=node.get_content()))
            for r in result:
                if r.node_id not in node_ids:  # exclude chunks from the same dataset
                    if r.score < similarity_threshold:
                        continue
                    # to avoid processing each pair twice, skip nodes with smaller url than current (lexicographically)
                    if r.node.ref_doc_id < dataset_url:
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
