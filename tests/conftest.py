
import asyncio
import os
from unittest.mock import MagicMock
import shutil

import psycopg2
import pytest
import yaml
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from neo4j import GraphDatabase

from EnhancedDatasetSearch.data_processing.NKOD.nkod_dataset_processing_pipeline import NkodDatasetProcessingPipeline
from EnhancedDatasetSearch.data_processing.NKOD.nkod_documents import NkodDocumentConverter
from EnhancedDatasetSearch.data_processing.database import Database
from EnhancedDatasetSearch.search_platform.search_pipeline.pipeline import SearchPipeline
from EnhancedDatasetSearch.search_platform.search_pipeline.query_prepocessing import QueryPreprocessor
from EnhancedDatasetSearch.data_processing.NKOD.nkod_knowledge_graph import NkodKnowledgeGraph
from EnhancedDatasetSearch.search_platform.nkod_data_catalog import NkodDataCatalog
from EnhancedDatasetSearch.ollama_client import OllamaClient


# copy old datasets file so that it can be used in dataset processing pipeline
src_file = "tests/data/test_datasets_old.json"
dst_file = "tests/data/test_datasets.json"
shutil.copy(src_file, dst_file)

# copy datasets_raw file so that it can be used in dataset processing pipeline and its modification date will be today
src_file = "tests/data/test_datasets_raw_copy.csv"
dst_file = "tests/data/test_datasets_raw.csv"
shutil.copy(src_file, dst_file)


@pytest.fixture(scope="module")
def config():
    with open("tests/test_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    if not os.path.exists(config["state_dir"]):  # create state dir
        os.makedirs(config["state_dir"])
    return config


def _cleanup_files(file_paths: list):
    """Delete specified files."""
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)


@pytest.fixture(scope="module")
def mock_ollama_client():
    mock = MagicMock(spec=OllamaClient)
    mock.get_llm_response.return_value = ""
    mock.get_llm_json_response.return_value = ({}, 0)
    return mock


@pytest.fixture(scope="module")
def mock_database():
    mock = MagicMock(spec=Database)
    mock.get_similar_datasets_by_embedding.return_value = {}
    return mock


@pytest.fixture(scope="module")
def driver():
    return GraphDatabase.driver(os.environ['NEO4J_URI'], auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD']))


@pytest.fixture(scope="module")
def knowledge_graph(config, mock_database):
    knowledge_graph = NkodKnowledgeGraph(config["data_processing"]["knowledge_graph"], mock_database)
    return knowledge_graph


@pytest.fixture(scope="module")
def dataset_processing_pipeline(config, mock_ollama_client):

    # delete distributions.json before data processing pipeline is created
    files_to_remove = [config["data"]["distributions"]["path"]]
    _cleanup_files(files_to_remove)

    dataset_processing_pipeline = NkodDatasetProcessingPipeline(config)
    files_to_remove = [
        config["data"]["datasets"]["path"],
        config["data"]["datasets_raw"]["path"],
    ]
    _cleanup_files(files_to_remove)  # remove datasets files from previous test runs in case there was an error
    dataset_processing_pipeline.client = mock_ollama_client

    asyncio.run(dataset_processing_pipeline.update_datasets())
    yield dataset_processing_pipeline

    files_to_remove = [config["data"]["datasets_raw"]["path"]]
    _cleanup_files(files_to_remove)  # remove transformed datasets file from this test run


@pytest.fixture(scope="module")
def data_catalog(config, knowledge_graph):
    # load the dataset processing pipeline before loading data catalog
    dataset_processing_pipeline = NkodDatasetProcessingPipeline(config)
    catalog = NkodDataCatalog(config, knowledge_graph)
    return catalog


@pytest.fixture(scope="module")
def document_converter():
    document_converter = NkodDocumentConverter()
    return document_converter


@pytest.fixture(scope="module")
def database(config):
    database = Database(config)
    return database


@pytest.fixture(scope="module")
def connection(config):
    load_dotenv()
    conn = psycopg2.connect(
        database=os.environ["POSTGRES_DB"],
        host=config["db"]["host"],
        password=os.environ["POSTGRES_PASSWORD"],
        port=config["db"]["port"],
        user=os.environ["POSTGRES_USER"]
    )
    return conn


@pytest.fixture(scope="module")
def search_pipeline(config, data_catalog, database):
    llm = Ollama(model=config["search_platform"]["llm"]["model_name"], context_window=config["search_platform"]["llm"]["context_length"])
    search_pipeline = SearchPipeline(config, llm, data_catalog, database)
    return search_pipeline


@pytest.fixture(scope="module")
def query_preprocessor(config):
    query_preprocessor = QueryPreprocessor(config)
    return query_preprocessor
