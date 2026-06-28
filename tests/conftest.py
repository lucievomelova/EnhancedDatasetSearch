
import asyncio
import os
from unittest.mock import MagicMock

import psycopg2
import pytest
import yaml
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from neo4j import GraphDatabase

from EnhancedDatasetSearch.app.pipeline import SearchPipeline
from EnhancedDatasetSearch.app.query_prepocessing import QueryPreprocessor
from EnhancedDatasetSearch.data_processing.database import Database
from EnhancedDatasetSearch.data_processing.NKOD.knowledge_graph import NkodKnowledgeGraph
from EnhancedDatasetSearch.data_processing.NKOD.nkod import NkodDataCatalog
from EnhancedDatasetSearch.ollama_client import OllamaClient


@pytest.fixture(scope="module")
def config():
    with open("tests/test_config.yaml", "r") as f:
        config = yaml.safe_load(f)
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
def data_catalog(config, mock_ollama_client):
    catalog = NkodDataCatalog(config, True)
    files_to_remove = [
        catalog._data_config["datasets_path"],
        catalog._data_config["datasets_transformed_path"]
    ]
    _cleanup_files(files_to_remove)  # remove datasets files from previous test runs in case there was an error
    catalog.client = mock_ollama_client

    # we have to load raw datasets from our file and then reload the remaining dataframes based on it
    # catalog._load_datasets_raw(True)
    # catalog._transform_datasets_raw(True)
    asyncio.run(catalog.update_datasets())
    yield catalog
    _cleanup_files(files_to_remove)  # remove datasets files from this test run


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
    llm = Ollama(model=config['llm']['model_name'], context_window=config['llm']['context_length'])
    search_pipeline = SearchPipeline(config, llm, data_catalog, database)
    return search_pipeline


@pytest.fixture(scope="module")
def query_preprocessor(config):
    query_preprocessor = QueryPreprocessor(config)
    return query_preprocessor
