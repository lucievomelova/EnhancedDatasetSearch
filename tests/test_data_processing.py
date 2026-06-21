"""Unit tests for data processing code. All LLM and embedding model calls are replaced with mock objects and empty
response is always returned, so that the results are deterministic"""

import asyncio
import os

import pytest
import yaml
import pandas as pd
from unittest.mock import MagicMock
import shutil

from data_processing.data_catalogs.nkod import NkodDataCatalog
from data_processing.database import Database
from data_processing.knowledge_graph import create_kg, get_similar_datasets
from ollama_client import OllamaClient
from neo4j import GraphDatabase



def _sort_items_in_list_cols(list_columns: list, df1: pd.DataFrame, df2: pd.DataFrame):
    """Sorts the items in list columns of two dataframes so that they can be compared with assert_frame_equal."""
    for col in list_columns:
        df1[col] = df1[col].apply(sorted)
        df2[col] = df2[col].apply(sorted)


def _cleanup_files(file_paths: list):
    """Delete specified files."""
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)

with open("tests/test_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# copy old datasets file so that it can be used in NKOD pipeline
src = "tests/data/test_datasets_old.json"
dst = "tests/data/test_datasets.json"
shutil.copy(src, dst)


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
def data_catalog(mock_ollama_client):
    catalog = NkodDataCatalog(config)
    files_to_remove = [
        catalog._data_config["datasets_path"],
        catalog._data_config["datasets_transformed_path"]
    ]
    _cleanup_files(files_to_remove)  # remove datasets files from previous test runs in case there was an error
    catalog.client = mock_ollama_client

    # we have to load raw datasets from our file and then reload the remaining dataframes based on it
    catalog._load_datasets_raw(False)
    catalog._transform_datasets_raw(True)
    asyncio.run(catalog.update_datasets())
    yield catalog
    _cleanup_files(files_to_remove)  # remove datasets files from this test run


# ======== NKOD ========

def test_rows_merged(data_catalog):
    """Test that rows belonging to the same dataset are correctly merged into one."""
    assert len(data_catalog.datasets_raw_transformed) == 4  # there are four unique dataset URLs in the file


def test_list_columns(data_catalog):
    """Test that list columns exist, that they contain only lists and that each item of each list is not nan."""
    for col in data_catalog.list_columns:
        assert col in data_catalog.datasets_raw_transformed.columns
        assert data_catalog.datasets_raw_transformed[col].map(lambda x: isinstance(x, list)).all()
        assert data_catalog.datasets_raw_transformed[col].map(lambda x: all(pd.notna(item) for item in x)).all()


def test_correct_datasets_transformed_content(data_catalog):
    """Test that the datasets_raw_transformed dataframe contains expected data."""
    df_correct = pd.read_csv("tests/data/test_datasets_raw_transformed_correct.csv",
                             sep=",", quotechar='"',
                             converters={col: pd.eval for col in data_catalog.list_columns})
    _sort_items_in_list_cols(data_catalog.list_columns, df_correct, data_catalog.datasets_raw_transformed)
    pd.testing.assert_frame_equal(
        df_correct.reset_index(drop=True),
        data_catalog.datasets_raw_transformed.reset_index(drop=True)
    )


def test_datasets_updated(data_catalog):
    """Test that the datasets dataframe was updated from its old version after get_new_dataset() was called."""
    df_old = pd.read_json("tests/data/test_datasets_old.json", orient="split")
    _sort_items_in_list_cols(data_catalog.list_columns, df_old, data_catalog.datasets)
    # the old file contained more datasets than the new file - check that they were removed
    assert not len(df_old) == len(data_catalog.datasets)


def test_correct_datasets_content(data_catalog):
    """Test that the datasets dataframe contains expected data."""
    df_correct = pd.read_json("tests/data/test_datasets_correct.json", orient="split")
    _sort_items_in_list_cols(data_catalog.list_columns, df_correct, data_catalog.datasets)
    pd.testing.assert_frame_equal(
        df_correct.reset_index(drop=True),
        data_catalog.datasets.reset_index(drop=True)
    )


def test_get_dataset_by_url_existing_url(data_catalog):
    """Test that get_dataset_by_url() works for an existing URL."""
    url = "http://example.com/ds1"
    dataset_info = data_catalog.get_dataset_by_url(url)
    assert dataset_info is not None
    assert dataset_info["url"] == url


def test_get_dataset_by_url_nonexistent_url(data_catalog):
    """Test that get_dataset_by_url() returns None for non-existent URL."""
    url = "http://nonexistent"
    dataset_info = data_catalog.get_dataset_by_url(url)
    assert dataset_info is None


def test_metadata_filters_contain_correct_values(data_catalog):
    """Test that get_filters_with_counts() contains keywords, themes, categories and providers with positive counts."""
    filters = data_catalog.get_filters_with_counts()
    for metadata_category in ["keywords", "themes", "categories", "provider"]:
        assert metadata_category in filters
        for count in filters[metadata_category]["value_counts"].values():
            assert count > 0


def test_metadata_filters_dont_contain_data_for_missing_metadata_category(data_catalog):
    """Test that get_filters_with_counts() do not contain values for metadata categories that are not present."""
    filters = data_catalog.get_filters_with_counts()
    for metadata_category in ["spatial_coverage", "temporal_coverage"]:
        assert metadata_category in filters  # the metadata category key must be present
        assert not filters[metadata_category]["value_counts"]  # value counts must be an empty dict


# ======== Knowledge graph ========


def test_create_kg(data_catalog, mock_database):
    """Test that KG creation doesn't throw any errors."""
    create_kg(data_catalog.datasets, mock_database, config["data_processing"]["knowledge_graph"])


def test_all_datasets_in_kg(driver):
    graph_name = config["data_processing"]["knowledge_graph"]["name"]
    query = f"MATCH (d:Dataset) WHERE d.graph = $graph_name RETURN count(d) AS count"
    with driver.session() as session:
        number_of_datasets = session.run(query, graph_name=graph_name).single()["count"]
        assert number_of_datasets == 4


def test_metadata_nodes_exist_in_kg(driver):
    """Test that metadata nodes exist in KG - at least one keyword, theme, category, provider."""
    graph_name = config["data_processing"]["knowledge_graph"]["name"]
    for node_type in ["Keyword", "Theme", "Category"]:
        rel_type = f"HAS_{node_type}".upper()
        query = f"MATCH (d:Dataset)-[:{rel_type}]->(n:{node_type}) WHERE d.graph = $graph_name RETURN count(DISTINCT n) AS count"
        with driver.session() as session:
            count = session.run(query, graph_name=graph_name).single()["count"]
            assert count > 0

    # provider has a different relationship name
    query = "MATCH (d:Dataset)-[:PROVIDED_BY]->(p:Provider) WHERE d.graph = $graph_name RETURN count(DISTINCT p) AS count"
    with driver.session() as session:
        count = session.run(query, graph_name=graph_name).single()["count"]
        assert count > 0


def test_all_keyword_nodes_exist_in_kg(data_catalog, driver):
    """Test that each keyword has a keyword node in the KG and no extra keyword nodes exist."""
    graph_name = config["data_processing"]["knowledge_graph"]["name"]
    # we have to use distinct, otherwise each keyword will be added once for each dataset it is connected to
    query = "MATCH (d:Dataset)-[:HAS_KEYWORD]->(k:Keyword) WHERE d.graph = $graph_name RETURN DISTINCT k.name AS name"
    with driver.session() as session:
        result = session.run(query, graph_name=graph_name)
        keywords = [record["name"].lower() for record in result]
        assert sorted(data_catalog.all_keywords) == sorted(keywords)


def test_no_similarity_edges_exist(driver):
    """Test that no SIMILAR relationships exist in the test KG."""
    graph_name = config["data_processing"]["knowledge_graph"]["name"]
    query = "MATCH (d1:Dataset)-[rel:SIMILAR]->(d2:Dataset) WHERE d1.graph = $graph_name AND d2.graph = $graph_name RETURN count(DISTINCT rel) AS count"
    with driver.session() as session:
        count = session.run(query, graph_name=graph_name).single()["count"]
        assert count == 0


def test_get_similar_datasets(driver):
    """Test that get_similar_datasets works."""
    similar_datasets = get_similar_datasets("http://example.com/ds2",config["data_processing"]["knowledge_graph"])
    assert len(similar_datasets["themes"]) > 0  # there is a similar dataset based on themes
    assert len(similar_datasets["description"]) == 0  # description similarity is not calculated for test KG
