import asyncio
import os

import pytest
import yaml
import pandas as pd
from unittest.mock import MagicMock
import shutil

from data_processing.data_catalogs.nkod import NkodDataCatalog
from ollama_client import OllamaClient

with open("tests/test_config.yaml", "r") as f:
    config = yaml.safe_load(f)


def _sort_items_in_list_cols(list_columns: list, df1: pd.DataFrame, df2: pd.DataFrame):
    for col in list_columns:
        df1[col] = df1[col].apply(sorted)
        df2[col] = df2[col].apply(sorted)


def _cleanup_files(file_paths: list):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)


def copy_old_datasets_file():
    """Copy old datasets file so that it can be used in NKOD pipeline."""
    src = "tests/data/test_datasets_old.json"
    dst = "tests/data/test_datasets.json"
    shutil.copy(src, dst)


copy_old_datasets_file()


@pytest.fixture(scope="module")
def mock_ollama_client():
    mock = MagicMock(spec=OllamaClient)
    mock.get_llm_response.return_value = ""
    mock.get_llm_json_response.return_value = ({}, 0)
    return mock


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
    asyncio.run(catalog.get_new_datasets())
    yield catalog
    _cleanup_files(files_to_remove)  # remove datasets files from this test run


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
