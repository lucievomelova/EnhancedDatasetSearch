import os

import pytest
import yaml
import pandas as pd

from data_processing.data_catalogs.nkod import NkodDataCatalog


with open("tests/test_config.yaml", "r") as f:
    config = yaml.safe_load(f)


@pytest.fixture(scope="module")
def data_catalog():
    catalog = NkodDataCatalog(config)

    # we have to load raw datasets from our file and then reload the remaining dataframes based on it
    catalog._load_datasets_raw(False)
    catalog._transform_datasets_raw(True)
    yield catalog
    if os.path.exists(catalog._data_config["datasets_path"]):
        os.remove(catalog._data_config["datasets_path"])
    if os.path.exists(catalog._data_config["datasets_transformed_path"]):
        os.remove(catalog._data_config["datasets_transformed_path"])


def test_dataset_rows_merged(data_catalog):
    """Test that rows belonging to the same dataset are correctly merged into one."""
    assert len(data_catalog.datasets_raw_transformed) == 4  # there are four unique dataset URLs in the file


def test_correct_datasets_transformed_content(data_catalog):
    """Test that the datasets_raw_transformed dataframe contains expected data."""
    df_correct = pd.read_csv("tests/data/test_datasets_raw_transformed_correct.csv",
                             sep=",", quotechar='"',
                             converters={col: pd.eval for col in data_catalog.list_columns})
    for col in data_catalog.list_columns:
        df_correct[col] = df_correct[col].apply(sorted)
        data_catalog.datasets_raw_transformed[col] = data_catalog.datasets_raw_transformed[col].apply(sorted)

    pd.testing.assert_frame_equal(
        df_correct.reset_index(drop=True),
        data_catalog.datasets_raw_transformed.reset_index(drop=True)
    )


def test_correct_datasets_content(data_catalog):
    """Test that the datasets dataframe contains expected data."""
    df_correct = pd.read_json("tests//data/test_datasets_correct.json", orient="split")
    for col in data_catalog.list_columns:
        df_correct[col] = df_correct[col].apply(sorted)
        data_catalog.datasets[col] = data_catalog.datasets[col].apply(sorted)

    pd.testing.assert_frame_equal(
        df_correct.reset_index(drop=True),
        data_catalog.datasets.reset_index(drop=True)
    )


def test_list_columns(data_catalog):
    """Test that list columns exist, that they contain only lists and that each item of each list is not nan."""
    for col in data_catalog.list_columns:
        assert col in data_catalog.datasets.columns
        assert data_catalog.datasets[col].map(lambda x: isinstance(x, list)).all()
        assert data_catalog.datasets[col].map(lambda x: all(pd.notna(item) for item in x)).all()

