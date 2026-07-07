"""Unit tests for dataset processing. All LLM and embedding model calls are replaced with mock
objects and empty response is always returned, so that the results are deterministic."""
import os

import pandas as pd


def _sort_items_in_list_cols(list_columns: list, df1: pd.DataFrame, df2: pd.DataFrame):
    """Sorts the items in list columns of two dataframes so that they can be compared with assert_frame_equal."""
    for col in list_columns:
        df1[col] = df1[col].apply(sorted)
        df2[col] = df2[col].apply(sorted)


def test_dataframes_exist(dataset_processing_pipeline):
    """Test that all needed dataframes are not None."""
    assert dataset_processing_pipeline.datasets_raw is not None
    assert dataset_processing_pipeline.datasets_raw_transformed is not None
    assert dataset_processing_pipeline.datasets is not None


def test_rows_merged(dataset_processing_pipeline):
    """Test that rows belonging to the same dataset are correctly merged into one."""
    assert len(dataset_processing_pipeline.datasets_raw_transformed) == 4  # there are four unique dataset URLs in the file


def test_list_columns(dataset_processing_pipeline):
    """Test that list columns exist, that they contain only lists and that each item of each list is not nan."""
    for col in dataset_processing_pipeline.list_columns:
        assert col in dataset_processing_pipeline.datasets_raw_transformed.columns
        assert dataset_processing_pipeline.datasets_raw_transformed[col].map(lambda x: isinstance(x, list)).all()
        assert dataset_processing_pipeline.datasets_raw_transformed[col].map(lambda x: all(pd.notna(item) for item in x)).all()


def test_correct_datasets_transformed_content(dataset_processing_pipeline):
    """Test that the datasets_raw_transformed dataframe contains expected data."""
    df_correct = pd.read_csv("tests/data/test_datasets_raw_transformed_correct.csv",
                             sep=",", quotechar='"',
                             converters={col: pd.eval for col in dataset_processing_pipeline.list_columns})
    _sort_items_in_list_cols(
        dataset_processing_pipeline.list_columns,
        df_correct,
        dataset_processing_pipeline.datasets_raw_transformed
    )
    pd.testing.assert_frame_equal(
        df_correct.reset_index(drop=True),
        dataset_processing_pipeline.datasets_raw_transformed.reset_index(drop=True)
    )


def test_datasets_updated(dataset_processing_pipeline):
    """Test that the datasets dataframe was updated from its old version after get_new_dataset() was called."""
    df_old = pd.read_json("tests/data/test_datasets_old.json", orient="split")
    _sort_items_in_list_cols(dataset_processing_pipeline.list_columns, df_old, dataset_processing_pipeline.datasets)
    # the old file contained more datasets than the new file - check that they were removed
    assert not len(df_old) == len(dataset_processing_pipeline.datasets)


def test_correct_datasets_content(dataset_processing_pipeline):
    """Test that the datasets dataframe contains expected data."""
    df_correct = pd.read_json("tests/data/test_datasets_correct.json", orient="split")
    _sort_items_in_list_cols(dataset_processing_pipeline.list_columns, df_correct, dataset_processing_pipeline.datasets)
    pd.testing.assert_frame_equal(
        df_correct.reset_index(drop=True),
        dataset_processing_pipeline.datasets.reset_index(drop=True)
    )

def test_distributions_file_exists(config):
    """Test that distributions.json file was created."""
    assert os.path.exists(config["data"]["distributions"]["path"])


def test_datasets_file_exists(config):
    """Test that datasets.json file exists."""
    assert os.path.exists(config["data"]["datasets"]["path"])
