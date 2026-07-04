"""Unit tests for data processing. All LLM and embedding model calls are replaced with mock
objects and empty response is always returned, so that the results are deterministic."""
import os
import shutil

import pandas as pd


def _sort_items_in_list_cols(list_columns: list, df1: pd.DataFrame, df2: pd.DataFrame):
    """Sorts the items in list columns of two dataframes so that they can be compared with assert_frame_equal."""
    for col in list_columns:
        df1[col] = df1[col].apply(sorted)
        df2[col] = df2[col].apply(sorted)


# copy old datasets file so that it can be used in NKOD pipeline
src_file = "tests/data/test_datasets_old.json"
dst_file = "tests/data/test_datasets.json"
shutil.copy(src_file, dst_file)

# set last modification time to now to ensure that the file will not be overwritten in tests
src_file = "tests/data/test_datasets_raw_copy.csv"
dst_file = "tests/data/test_datasets_raw.csv"
shutil.copy(src_file, dst_file)


def test_dataframes_exist(data_processing_pipeline):
    """Test that all needed dataframes are not None."""
    assert data_processing_pipeline.datasets_raw is not None
    assert data_processing_pipeline.datasets_raw_transformed is not None
    assert data_processing_pipeline.datasets is not None


def test_rows_merged(data_processing_pipeline):
    """Test that rows belonging to the same dataset are correctly merged into one."""
    assert len(data_processing_pipeline.datasets_raw_transformed) == 4  # there are four unique dataset URLs in the file


def test_list_columns(data_processing_pipeline):
    """Test that list columns exist, that they contain only lists and that each item of each list is not nan."""
    for col in data_processing_pipeline.list_columns:
        assert col in data_processing_pipeline.datasets_raw_transformed.columns
        assert data_processing_pipeline.datasets_raw_transformed[col].map(lambda x: isinstance(x, list)).all()
        assert data_processing_pipeline.datasets_raw_transformed[col].map(lambda x: all(pd.notna(item) for item in x)).all()


def test_correct_datasets_transformed_content(data_processing_pipeline):
    """Test that the datasets_raw_transformed dataframe contains expected data."""
    df_correct = pd.read_csv("tests/data/test_datasets_raw_transformed_correct.csv",
                             sep=",", quotechar='"',
                             converters={col: pd.eval for col in data_processing_pipeline.list_columns})
    _sort_items_in_list_cols(
        data_processing_pipeline.list_columns,
        df_correct,
        data_processing_pipeline.datasets_raw_transformed
    )
    pd.testing.assert_frame_equal(
        df_correct.reset_index(drop=True),
        data_processing_pipeline.datasets_raw_transformed.reset_index(drop=True)
    )


def test_datasets_updated(data_processing_pipeline):
    """Test that the datasets dataframe was updated from its old version after get_new_dataset() was called."""
    df_old = pd.read_json("tests/data/test_datasets_old.json", orient="split")
    _sort_items_in_list_cols(data_processing_pipeline.list_columns, df_old, data_processing_pipeline.datasets)
    # the old file contained more datasets than the new file - check that they were removed
    assert not len(df_old) == len(data_processing_pipeline.datasets)


def test_correct_datasets_content(data_processing_pipeline):
    """Test that the datasets dataframe contains expected data."""
    df_correct = pd.read_json("tests/data/test_datasets_correct.json", orient="split")
    _sort_items_in_list_cols(data_processing_pipeline.list_columns, df_correct, data_processing_pipeline.datasets)
    pd.testing.assert_frame_equal(
        df_correct.reset_index(drop=True),
        data_processing_pipeline.datasets.reset_index(drop=True)
    )

def test_distributions_file_exists(config):
    """Test that distributions.json file was created."""
    assert os.path.exists(config["data"]["distributions"]["path"])


def test_datasets_file_exists(config):
    """Test that datasets.json file exists."""
    assert os.path.exists(config["data"]["datasets"]["path"])
