"""Helper functions for NKOD data processing."""

import numpy as np
import pandas as pd
import requests

from EnhancedDatasetSearch.utils import setup_logger

logger = setup_logger(__name__)


def split_dataframe(df: pd.DataFrame, chunk_size=100) -> list[pd.DataFrame]:
    """Split dataframe into chunks of specified size."""
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size:(i + 1) * chunk_size])
    return chunks


def download_df(file_path: str, url: str) -> pd.DataFrame:
    """Download a csv file, save it, load it and return the loaded pandas dataframe.

    Args:
        file_path: path to where the csv file will be stored
        url: download URL
    """
    response = requests.get(url)
    with open(file_path, "wb") as f:
        logger.info(f"Downloading {file_path.split("/")[-1]}.")
        f.write(response.content)
    df = pd.read_csv(file_path, sep=",", dtype="string")
    return df


def drop_irrelevant_columns(df: pd.DataFrame, irrelevant_columns: list) -> pd.DataFrame:
    """Drop irrelevant columns from datasets_raw."""
    # columns to drop - intersection of actual list of columns and specified list of columns to be dropped
    columns_to_drop = list(set(irrelevant_columns) & set(df.columns))
    df = df.drop(columns=columns_to_drop)
    return df


def merge_keywords_and_themes_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Merge rows about the same dataset into one so that we have one row per dataset.

    In the original csv downloaded from the Czech Dataset Portal there is a separate row for the same dataset
    for each keyword and theme associated with it. We want a single row per dataset with all
    keywords and themes merged into one list.
    """
    list_columns = ["keywords", "themes"]  # columns that contain multiple values per dataset -> merge into one list
    groupby_column = "url"  # group by dataset URL
    for col in list_columns:
        sub_df = df.groupby(groupby_column)[col].apply(lambda x: sorted(list(set(x.dropna())))).reset_index()
        df = df.drop(columns=[col])
        df = pd.merge(df, sub_df, on=groupby_column, how='left')

        # replace Nan and empty values with [], because the column should contain lists
        df[col] = df[col].map(lambda x: [] if x is None or x == np.nan or x == "" else x)

    df = df.drop_duplicates(subset=[groupby_column])

    # if a list contains NaN value, remove it from the list
    df = df.map(lambda x: x if not isinstance(x, list) else [i for i in x if pd.notna(i) and i != ""])
    df = df.replace(np.nan, None)  # remaining NaNs to None
    logger.info(f"Keywords and themes merged, number of rows: {df.shape[0]}.")
    return df
