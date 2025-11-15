"""File containing methods related specifically to the NKOD portal."""

from datetime import datetime

import numpy as np
import pandas as pd
import requests
import os

import yaml
from llama_index.core import Document
from pandas import Series

from utils import setup_logger

logger = setup_logger(__name__)


def get_informative_dataset(config: dict) -> pd.DataFrame:
    """Get the NKOD datasets dataframe with merged rows."""
    data_df = _load_df(**config["data"]["datasets"])
    data_df = _merge_dataset_rows_into_one_row(data_df)
    # distribution_df = load_df(**config["data"]["distribution"])
    return data_df


def _load_df(path: str, url: str) -> pd.DataFrame:
    """Load dataset from csv and return it as a pandas dataframe.

    Check if csv file exists and is up to date - if not, download it again, load it and return the loaded df.
    """
    response = requests.get(url)
    if os.path.exists(path):
        mod_time = os.path.getmtime(path)
        mod_datetime = datetime.fromtimestamp(mod_time)
        today = datetime.today().date()
    if not os.path.exists(path) or mod_datetime.date() != today:
        with open(path, "wb") as f:
            logger.info(f"Downloading {path.split("/")[-1]}.")
            f.write(response.content)
    df = pd.read_csv(path, sep=",")
    logger.info("Dataset info loaded.")
    return df


def _merge_dataset_rows_into_one_row(data_df: pd.DataFrame) -> pd.DataFrame:
    """Merge rows about the same dataset into one so that we have one row per dataset in data_df.

    Right now, there is a separate row for the same dataset for each keyword, theme, location, legal regulation
    and category associated with it. We want a single row per dataset with all keywords merged into one column.
    """

    columns_with_duplicates = ["klíčová_slova", "prostorové_pokrytí", "téma", "právní_předpis", "kategorie_hvd_název"]
    # columns that are not needed - e.g. tema_IRI when we also have column tema, je_součástí_IRI is always empty
    columns_to_be_dropped = ["kategorie_hvd_IRI", "je_součástí_IRI", "periodicita_aktualizace_IRI", "téma_IRI", "poskytovatel_IRI"]
    data_df = data_df.drop(columns=columns_to_be_dropped)

    for col in columns_with_duplicates:
        sub_df = data_df.groupby('datová_sada')[col].apply(lambda x: list(set(x))).reset_index()
        data_df = data_df.drop(columns=[col])
        data_df = pd.merge(data_df, sub_df, on='datová_sada', how='left')

    data_df = data_df.drop_duplicates(subset=['datová_sada'])
    logger.info(f"Number of rows: {data_df.shape[0]}.")

    # handle NaN values
    # change all single NaN lists to None
    data_df = data_df.map(lambda x: None if isinstance(x, list) and len(x) == 1 and pd.isna(x[0]) else x)
    # if a longer list contains NaN, remove it
    data_df = data_df.map(lambda x: x if not isinstance(x, list) else [i for i in x if pd.notna(i)])
    # remaining NaNs
    data_df = data_df.replace(np.nan, None)

    logger.info(f"Number of rows: {data_df.shape[0]}.")
    return data_df


def create_document_from_row(row: Series) -> Document:
    """Create a llama index Document for a  llama index Document from a dataframe row."""
    # columns - datová_sada, název, popis, poskytovatel, klíčová_slova, prostorové_pokrytí, téma,
    # periodicita_aktualizace, právní_předpis, kategorie_hvd_název
    content = f"{row['název']}\n{row['popis']}"
    metadata = {
        "title": row['název'],
        "url": row['datová_sada'],
        "keywords": row['klíčová_slova'],  # list of keywords
        "provider": row['poskytovatel'],
        "themes": row['téma'],  # list of themes
        "legal_regulations": row['právní_předpis'],  # list of legal regulations
        "categories": row['kategorie_hvd_název'],  # list of categories
    }
    return Document(content=content, metadata=metadata, doc_id=row['datová_sada'])
