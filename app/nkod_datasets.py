"""File containing methods related specifically to the NKOD portal.

There will be a RAG database containing info about all datasets. Every day, the new datove_sady and distribuce csvs
will be downloaded and if there are changes detected in some datasets at NKOD, their info will be deleted from DB and
then added again.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import requests
import os

from llama_index.core import Document
from pandas import Series

from utils import setup_logger

logger = setup_logger(__name__)


class InformativeDatasetClass:
    """Base class for informative dataset class.

     This class contains information about all datasets present at the dataset portal used
     - e.g. the Czech portal https://data.gov.cz/ or the EU portal data.europa.eu."""

    def __init__(self):
        self.super_df: pd.DataFrame
        """The super dataset - contains info about each dataset present on the portal."""

    def get_new_datasets(self) -> list[Document]:
        """List of new or updated datasets as llama index Documents."""
        pass


class NKOD(InformativeDatasetClass):
    """Class for handling NKOD datasets."""

    def __init__(self, config: dict):
        super().__init__()
        # indicates whether the RAG is up to date - if True, there are no new or updated datasets so
        # no new documents need to be added to RAG store
        self.rag_up_to_date: bool = False

        # a dataframe that contains the old version of the super dataset - used to compare with the new one
        # and find new datasets, that will be added to RAG store
        self._old_super_df: pd.DataFrame | None = None

        self._load_super_df(**config["data"]["datasets"])

    def _load_super_df(self, path: str, url: str) -> None:
        """Load the NKOD super dataset.

        Check if csv file exists and is up to date - if not, download it again, load it and return the loaded df."""
        today = datetime.today().date()
        if os.path.exists(path):
            logger.info("File exists")
            mod_time = os.path.getmtime(path)
            mod_datetime = datetime.fromtimestamp(mod_time)
            if mod_datetime.date() != today:
                logger.info("Not modified today")
                # if the file is outdated, save a copy of the old file and save the contents into a df
                # we will use later compare the old and new df to find new or updated rows
                self._old_super_df = pd.read_csv(path, sep=",")
                self._old_super_df.to_csv(path.replace(".csv", "_old.csv"), index=False)
        if not os.path.exists(path) or mod_datetime.date() != today:
            self.super_df = self._download_df(path, url)
        else:
            logger.info("File is up to date, loading from disk.")
            self.super_df = pd.read_csv(path, sep=",")
            self.rag_up_to_date = True
        # self.super_df = pd.read_csv(path, sep=",")
        self.super_df = self._merge_dataset_rows_into_one_row(self.super_df)
        logger.info("Dataset info loaded.")

    def _download_df(self, path: str, url: str) -> pd.DataFrame:
        """Download dataset from data.gov.cz, load csv and return it as a pandas dataframe."""
        response = requests.get(url)
        with open(path, "wb") as f:
            logger.info(f"Downloading {path.split("/")[-1]}.")
            f.write(response.content)
        df = pd.read_csv(path, sep=",")
        return df

    def _merge_dataset_rows_into_one_row(self, data_df: pd.DataFrame) -> pd.DataFrame:
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

        # handle NaN values
        # change all single NaN lists to None
        data_df = data_df.map(lambda x: None if isinstance(x, list) and len(x) == 1 and pd.isna(x[0]) else x)
        # if a longer list contains NaN, remove it
        data_df = data_df.map(lambda x: x if not isinstance(x, list) else [i for i in x if pd.notna(i)])
        # remaining NaNs
        data_df = data_df.replace(np.nan, None)

        logger.info(f"Number of rows: {data_df.shape[0]}.")
        return data_df

    def _create_document_from_row(self, row: Series) -> Document:
        """Create a llama index Document for a  llama index Document from a dataframe row."""
        # columns - datová_sada, název, popis, poskytovatel, klíčová_slova, prostorové_pokrytí, téma,
        # periodicita_aktualizace, právní_předpis, kategorie_hvd_název
        text = f"{row['název']}\n{row['popis']}"
        metadata = {
            "title": row['název'],
            "url": row['datová_sada'],
            "keywords": row['klíčová_slova'],  # list of keywords
            "provider": row['poskytovatel'],
            "themes": row['téma'],  # list of themes
            "legal_regulations": row['právní_předpis'],  # list of legal regulations
            "categories": row['kategorie_hvd_název'],  # list of categories
        }
        return Document(text=text, metadata=metadata, doc_id=row['datová_sada'])

    def get_new_datasets(self) -> list[Document]:
        """Get the list of new or updated datasets as llama index Documents."""
        if self.rag_up_to_date:
            logger.info("No new datasets found. RAG store is up to date.")
            return []
        if self._old_super_df is not None:
            merged_df = pd.merge(self.super_df, self._old_super_df, on='datová_sada', how='left', indicator=True)
            new_or_updated_df = merged_df[merged_df['_merge'] != 'both']
            new_or_updated_df = new_or_updated_df[self.super_df.columns]
            logger.info(f"Number of new or updated datasets: {new_or_updated_df.shape[0]}.")
            documents = [self._create_document_from_row(row) for _, row in new_or_updated_df.iterrows()]
        else:
            logger.info(f"Old file not found. Adding all datasets to RAG store (number of datasets: {self.super_df.shape[0]}).")
            documents = [self._create_document_from_row(row) for _, row in self.super_df.iterrows()]
        logger.info("Documents created.")
        return documents
