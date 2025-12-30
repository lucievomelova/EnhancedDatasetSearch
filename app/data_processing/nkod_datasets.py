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

from data_processing.llamaindex_documents import assign_keywords, assign_theme, \
    classify_into_domain, detect_region, detect_time_period, create_documents
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
        self.extended_df: pd.DataFrame | None = None

        self._extended_df_path = config["data"]["extended_df"]["path"]
        self._load_extended_df(self._extended_df_path)

    def _load_extended_df(self, path: str) -> None:
        """Load the extended NKOD dataset containing also dataset metadata."""
        if not os.path.exists(path):
            logger.warning("Extended NKOD dataset file not found.")
            self.extended_df = pd.DataFrame()
        else:
            logger.info("Loading extended NKOD dataset.")
            self.extended_df = pd.read_csv(path, sep=",")

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
                self._old_super_df = pd.read_csv(path, sep=",", dtype="string")
                self._old_super_df.to_csv(path.replace(".csv", "_old.csv"), index=False)
        if not os.path.exists(path) or mod_datetime.date() != today:
            self.super_df = self._download_df(path, url)
        else:
            logger.info("File is up to date, loading from disk.")
            self.super_df = pd.read_csv(path, sep=",", dtype="string")
            self.rag_up_to_date = True
        self.super_df = self._merge_dataset_rows_into_one_row(self.super_df)
        logger.info("Dataset info loaded.")

    def _download_df(self, path: str, url: str) -> pd.DataFrame:
        """Download dataset from data.gov.cz, load csv and return it as a pandas dataframe."""
        response = requests.get(url)
        with open(path, "wb") as f:
            logger.info(f"Downloading {path.split("/")[-1]}.")
            f.write(response.content)
        df = pd.read_csv(path, sep=",", dtype="string")
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

    def get_new_datasets(self) -> list[Document]:
        """Get the list of new or updated datasets as llama index Documents."""
        # if self.rag_up_to_date:
        #     logger.info("No new datasets found. RAG store is up to date.")
        #     return []
        # if self._old_super_df is not None:
        #     merged_df = pd.merge(self.super_df, self._old_super_df, on='datová_sada', how='left', indicator=True)
        #     new_datasets = merged_df[merged_df['_merge'] != 'both']
        #     new_datasets = new_datasets[self.super_df.columns]
        #     logger.info(f"Number of new or updated datasets: {new_datasets.shape[0]}.")
        # else:
        #     logger.info(f"Old file not found. Adding all datasets to RAG db ({self.super_df.shape[0]} datasets).")
        #     new_datasets = self.super_df
        new_datasets = self.super_df
        new_rows = [self.create_metadata_for_row(row, self.all_keywords(), self.all_themes()) for _, row in new_datasets.iterrows()]
        self.extended_df = pd.concat([self.extended_df, pd.DataFrame(new_rows)], ignore_index=True)  # append new datasets to extended df
        self.extended_df.to_csv(self._extended_df_path, index=False)
        documents = create_documents(self.extended_df)
        logger.info("Documents created.")
        input()
        return documents

    def all_keywords(self) -> list:
        """Get a list of all keywords present in the super dataset."""
        keywords = list(self.super_df["klíčová_slova"].explode().unique())
        keywords = [kw for kw in keywords if kw is not None]
        with open("keywords.txt", "w", encoding="utf-8") as f:
            for theme in keywords:
                f.write(f"{theme}\n")
        return keywords

    def all_themes(self) -> list:
        """Get a list of all themes present in the super dataset."""
        themes = list(self.super_df["téma"].explode().unique())
        themes = [theme for theme in themes if theme is not None]
        # save to file themes.txt
        with open("themes.txt", "w", encoding="utf-8") as f:
            for theme in themes:
                f.write(f"{theme}\n")

        return themes

    def create_metadata_for_row(self, row: Series, all_keywords: list, all_themes: list) -> dict:
        """Create metadata dictionary from a dataframe row."""
        # columns - datová_sada, název, popis, poskytovatel, klíčová_slova, prostorové_pokrytí, téma,
        # periodicita_aktualizace, právní_předpis, kategorie_hvd_název

        keyword_list = assign_keywords(row, all_keywords)
        theme_list = assign_theme(row, all_themes) if row['téma'] is None else row['téma']

        row = {
            "title": row['název'],
            "description": row['popis'],
            "url": row['datová_sada'],
            "keywords": keyword_list,  # list of keywords
            "themes": theme_list,  # list of themes
            "provider": row['poskytovatel'],
            "legal_regulations": row['právní_předpis'],  # list of legal regulations
            "categories": row['kategorie_hvd_název'],  # list of categories
            "domain": classify_into_domain(row),
            "region": detect_region(row),
            "time_period": detect_time_period(row),
        }

        print(row)
        # input()
        return row
