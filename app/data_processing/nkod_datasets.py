"""File containing methods related specifically to the NKOD portal.

There will be a RAG database containing info about all datasets. Every day, the new datove_sady and distribuce csvs
will be downloaded and if there are changes detected in some datasets at NKOD, their info will be deleted from DB and
then added again.
"""
import ast
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import os

from llama_index.core import Document
from pandas import Series

from data_processing.llamaindex_documents import create_documents, enrich_metadata
from utils import setup_logger

logger = setup_logger(__name__)

executor = ThreadPoolExecutor(max_workers=4)


def split_dataframe(df, chunk_size=100) -> list[pd.DataFrame]:
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size:(i + 1) * chunk_size])
    return chunks


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
        self._download_new_data: bool = False
        self._recreate_extended_df: bool = False

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
            list_columns = ["keywords", "themes", "categories", "legal_regulations", "region", "time_period"]
            self.extended_df = pd.read_csv(path, sep=",", converters={col: pd.eval for col in list_columns})

    def _load_super_df(self, path: str, url: str) -> None:
        """Load the NKOD super dataset.

        Check if csv file exists and is up to date - if not, download it again, load it and return the loaded df."""
        if self._download_new_data:
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
        else:
            logger.info("Using old dataset file, loading from disk.")
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

        columns_with_duplicates = ["klíčová_slova", "prostorové_pokrytí", "téma", "právní_předpis"]
        # columns that are not needed - e.g. tema_IRI when we also have column tema, je_součástí_IRI is always empty
        # also kategorie_hvd is mostly None and it should only be use for legislation related data -> drop it
        columns_to_be_dropped = ["kategorie_hvd_IRI", "kategorie_hvd_název", "je_součástí_IRI", "periodicita_aktualizace_IRI", "téma_IRI", "poskytovatel_IRI"]
        data_df = data_df.drop(columns=columns_to_be_dropped)

        for col in columns_with_duplicates:
            sub_df = data_df.groupby('datová_sada')[col].apply(lambda x: list(set(x))).reset_index()
            data_df = data_df.drop(columns=[col])
            data_df = pd.merge(data_df, sub_df, on='datová_sada', how='left')

            # TODO check if it works
            # replace Nan and empty values with []
            data_df[col] = data_df[col].map(lambda x: [] if x is None or x == np.nan or x == "" else x)

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

    async def get_new_datasets(self) -> list[Document]:
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
        # new_rows = [self.create_metadata_for_row(row) for _, row in new_datasets.iterrows()]

        if self._recreate_extended_df:
            new_datasets = self.super_df
            chunks = split_dataframe(new_datasets, chunk_size=50)
            for chunk in chunks:
                logger.info(f"Processing chunk with {chunk.shape[0]} datasets.")
                new_rows = await self.run_all(chunk)
                if not os.path.exists(self._extended_df_path):
                    logger.info("Creating extended csv file.")
                    pd.DataFrame(new_rows).to_csv(self._extended_df_path, index=False, header=True)
                else:
                    pd.DataFrame(new_rows).to_csv(self._extended_df_path, index=False, header=False, mode="a")
        self._load_extended_df(self._extended_df_path)
        documents = create_documents(self.extended_df)
        logger.info("Documents created.")
        return documents

    def get_keywords(self) -> list:
        """Get a list of all keywords present in the super dataset."""
        keywords = list(self.super_df["klíčová_slova"].explode().unique())
        keywords = [kw for kw in keywords if kw is not None]
        return keywords

    def get_themes(self) -> list:
        """Get a list of all themes present in the super dataset."""
        themes = list(self.super_df["téma"].explode().unique())
        themes = [theme for theme in themes if theme is not None]
        return themes

    def get_categories(self) -> list:
        """Get a list of categories.

        Based on categories from the EU data portal -
        https://op.europa.eu/en/web/eu-vocabularies/concept-scheme/-/resource?uri=http://publications.europa.eu/resource/authority/data-theme"""
        categories = ["Zemědělství, rybolov, lesnictví a výživa", "Vzdělávání, kultura a sport", "Životní prostředí",
                   "Energie", "Doprava", "Věda a technika", "Hospodářství a finance", "Populace a společnost", "Zdraví",
                   "Vláda a veřejný sektor", "Regiony a města", "Spravedlnost, právní systém a veřejná bezpečnost",
                   "Mezinárodní otázky"]
        return categories

    async def run_all(self, new_datasets):
        new_rows = [self.enrich_async(row) for _, row in new_datasets.iterrows()]
        return await asyncio.gather(*new_rows)

    async def enrich_async(self, row: Series) -> dict:
        loop = asyncio.get_running_loop()
        while True:
            try:
                result = await loop.run_in_executor(executor, self.create_metadata_for_row, row)
                return result
            except Exception as e:
                logger.error(f"Error processing row {row['název']}: {e}, retrying...")
        # logger.info(f"Row: {row["název"]}")
        # return result

    def create_metadata_for_row(self, row: Series) -> dict:
        """Create metadata dictionary from a dataframe row."""
        # columns - datová_sada, název, popis, poskytovatel, klíčová_slova, prostorové_pokrytí, téma,
        # periodicita_aktualizace, právní_předpis, kategorie_hvd_název

        generated_metadata = enrich_metadata(row, self.get_keywords(), self.get_themes(), self.get_categories())
        keywords = row['klíčová_slova']
        keywords = keywords + generated_metadata["keywords"] if keywords is not None else generated_metadata["keywords"]
        themes = row['téma']
        themes = themes + generated_metadata["themes"] if themes is not None else generated_metadata["themes"]

        for category in self.get_categories():
            if category in themes and category not in generated_metadata["categories"]:
                generated_metadata["categories"].append(category)

        metadata = {
            "title": row['název'],
            "description": row['popis'],
            "url": row['datová_sada'],
            "keywords": keywords,  # list of keywords
            "themes": themes,  # list of themes
            "provider": row['poskytovatel'],
            "legal_regulations": row['právní_předpis'],  # list of legal regulations
            "categories": generated_metadata["categories"],
            "region": generated_metadata["regions"],
            "time_period": generated_metadata["time_periods"],
        }
        return metadata
