"""File containing methods related specifically to the NKOD portal.

There will be a RAG database containing info about all datasets. Every day, the new datove_sady and distribuce csvs
will be downloaded and if there are changes detected in some datasets at NKOD, their info will be deleted from DB and
then added again.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import os

from llama_index.core import Document
from pandas import Series

from data_processing.keywords import get_representatives
from data_processing.metadata import create_documents, enrich_metadata, clean_metadata, \
    add_cluster_representatives_to_metadata
from utils import setup_logger

logger = setup_logger(__name__)
executor = ThreadPoolExecutor(max_workers=4)


def split_dataframe(df, chunk_size=100) -> list[pd.DataFrame]:
    chunks = list()
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        chunks.append(df[i * chunk_size:(i + 1) * chunk_size])
    return chunks


def download_df(path: str, url: str) -> pd.DataFrame:
    """Download a csv file, load it and return it as a pandas dataframe."""
    response = requests.get(url)
    with open(path, "wb") as f:
        logger.info(f"Downloading {path.split("/")[-1]}.")
        f.write(response.content)
    df = pd.read_csv(path, sep=",", dtype="string")
    return df


class InformativeDatasetClass:
    """Base class for informative dataset class.

     This class contains information about all datasets present at the dataset portal used
     - e.g. the Czech portal https://data.gov.cz/ or the EU portal data.europa.eu."""

    def __init__(self):
        self.super_df: pd.DataFrame | None
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
        self._always_download_new_data: bool = False  # TODO just for debugging
        self._update_extended_df: bool = True

        self._data_processing_config = config["rag"]["data_processing"]
        self._column_mapping: dict = self._data_processing_config["column_mapping"]

        self.super_df: pd.DataFrame | None = None

        self._old_super_df: pd.DataFrame | None = None
        """Old version of the super dataset used to find new datasets by comparing it to the new super_df."""

        self.extended_df: pd.DataFrame | None = None
        """The extended NKOD dataset containing also dataset metadata, with renamed columns based on column mapping."""

        self._data_config = {
            "super_df_path": config["data"]["datasets"]["path"],
            "super_df_url": config["data"]["datasets"]["url"],
            "old_super_df_path": config["data"]["datasets"]["path"].replace(".csv", "_old.csv"),
            "extended_df_path": config["data"]["extended_df"]["path"]
        }

        self._all_keywords: list | None = None
        """List of all keywords present in the datasets metadata."""

        self._all_themes: list | None = None
        """List of all themes present in the datasets metadata."""

    def _load_extended_df(self) -> None:
        """Load the extended NKOD dataset containing also dataset metadata."""
        if not os.path.exists(self._data_config["extended_df_path"]):
            logger.warning("Extended NKOD dataset file not found.")
            self.extended_df = pd.DataFrame()
        else:
            logger.info("Loading extended NKOD dataset.")
            list_columns = ["keywords", "themes", "categories", "legal_regulations", "region", "time_period"]
            self.extended_df = pd.read_csv(self._data_config["extended_df_path"], sep=",",
                                           converters={col: pd.eval for col in list_columns})

    def _load_super_df(self) -> None:
        """Load the NKOD super dataset.

        Check if csv file exists and is up to date - if not, download it again, load it and return the loaded df."""
        today = datetime.today().date()
        if os.path.exists(self._data_config["super_df_path"]):
            logger.info("File exists")
            mod_time = os.path.getmtime(self._data_config["super_df_path"])
            mod_datetime = datetime.fromtimestamp(mod_time)
            if mod_datetime.date() != today:
                logger.info("Not modified today")
                # if the file is outdated, save a copy -> later compare the old and new df to find new / updated rows
                self._old_super_df = pd.read_csv(self._data_config["super_df_path"], sep=",", dtype="string")
                self._old_super_df.to_csv(self._data_config["old_super_df_path"], index=False)
        if not os.path.exists(self._data_config["super_df_path"]) or mod_datetime.date() != today:
            self.super_df = download_df(self._data_config["super_df_path"], self._data_config["super_df_url"])
        else:
            logger.info("File is up to date, loading from disk.")
            self.super_df = pd.read_csv(self._data_config["super_df_path"], sep=",", dtype="string")
            # self.rag_up_to_date = True

        self.super_df = self.super_df.rename(columns=self._column_mapping)  # rename columns based on column mapping
        self._old_super_df = self._old_super_df.rename(columns=self._column_mapping)
        logger.info("Dataset info loaded.")


    def _merge_dataset_rows_into_one_row(self) -> None:
        """Merge rows about the same dataset into one so that we have one row per dataset in super_df.

        Right now, there is a separate row for the same dataset for each keyword, theme, location, legal regulation
        and category associated with it. We want a single row per dataset with all keywords merged into one column.
        """
        self.super_df = self.super_df.drop(columns=self._data_processing_config["unwanted_columns"])

        # columns that contain multiple values per dataset -> merge the values into one list
        list_columns = ["keywords", "spatial_coverage", "themes", "legal_regulations"]
        groupby_column = "url"  # group by dataset URL
        for col in list_columns:
            sub_df = self.super_df.groupby(groupby_column)[col].apply(lambda x: list(set(x))).reset_index()
            self.super_df = self.super_df.drop(columns=[col])
            self.super_df = pd.merge(self.super_df, sub_df, on=groupby_column, how='left')

            # replace Nan and empty values with [], because the column should contain lists
            self.super_df[col] = self.super_df[col].map(lambda x: [] if x is None or x == np.nan or x == "" else x)

        self.super_df = self.super_df.drop_duplicates(subset=[groupby_column])

        # if a list contains NaN value, remove it from the list
        self.super_df = self.super_df.map(lambda x: x if not isinstance(x, list) else [i for i in x if pd.notna(i)])
        self.super_df = self.super_df.replace(np.nan, None)  # remaining NaNs to None
        logger.info(f"Number of rows: {self.super_df.shape[0]}.")

    def _preprocess_metadata(self):
        """Preprocess metadata columns in the super_df."""
        # init new columns as empty lists
        self.super_df["categories"] = [[] for _ in range(len(self.super_df))]
        self.super_df["keyword_cluster_representatives"] = [[] for _ in range(len(self.super_df))]

        clean_metadata(self.super_df, self._data_processing_config["categories"])
        add_cluster_representatives_to_metadata(self.super_df)

    def init(self):
        """Initialize the NKOD class - super_df, extended_df, all_keywords, all_themes."""
        if self._always_download_new_data:
            self.super_df = download_df(self._data_config["super_df_path"], self._data_config["super_df_url"])
            self.super_df = self.super_df.rename(columns=self._column_mapping)  # rename columns based on column mapping
            logger.info("Dataset info loaded.")
        elif self._download_new_data:
            self._load_super_df()
        else:  # TODO this is just for debugging, old file should not be used
            logger.info("Using old dataset file, loading from disk.")
            self.super_df = pd.read_csv(self._data_config["super_df_path"], sep=",", dtype="string")
            self.super_df = self.super_df.rename(columns=self._column_mapping)  # rename columns based on column mapping
            # self.rag_up_to_date = True
        self._merge_dataset_rows_into_one_row()

        self._preprocess_metadata()
        self._load_extended_df()

        # metadata cleaned -> find all unique keywords and themes, which will be used for metadata enrichment
        self._all_keywords = list(self.super_df["keywords"].explode().dropna().unique())
        self._all_themes = list(self.super_df["themes"].explode().dropna().unique())


    async def get_new_datasets(self) -> list[Document]:
        """Get the list of new or updated datasets as llama index Documents."""
        self.init()
        if self.rag_up_to_date:
            logger.info("No new datasets found. RAG store is up to date.")
            return []
        if self._old_super_df is not None:
            merged_df = pd.merge(self.super_df, self._old_super_df, on="url", how='left', indicator=True)
            new_datasets = merged_df[merged_df['_merge'] != 'both']
            new_datasets = new_datasets[self.super_df.columns]
            logger.info(f"Number of new or updated datasets: {new_datasets.shape[0]}.")
        else:
            logger.info(f"Old file not found. Adding all datasets to RAG db ({self.super_df.shape[0]} datasets).")
            new_datasets = self.super_df

        if self._update_extended_df:
            if not self.extended_df.empty:  # dont use rows that are already loaded in the extended df
                # find which rows are already in the extended_df based on url
                new_datasets = pd.merge(self.extended_df["url"], new_datasets, on="url", how='outer', indicator=True)
                # keep only rows that are not in both dataframes (=rows that do not have the merge label "_both")
                new_datasets = new_datasets.query("_merge != 'both'").drop('_merge', axis=1).reset_index(drop=True)
            chunks = split_dataframe(new_datasets, chunk_size=20)
            for chunk in chunks:
                logger.info(f"Processing chunk with {chunk.shape[0]} datasets.")
                new_rows = await self.create_metadata_for_chunk(chunk)
                if not os.path.exists(self._data_config["extended_df_path"]):
                    logger.info("Creating extended csv file.")
                    pd.DataFrame(new_rows).to_csv(self._data_config["extended_df_path"], index=False, header=True)
                    continue

                pd.DataFrame(new_rows).to_csv(self._data_config["extended_df_path"], index=False, header=False, mode="a")

        self._load_extended_df()
        documents = create_documents(self.extended_df)
        return documents

    async def create_metadata_for_chunk(self, new_datasets:  pd.DataFrame):
        """Create metadata dict for a chunk of new datasets asynchronously."""
        new_rows = [self.get_metadata_for_row_async(row) for _, row in new_datasets.iterrows()]
        return await asyncio.gather(*new_rows)

    async def get_metadata_for_row_async(self, row: Series) -> dict:
        """Asynchronous wrapper for get_metadata_for_row method."""
        # this method is needed for the async code to work properly
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.get_metadata_for_row, row)

    def get_metadata_for_row(self, row: Series) -> dict:
        """Get metadata for a row which represents one dataset.

        The metadata are enriched with keywords, themes, categories, regions and time periods
        generated by a LLM."""
        categories = self._data_processing_config["categories"]
        other_category = self._data_processing_config["other_category"]
        generated_metadata = enrich_metadata(row, self._all_keywords, self._all_themes, categories, other_category)

        keywords = row["keywords"] if row["keywords"] is not None else []
        keywords = keywords + generated_metadata["keywords"]
        themes = row["themes"] if row["themes"] is not None else []
        themes = themes + generated_metadata["themes"]
        for category in categories:
            if category in themes and category not in generated_metadata["categories"]:
                generated_metadata["categories"].append(category)

        metadata = {
            "title": row["title"],
            "description": row["description"],
            "url": row["url"],
            "keywords": keywords,
            "keyword_cluster_representatives": row["keyword_cluster_representatives"],
            "themes": themes,
            "provider": row["provider"],
            "legal_regulations": row["legal_regulations"],
            "categories": generated_metadata["categories"],
            "region": generated_metadata["regions"],
            "time_period": generated_metadata["time_periods"],
        }
        return metadata
