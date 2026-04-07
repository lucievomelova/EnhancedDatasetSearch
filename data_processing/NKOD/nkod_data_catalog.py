"""File containing methods related specifically to the NKOD portal.

There will be a database containing info about all datasets. Every day, the new datove_sady and distribuce csvs
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

from data_processing.NKOD.data_catalog import DataCatalog
from data_processing.NKOD.spatial_and_temporal_data import add_metadata_to_datasets_from_sparql
from llama_index.core import Document
from pandas import Series

from data_processing.metadata import create_documents, enrich_metadata, clean_metadata, preprocess_temporal_coverage, \
    replace_nonfrequent_keywords_with_cluster_representatives
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



class NkodDataCatalog(DataCatalog):
    """Class for handling NKOD datasets."""

    def __init__(self, config: dict):
        super().__init__()
        # indicates whether the dataset DB is up to date - if True, there are no new or updated datasets so
        # no new documents need to be added to dataset DB
        self.db_up_to_date: bool = False
        self._download_new_data: bool = False
        self._always_download_new_data: bool = False  # TODO just for debugging
        self._update_datasets: bool = True
        self._run_preprocessing: bool = False  # TODO just for debugging

        self.config = config

        self._llm_config = config["llm"]
        self._data_processing_config = config["data_processing"]
        self._state_dir = config["state_dir"]
        self._column_mapping: dict = self._data_processing_config["column_mapping"]

        self.datasets_raw: pd.DataFrame | None = None
        """Dataset of datasets - contains information about each dataset on NKOD."""

        self._old_datasets_raw: pd.DataFrame | None = None
        """Old version of datasets_raw. Used to find new datasets by comparing it to the new datasets_raw."""

        self.datasets: pd.DataFrame | None = None
        """Dataset of datasets containing also dataset metadata, with renamed columns based on column mapping.
        Obtained by processing datasets_raw."""

        self.preprocessed_datasets_raw: pd.DataFrame | None = None
        """The preprocessed dataset that has cleaned metadata and merged list columns."""

        self._data_config = {
            "datasets_raw_path": config["data"]["datasets_raw"]["path"],
            "datasets_raw_url": config["data"]["datasets_raw"]["url"],
            "old_datasets_raw_path": config["data"]["datasets_raw"]["path"].replace(".csv", "_old.csv"),
            "datasets_path": config["data"]["datasets"]["path"],
        }

        self.all_keywords: list | None = None
        """List of all keywords present in the datasets metadata."""

        self.all_themes: list | None = None
        """List of all themes present in the datasets metadata."""

        self.all_categories: list = self._data_processing_config["categories"]
        """List of all categories."""

        self.all_providers: list | None = None
        """List of all providers of datasets at NKOD."""

        self.all_spatial_coverages: list | None = None
        """List of all spatial_coverages used in the datasets."""

        self.all_temporal_coverages: list | None = None
        """List of all temporal coverages used in the datasets."""

        self.all_categories_with_other_category = self.all_categories + [self._data_processing_config["other_category"]]
        """List of all categories including "other" category used when a dataset does not belong into any category."""

        self._all_keywords_raw: list | None = None
        """List of all keywords present in datasets_raw."""

        self._all_themes_raw: list | None = None
        """List of all themes present in the datasets_raw."""

    def _load_datasets(self) -> None:
        """Load datasets from csv file."""
        if not os.path.exists(self._data_config["datasets_path"]):
            logger.warning("Datasets file not found.")
            self.datasets = pd.DataFrame()
        else:
            logger.info("Loading datasets file.")
            self.datasets = pd.read_json(self._data_config["datasets_path"], orient="split")

        self.all_keywords = list(self.datasets["keywords"].explode().dropna().unique())
        self.all_themes = list(self.datasets["themes"].explode().dropna().unique())
        self.all_providers = list(self.datasets["provider"].dropna().unique())
        self.all_spatial_coverages = list(self.datasets["spatial_coverage"].explode().dropna().unique())
        self.all_temporal_coverages = list(self.datasets["temporal_coverage"].explode().dropna().unique())

    def _load_datasets_raw(self) -> None:
        """Load the raw NKOD dataset of datasets.

        Check if csv file exists and is up to date - if not, download it again and load it."""
        today = datetime.today().date()
        if os.path.exists(self._data_config["datasets_raw_path"]):
            logger.info("File datasets_raw exists.")
            mod_time = os.path.getmtime(self._data_config["datasets_raw_path"])
            mod_datetime = datetime.fromtimestamp(mod_time)
            if mod_datetime.date() != today:
                logger.info("Not modified today.")
                # if the file is outdated, save a copy -> later compare the old and new df to find new / updated rows
                self._old_datasets_raw = pd.read_csv(self._data_config["datasets_raw_path"], sep=",", dtype="string")
                self._old_datasets_raw.to_csv(self._data_config["old_datasets_raw_path"], index=False)
        if not os.path.exists(self._data_config["datasets_raw_path"]) or mod_datetime.date() != today:
            self.datasets_raw = download_df(self._data_config["datasets_raw_path"], self._data_config["datasets_raw_url"])
        else:
            logger.info("File datasets_raw is up to date, loading from disk.")
            self.datasets_raw = pd.read_csv(self._data_config["datasets_raw_path"], sep=",", dtype="string")
            # self.db_up_to_date = True

        self.datasets_raw = self.datasets_raw.rename(columns=self._column_mapping)  # rename columns based on column mapping
        if self._old_datasets_raw is not None:
            self._old_datasets_raw = self._old_datasets_raw.rename(columns=self._column_mapping)
        logger.info("Raw dataset info loaded.")


    def _merge_dataset_rows_into_one_row(self) -> None:
        """Merge rows about the same dataset into one so that we have one row per dataset in datasets_raw.

        Right now, there is a separate row for the same dataset for each keyword, theme, location,
        and category associated with it. We want a single row per dataset with all keywords merged into one column.
        """
        unwanted_columns = self._data_processing_config["unwanted_columns"]
        if all(col in self.datasets_raw.columns for col in unwanted_columns):
            self.datasets_raw = self.datasets_raw.drop(columns=unwanted_columns)

        # columns that contain multiple values per dataset -> merge the values into one list
        list_columns = ["keywords", "themes"]
        groupby_column = "url"  # group by dataset URL
        for col in list_columns:
            sub_df = self.datasets_raw.groupby(groupby_column)[col].apply(lambda x: list(set(x))).reset_index()
            self.datasets_raw = self.datasets_raw.drop(columns=[col])
            self.datasets_raw = pd.merge(self.datasets_raw, sub_df, on=groupby_column, how='left')

            # replace Nan and empty values with [], because the column should contain lists
            self.datasets_raw[col] = self.datasets_raw[col].map(lambda x: [] if x is None or x == np.nan or x == "" else x)

        self.datasets_raw = self.datasets_raw.drop_duplicates(subset=[groupby_column])

        # if a list contains NaN value, remove it from the list
        self.datasets_raw = self.datasets_raw.map(lambda x: x if not isinstance(x, list) else [i for i in x if pd.notna(i)])
        self.datasets_raw = self.datasets_raw.replace(np.nan, None)  # remaining NaNs to None
        logger.info(f"Number of rows: {self.datasets_raw.shape[0]}.")

    def _preprocess_metadata(self):
        """Preprocess metadata columns in the datasets_raw."""
        # init new columns as empty lists
        self.datasets_raw["categories"] = [[] for _ in range(len(self.datasets_raw))]
        self.datasets_raw["keyword_cluster_representatives"] = [[] for _ in range(len(self.datasets_raw))]

        clean_metadata(self.datasets_raw, self._data_processing_config["categories"], self._llm_config["model_name"], self._state_dir)


    def init(self):
        """Initialize the NKOD class - datasets_raw, datasets, all_keywords, all_themes."""
        if self._always_download_new_data:  # TODO debug option
            self.datasets_raw = download_df(self._data_config["datasets_raw_path"], self._data_config["datasets_raw_url"])
            self.datasets_raw = self.datasets_raw.rename(columns=self._column_mapping)  # rename columns based on column mapping
            logger.info("Raw dataset info loaded.")
        elif self._download_new_data:
            self._load_datasets_raw()
        else:  # TODO this is just for debugging, old file should not be used
            logger.info("Using old datasets_raw file, loading from disk - DEBUG OPTION.")
            self.datasets_raw = pd.read_csv(self._data_config["datasets_raw_path"], sep=",", dtype="string")
            self.datasets_raw = self.datasets_raw.rename(columns=self._column_mapping)  # rename columns based on column mapping
            # self.db_up_to_date = True

        if self._run_preprocessing:  # TODO debug option, to skip datasets_raw preprocessing
            self._merge_dataset_rows_into_one_row()
            self._preprocess_metadata()
            add_metadata_to_datasets_from_sparql(self.config, self.datasets_raw)

            # metadata cleaned -> find all unique keywords and themes, which will be used for metadata enrichment
            self._all_keywords_raw = list(self.datasets_raw["keywords"].explode().dropna().unique())
            logger.info(f"Number of unique keywords in datasets_raw: {len(self._all_keywords_raw)}")
            self._all_themes_raw = list(self.datasets_raw["themes"].explode().dropna().unique())
            logger.info(f"Number of unique themes in datasets_raw: {len(self._all_themes_raw)}")
        self._load_datasets()


    async def get_new_datasets(self) -> pd.DataFrame:
        """Get the list of new or updated datasets."""
        self.init()
        if self.db_up_to_date:
            logger.info("No new datasets found. DB is up to date.")
            return []
        if self._old_datasets_raw is not None:
            merged_df = pd.merge(self.datasets_raw, self._old_datasets_raw, on="url", how='left', indicator=True)
            new_datasets = merged_df[merged_df['_merge'] != 'both']
            new_datasets = new_datasets[self.datasets_raw.columns]
            logger.info(f"Number of new or updated datasets: {new_datasets.shape[0]}.")
        else:
            logger.info(f"Old file not found. Adding all datasets to DB ({self.datasets_raw.shape[0]} datasets).")
            new_datasets = self.datasets_raw

        # update datasets - process new datasets and add them to the existing datasets dataframe
        # do the updates in chunks -> in case of script failure we can resume from the last chunk
        if self._update_datasets:
            if not self.datasets.empty:  # don't use rows that are already loaded in datasets
                # find which rows are already in the datasets based on url
                new_datasets = pd.merge(self.datasets["url"], new_datasets, on="url", how='outer', indicator=True)
                # keep only rows that are not in both dataframes (=rows that do not have the merge label "_both")
                new_datasets = new_datasets.query("_merge != 'both'").drop('_merge', axis=1).reset_index(drop=True)
            chunks = split_dataframe(new_datasets, chunk_size=32)
            logger.info(f"Extending dataset metadata.")
            for i in range(len(chunks)):
                chunk = chunks[i]
                if chunk.empty:
                    break
                logger.info(f"Processing chunk {i+1}/{len(chunks)}.")
                new_rows = await self.create_metadata_for_chunk(chunk)
                new_rows_df = pd.DataFrame(new_rows)

                # store the results as csv, because json doesnt have an append option
                if not os.path.exists(self._data_config["datasets_path"]):
                    logger.info("Creating csv file for datasets.")
                    new_rows_df.to_csv(self._data_config["datasets_path"], index=False, header=True)
                else:
                    new_rows_df.to_csv(self._data_config["datasets_path"], index=False, header=False, mode="a")

        # load the csv that we were gradually writing to and store it in json, so that it can be
        # loaded faster in subsequent loads, because we dont have to use pd converters for list columns
        list_columns = ["keywords", "themes", "categories", "spatial_coverage", "temporal_coverage"]
        self.datasets = pd.read_csv(self._data_config["datasets_path"], sep=",",
                                           converters={col: pd.eval for col in list_columns})

        # validate that each list column truly contains a list - otherwise convert it to empty list
        for col in list_columns:
            self.datasets[col] = self.datasets[col].apply(lambda x: x if isinstance(x, list) else [])
        self.datasets.to_json(self._data_config["datasets_path"], orient="split", force_ascii=False)

        # metadata cleaning for the preprocessed datasets
        preprocess_temporal_coverage(self.datasets)
        clean_metadata(self.datasets, self._data_processing_config["categories"],
                       self._llm_config["model_name"], self._state_dir)
        replace_nonfrequent_keywords_with_cluster_representatives(self.datasets, self._llm_config["model_name"],
                                                                  self._state_dir)

        # store datasets after metadata cleaning
        self.datasets.to_csv(self._data_config["datasets_path"], index=False, header=True)

        new_datasets = self.datasets[self.datasets["url"].isin(new_datasets["url"])]
        return new_datasets

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

        The metadata are enriched with keywords, themes, categories, spatial_coverages and time periods
        generated by a LLM."""
        categories = self._data_processing_config["categories"]
        other_category = self._data_processing_config["other_category"]
        generated_metadata = enrich_metadata(row, self._all_keywords_raw, self._all_themes_raw, categories, other_category)

        keywords = row["keywords"] if row["keywords"] is not None else []
        keywords = list(set(keywords + generated_metadata["keywords"]))
        themes = row["themes"] if row["themes"] is not None else []
        themes = list(set(themes + generated_metadata["themes"]))
        categories = row["categories"] if row["categories"] is not None else []
        categories = list(set(categories + generated_metadata["categories"]))

        metadata = {
            "title": row["title"],
            "description": row["description"],
            "url": row["url"],
            "keywords": keywords,
            "themes": themes,
            "provider": row["provider"],
            "categories": categories,
            "spatial_coverage": generated_metadata["spatial_coverage"],
            "temporal_coverage": generated_metadata["temporal_coverage"],
        }
        return metadata

    def get_dataset_by_url(self, url: str) -> dict | None:
        """Get extended dataset info by URL."""
        dataset_row = self.datasets[self.datasets['url'] == url]
        if dataset_row.empty:
            return None

        row = dataset_row.iloc[0]
        return {
            'title': row['title'],
            'url': row['url'],
            'text': row['description'] if pd.notna(row['description']) else "",
            'metadata': {
                'keywords': row['keywords'] if isinstance(row['keywords'], list) else [],
                'themes': row['themes'] if isinstance(row['themes'], list) else [],
                'categories': row['categories'] if isinstance(row['categories'], list) else [],
                'spatial_coverage': row['spatial_coverage'] if isinstance(row['spatial_coverage'], list) else [],
                'temporal_coverage': row['temporal_coverage'] if isinstance(row['temporal_coverage'], list) else [],
                'provider': row['provider'] if 'provider' in row and not pd.isna(row['provider']) else '',
            }
        }
