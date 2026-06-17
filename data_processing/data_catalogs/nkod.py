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
import os

from data_processing.data_catalogs.data_catalog import DataCatalog
from data_processing.data_catalogs.distribution import download_distribution_info
from data_processing.data_catalogs.spatial_and_temporal_data import add_metadata_to_datasets_from_sparql
from ollama_client import OllamaClient
from pandas import Series

from data_processing.metadata import  enrich_metadata, clean_metadata, preprocess_temporal_coverage
from utils import setup_logger, dataset_detail_url
from data_processing.data_catalogs.df_utils import drop_irrelevant_columns, merge_keywords_and_themes_rows, split_dataframe, download_df

logger = setup_logger(__name__)
executor = ThreadPoolExecutor(max_workers=4)


class NkodDataCatalog(DataCatalog):
    """Class for handling NKOD datasets."""

    def __init__(self, config: dict, download_new_data: bool = False) -> None:
        """Initialize the NKOD data catalog class.

        Args:
            config: the loaded config
            download_new_data: specifies if new datasets should be downloaded from NKOD. This should be True if data
            processing will be started, otherwise it should be False, because it will trigger a knowledge base update.
        """
        super().__init__()

        self._db_up_to_date: bool = False
        """Indicates whether the knowledge base is up to date - if True, no new documents need to be added"""

        self._download_new_data: bool = download_new_data
        """Indicates if new datasets_raw file should be downloaded and processed or not."""

        self.config = config
        self.client = OllamaClient(self.config["llm"])

        self.distributions = download_distribution_info(config["data"]["distributions"]["path"],
                                                        config["data"]["distributions"]["url"],
                                                        config["data_processing"]["distribution_column_mapping"])

        self._data_processing_config = config["data_processing"]
        self._state_dir = config["state_dir"]
        self._column_mapping: dict = self._data_processing_config["column_mapping"]

        self.datasets_raw: pd.DataFrame
        """Dataset of datasets - contains information about each dataset on NKOD."""

        self._old_datasets_raw: pd.DataFrame | None = None
        """Old version of datasets_raw. Used to find new datasets by comparing it to the new datasets_raw."""

        self.datasets: pd.DataFrame
        """Dataset of datasets containing also dataset metadata, with renamed columns based on column mapping.
        Obtained by processing datasets_raw."""

        self._data_config = {
            "datasets_raw_path": config["data"]["datasets_raw"]["path"],
            "datasets_raw_url": config["data"]["datasets_raw"]["url"],
            "old_datasets_raw_path": config["data"]["datasets_raw"]["path"].replace(".csv", "_old.csv"),
            "datasets_path": config["data"]["datasets"]["path"],
        }

        self.all_keywords: set = set()
        """Set of all keywords present in the datasets metadata."""

        self.all_themes: set = set()
        """Set of all themes present in the datasets metadata."""

        self.all_categories: set = set(self._data_processing_config["categories"])
        """Set of all categories."""

        self.all_providers: set = set()
        """Set of all providers of datasets at NKOD."""

        self.all_spatial_coverages: set = set()
        """Set of all spatial_coverages used in the datasets."""

        self.all_temporal_coverages: set = set()
        """Set of all temporal coverages used in the datasets."""

        self.all_categories_with_other_category: set = self.all_categories | set(self._data_processing_config["other_category"])
        """Set of all categories including "other" category used when a dataset does not belong into any category."""

        self._all_keywords_raw: set
        """Set of all keywords present in datasets_raw."""

        self._all_themes_raw: set
        """Set of all themes present in the datasets_raw."""

        self._load_datasets_raw()
        self._load_datasets()

    def _load_datasets(self) -> None:
        """Load datasets file, which contains the preprocessed and enriched data about all datasets in NKOD."""

        # for checking if temporary file exists
        tmp_file_name = self._data_config["datasets_path"].replace(".json", f"_tmp.csv")

        if os.path.exists(self._data_config["datasets_path"]):
            logger.info("Loading datasets file.")
            self.datasets = pd.read_json(self._data_config["datasets_path"], orient="split")
        elif os.path.exists(tmp_file_name):  # tmp file exists -> loading of new datasets was interrupted -> continue
            list_cols = ["keywords", "themes", "categories", "spatial_coverage", "temporal_coverage"]
            self.datasets = pd.read_csv(tmp_file_name, sep=",", converters={col: pd.eval for col in list_cols})
        else:
            logger.warning("Datasets file not found.")
            self.datasets = pd.DataFrame()
            return

        self.all_keywords = set(self.datasets["keywords"].explode().dropna().unique())
        self.all_themes = set(self.datasets["themes"].explode().dropna().unique())
        self.all_providers = set(self.datasets["provider"].dropna().unique())
        self.all_spatial_coverages = set(self.datasets["spatial_coverage"].explode().dropna().unique())
        self.all_temporal_coverages = set(self.datasets["temporal_coverage"].explode().dropna().unique())

    def _load_datasets_raw(self) -> None:
        """Load the raw NKOD dataset of datasets.

        Check if csv file exists and is up to date - if not, download it again and load it."""

        if not self._download_new_data:  # TODO this is just for debugging, old file should not be used
            logger.info("Using old datasets_raw file, loading from disk.")
            self.datasets_raw = pd.read_csv(self._data_config["datasets_raw_path"], sep=",", dtype="string")
            self.datasets_raw = self.datasets_raw.rename(columns=self._column_mapping)  # rename columns based on column mapping

        else:
            today = datetime.today().date()
            if os.path.exists(self._data_config["datasets_raw_path"]):
                logger.info("File datasets_raw exists.")
                mod_time = os.path.getmtime(self._data_config["datasets_raw_path"])
                mod_datetime = datetime.fromtimestamp(mod_time)
                if mod_datetime.date() != today:
                    logger.info("Not modified today.")
                    # if the file is outdated, save a copy to have a backup
                    self._old_datasets_raw = pd.read_csv(self._data_config["datasets_raw_path"], sep=",", dtype="string")
                    self._old_datasets_raw.to_csv(self._data_config["old_datasets_raw_path"], index=False)
            if not os.path.exists(self._data_config["datasets_raw_path"]) or mod_datetime.date() != today:
                self.datasets_raw = download_df(self._data_config["datasets_raw_path"], self._data_config["datasets_raw_url"])
            else:
                logger.info("File datasets_raw is up to date, loading from disk.")
                self.datasets_raw = pd.read_csv(self._data_config["datasets_raw_path"], sep=",", dtype="string")

            self.datasets_raw = self.datasets_raw.rename(columns=self._column_mapping)  # rename columns based on column mapping
            logger.info("Raw dataset info loaded, starting preprocessing.")

        self.datasets_raw = merge_keywords_and_themes_rows(self.datasets_raw)
        self.datasets_raw = drop_irrelevant_columns(self.datasets_raw, self._data_processing_config["irrelevant_columns"])
        # add category column, now empty for each dataset
        self.datasets_raw["categories"] = [[] for _ in range(len(self.datasets_raw))]
        clean_metadata(self.datasets_raw,
                       self.client,
                       self._data_processing_config["categories"],
                       self.config["llm"]["model_name"],
                       self._state_dir)
        add_metadata_to_datasets_from_sparql(self.config, self.datasets_raw)

        # metadata cleaned -> find all unique keywords and themes, which will be used for metadata enrichment
        self._all_keywords_raw = set(self.datasets_raw["keywords"].explode().dropna().unique())
        logger.info(f"Number of unique keywords in datasets_raw: {len(self._all_keywords_raw)}")
        self._all_themes_raw = set(self.datasets_raw["themes"].explode().dropna().unique())
        logger.info(f"Number of unique themes in datasets_raw: {len(self._all_themes_raw)}")


    async def get_new_datasets(self) -> pd.DataFrame:
        """Get the list of new or updated datasets."""
        if self._db_up_to_date:
            logger.info("No new datasets found. DB is up to date.")
            return pd.DataFrame()

        # file for storing intermediate results - it must be csv, because json doesn't have an append option
        # but later we want to use json because of faster loading time
        tmp_file_name = self._data_config["datasets_path"].replace(".json", f"_tmp.csv")

        # find new or updated datasets by comparing new and old datasets if old datasets exist
        removed_urls = None  # for tracking which urls were present before but are not present now
        if not self.datasets.empty:
            merged_df = pd.merge(self.datasets_raw, self.datasets["url"], on="url", how='outer', indicator=True)
            new_datasets = merged_df[merged_df['_merge'] == 'left_only'][self.datasets_raw.columns]
            removed_urls = merged_df[merged_df['_merge'] == 'right_only']["url"].tolist()
            logger.info(f"Number of new or updated datasets: {new_datasets.shape[0]}.")
        else:  # otherwise all datasets are new
            logger.info(f"Old file not found. Adding all datasets to DB ({self.datasets_raw.shape[0]} datasets).")
            new_datasets = self.datasets_raw

        # update datasets - process new datasets and add them to the existing datasets dataframe
        if not self.datasets.empty:
            self.datasets.to_csv(tmp_file_name, index=False, header=True)  # store current state of datasets in tmp file
            if removed_urls:   # remove deleted datasets from self.datasets
                self.datasets = self.datasets[~self.datasets['url'].isin(removed_urls)]
                logger.info(f"Removed {len(removed_urls)} datasets from DB.")
            # find which rows are already in the datasets based on url - don't add them again
            merged_df = pd.merge(self.datasets["url"], new_datasets, on="url", how='outer', indicator=True)
            new_datasets = merged_df.query("_merge == 'right_only'").drop('_merge', axis=1).reset_index(drop=True)

        # do the updates in chunks -> in case of script failure we can resume from the last chunk
        chunks = split_dataframe(new_datasets, chunk_size=32)
        logger.info(f"Extending dataset metadata.")
        for index, chunk in enumerate(chunks, start=1):
            if chunk.empty:
                break
            logger.info(f"Processing chunk {index}/{len(chunks)}.")
            new_rows = await self._create_metadata_for_chunk(chunk)
            new_rows_df = pd.DataFrame(new_rows)

            # store / append new results
            if not os.path.exists(tmp_file_name):
                logger.info("Creating csv file for datasets.")
                new_rows_df.to_csv(tmp_file_name, index=False, header=True)
            else:
                new_rows_df.to_csv(tmp_file_name, index=False, header=False, mode="a")

        # load the csv that we were gradually writing to and save it as json, so that it can be
        # loaded faster in subsequent loads, because we don't have to use pd converters for list columns
        list_cols = ["keywords", "themes", "categories", "spatial_coverage", "temporal_coverage"]
        self.datasets = pd.read_csv(tmp_file_name, sep=",", converters={col: pd.eval for col in list_cols})
        if os.path.exists(tmp_file_name):  # remove temporary file, it is not needed anymore
            os.remove(tmp_file_name)

        for col in list_cols:  # validate that each list column truly contains a list - otherwise set it as empty list
            self.datasets[col] = self.datasets[col].apply(lambda x: x if isinstance(x, list) else [])

        # metadata cleaning for the enhanced datasets
        preprocess_temporal_coverage(self.datasets)
        clean_metadata(self.datasets, self.client, self._data_processing_config["categories"],
                       self.config["llm"]["model_name"], self._state_dir)

        # save after metadata cleaning as json
        self.datasets.to_json(self._data_config["datasets_path"], orient="split", force_ascii=False)

        new_datasets = self.datasets[self.datasets["url"].isin(new_datasets["url"])]
        return new_datasets

    async def _create_metadata_for_chunk(self, new_datasets:  pd.DataFrame):
        """Create metadata dict for a chunk of new datasets asynchronously."""
        new_rows = [self._get_metadata_for_row_async(row) for _, row in new_datasets.iterrows()]
        return await asyncio.gather(*new_rows)

    async def _get_metadata_for_row_async(self, row: Series) -> dict:
        """Asynchronous wrapper for get_metadata_for_row method."""
        # this method is needed for the async code to work properly
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self._get_metadata_for_row, row)

    def _get_metadata_for_row(self, row: Series) -> dict:
        """Get metadata for a row which represents one dataset.

        The metadata are enriched with keywords, themes, categories, spatial_coverages and time periods
        generated by a LLM."""
        categories = self._data_processing_config["categories"]
        other_category = self._data_processing_config["other_category"]
        generated_metadata = enrich_metadata(row, self.client, self._all_keywords_raw, categories, other_category)

        keywords = row["keywords"] if row["keywords"] is not None else []
        keywords = list(set(keywords + generated_metadata["keywords"]))
        themes = row["themes"] if row["themes"] is not None else []
        categories = row["categories"] if row["categories"] is not None else []
        categories = list(set(categories + generated_metadata["categories"]))
        self._all_keywords_raw.update(set(keywords))  # update keywords set

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
