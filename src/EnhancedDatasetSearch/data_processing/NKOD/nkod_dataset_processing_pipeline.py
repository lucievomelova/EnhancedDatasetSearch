"""File containing methods related specifically to the NKOD portal.

There will be a database containing info about all datasets. Every day, the new datove_sady and distribuce csvs
will be downloaded and if there are changes detected in some datasets at NKOD, their info will be deleted from DB and
then added again.
"""
import asyncio
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pandas as pd
from pandas import Series

from EnhancedDatasetSearch.data_processing.dataset_processing_pipeline import DatasetProcessingPipeline
from EnhancedDatasetSearch.data_processing.NKOD.metadata import (
    clean_metadata, enrich_metadata, process_spatial_and_temporal_coverage
)
from EnhancedDatasetSearch.data_processing.NKOD.spatial_and_temporal_data_sparql import add_metadata_to_datasets_from_sparql
from EnhancedDatasetSearch.data_processing.NKOD.utils import (
    download_df, merge_keywords_and_themes_rows, split_dataframe, drop_irrelevant_columns
)
from EnhancedDatasetSearch.ollama_client import OllamaClient
from EnhancedDatasetSearch.utils import setup_logger

logger = setup_logger(__name__)
executor = ThreadPoolExecutor(max_workers=4)


class NkodDatasetProcessingPipeline(DatasetProcessingPipeline):
    """Class representing the NKOD dataset processing pipeline.

    It downloads the metadata dataset from the Czech Dataset Portal, applies transformation, cleaning,
    metadata enrichment and then the resulting dataset is stored in memory. The same dataset is then loaded by
    the NkodDataCatalog class."""

    def __init__(self, config: dict) -> None:
        """Initialize the NkodDataProcessingPipeline class.

        Args:
            config: the loaded config
            processing will be started, otherwise it should be False, because it will trigger a knowledge base update.
        """
        super().__init__()

        self._download_new_data: bool = True
        """Indicates if new datasets_raw file should be downloaded and processed or not.
        This should always be true, unless you want to load a specific dataset file instead of the current NKOD file."""

        self.config = config
        self.client = OllamaClient(self.config["data_processing"]["llm"], self.config["data_processing"]["llm"]["timeout"])

        self._data_processing_config = config["data_processing"]
        self._state_dir = config["state_dir"]
        self._column_mapping: dict = self._data_processing_config["column_mapping"]

        self.datasets_raw: pd.DataFrame
        """Dataset of all datasets' metadata - contains raw information about each dataset in NKOD.
        This is the dataset that is donwloaded frm the Czech Dataset Portal, before any processing is applied."""

        self.datasets_raw_transformed: pd.DataFrame
        """Transformed datasets_raw - columns are renamed, rows belonging to the same dataset are merged into one,
        spatial and temporal coverage is added and initial preprocessing is applied."""

        self.datasets: pd.DataFrame
        """Processed dataset of datasets, with renamed columns based on column mapping.
        Obtained by enhancing the datasets' metadata by an LLM. This is the dataset that represents NKOD in our system.
        After data processing pipeline is completed, it is stored in memory and loaded by the NkodDataCatalog class."""

        self._data_config = {
            "datasets_raw_path": config["data"]["datasets_raw"]["path"],
            "datasets_raw_url": config["data"]["datasets_raw"]["url"],
            "datasets_transformed_path": config["data"]["datasets_raw"]["path"].replace(".csv", f"_transformed.csv"),
            "datasets_path": config["data"]["datasets"]["path"]
        }

        self.all_categories: set = set(self._data_processing_config["categories"])
        """Set of all categories."""

        self._all_keywords_raw: set
        """Set of all keywords present in datasets_raw."""

        self.list_columns: list = ["keywords", "themes", "categories", "spatial_coverage", "temporal_coverage"]
        """Columns in self.datasets that contain lists."""

        self._filter_columns: list = ["keywords", "themes", "categories", "spatial_coverage", "temporal_coverage", "provider"]
        """Columns that can be used for search result filtering."""

        self._filter_column_names: list = ["Keywords", "Themes", "Categories", "Spatial coverage", "Temporal coverage", "Provider"]
        self.filters_with_counts: dict = {}
        """Dictionary that stores occurrence counts of each unique value in every filter category."""

        self._load_datasets_raw(self._download_new_data)
        self._transform_datasets_raw(self._download_new_data)
        self._load_datasets()
        self._load_distribution_info(self._download_new_data)

    def _load_datasets(self) -> None:
        """Load datasets file, which contains the preprocessed and enriched data about all datasets in NKOD."""

        # temporary file is used to store intermediate data during metadata enrichment
        tmp_file_name = self._data_config["datasets_path"].replace(".json", f"_tmp.csv")

        if os.path.exists(self._data_config["datasets_path"]):
            logger.info("Loading datasets file.")
            self.datasets = pd.read_json(self._data_config["datasets_path"], orient="split")

        elif os.path.exists(tmp_file_name):
            # tmp file exists -> metadata enrichment of new datasets was interrupted -> continue where we stopped
            logger.info("Loading datasets from tmp file.")
            self.datasets = pd.read_csv(tmp_file_name, sep=",", converters={col: pd.eval for col in self.list_columns})
        else:
            logger.warning("Datasets file not found.")
            self.datasets = pd.DataFrame()
            return

    def _load_datasets_raw(self, download_new_data: bool) -> None:
        """Load the raw NKOD dataset of datasets."""
        if not download_new_data:
            return
        today = datetime.today().date()
        if os.path.exists(self._data_config["datasets_raw_path"]):
            mod_time = os.path.getmtime(self._data_config["datasets_raw_path"])
            mod_datetime = datetime.fromtimestamp(mod_time)
            #  check if datasets_raw file is up to date - if it is, we don't have to download it again
            if mod_datetime.date() == today:
                logger.info("File containing datasets_raw exists and is up to date - loading.")
                self.datasets_raw = pd.read_csv(self._data_config["datasets_raw_path"], sep=",", dtype="string")
                return
            else:
                # if the file is outdated, rename it and keep it as a backup
                old_file_name = self._data_config["datasets_raw_path"].replace(".csv", f"_old.csv")
                os.rename(self._data_config["datasets_raw_path"], old_file_name)
        # download the most recent raw datasets file
        self.datasets_raw = download_df(self._data_config["datasets_raw_path"], self._data_config["datasets_raw_url"])
        logger.info("Raw dataset info loaded, starting preprocessing.")

    def _transform_datasets_raw(self, download_new_data: bool) -> None:
        """Apply initial transformations on the raw dataset."""
        if not download_new_data:
            return  # we don't need to load transformed_datasets_raw

        today = datetime.today().date()
        mod_datetime = None
        if os.path.exists(self._data_config["datasets_transformed_path"]):
            mod_time = os.path.getmtime(self._data_config["datasets_transformed_path"])
            mod_datetime = datetime.fromtimestamp(mod_time)
        #  check if datasets_raw_transformed file is up to date - if it is, we can load the data from there
        if mod_datetime is not None and mod_datetime.date() == today:
            logger.info("File containing datasets_raw_transformed exists and is up to date - loading.")
            self.datasets_raw_transformed = pd.read_csv(
                self._data_config["datasets_transformed_path"], sep=",", converters={col: pd.eval for col in self.list_columns})
        else:
            # transform data from datasets_raw to obtained datasets_raw_transformed
            self.datasets_raw_transformed = self.datasets_raw.rename(columns=self._column_mapping)
            self.datasets_raw_transformed = merge_keywords_and_themes_rows(self.datasets_raw_transformed)
            self.datasets_raw_transformed = drop_irrelevant_columns(self.datasets_raw_transformed,
                                                                    self._data_processing_config["irrelevant_columns"])

            add_metadata_to_datasets_from_sparql(self.config, self.datasets_raw_transformed)

            # store full description hash before any processing is applied so that later we are able to detect changes
            # in datasets' metadata
            self.datasets_raw_transformed["full_description"] = self.datasets_raw_transformed.apply(
                lambda r: f"{r["title"]}\n{r["description"]}\n({r["provider"]} | "
                          f"{", ".join(sorted(r["keywords"]))} | "
                          f"{", ".join(sorted(r["themes"]))} | "
                          f"{", ".join(sorted(r["spatial_coverage"]))} | "
                          f"{", ".join(sorted(r["temporal_coverage"]))})", axis=1
            )
            self.datasets_raw_transformed["full_description_hash"] = self.datasets_raw_transformed.apply(
                lambda r: hashlib.md5(r["full_description"].encode("utf-8")).hexdigest(), axis=1
            )
            # add category column, now empty for each dataset
            self.datasets_raw_transformed["categories"] = [[] for _ in range(len(self.datasets_raw_transformed))]
            clean_metadata(self.datasets_raw_transformed,
                           self.client,
                           self._data_processing_config["categories"],
                           self._state_dir)

            self.datasets_raw_transformed.to_csv(self._data_config["datasets_transformed_path"], index=False, header=True)

        # metadata cleaned -> find all unique keywords and themes, which will be used for metadata enrichment
        self._all_keywords_raw = set(self.datasets_raw_transformed["keywords"].explode().dropna().unique())
        logger.info(f"Number of unique keywords in transformed datasets_raw: {len(self._all_keywords_raw)}")

    def _load_distribution_info(self, download_new_data: bool) -> None:
        """Download distribution table from NKOD, transform it and store the strasnformed file as json."""
        path = self.config["data"]["distributions_raw"]["path"]
        url = self.config["data"]["distributions_raw"]["url"]
        column_mapping = self.config["data_processing"]["distribution_column_mapping"]
        if not download_new_data and os.path.exists(path):
            distributions_df = pd.read_csv(path, sep=",", dtype="string")
        else:
            distributions_df = download_df(path, url)

        columns_to_keep = list(column_mapping.keys())
        distributions_df = distributions_df[columns_to_keep]
        distributions_df = distributions_df.rename(columns=column_mapping)
        distributions_df.to_json(self.config["data"]["distributions"]["path"], orient="split", force_ascii=False)

    def get_new_datasets(self) -> tuple[pd.DataFrame, list]:
        """Get a dataframe of new or updated datasets and update self.datasets with the new data."""
        # find new or updated datasets by comparing new and old datasets if old datasets exist
        removed_urls = None  # for tracking which urls were present before but are not present now
        if not self.datasets.empty:
            # merge by url and full description to find all changes
            merged_df = pd.merge(
                self.datasets_raw_transformed, self.datasets[["url", "full_description_hash"]],
                on=["url", "full_description_hash"], how='outer', indicator=True
            )

            # get urls of removed and updated datasets, so we can remove them
            removed_urls = merged_df[merged_df['_merge'] == 'right_only']["url"].tolist()

            # new OR updated datasets, but we remove updated datasets and process them again, so we call them all new
            new_datasets = merged_df[merged_df['_merge'] == 'left_only'][self.datasets_raw_transformed.columns]
            new_datasets = new_datasets.drop(columns="full_description")

            logger.info(f"Number of new or updated datasets: {new_datasets.shape[0]}.")
        else:  # otherwise all datasets are new
            logger.info(f"Old file not found. Adding all datasets to DB ({self.datasets_raw_transformed.shape[0]} datasets).")
            new_datasets = self.datasets_raw_transformed

        # update datasets - process new datasets and add them to the existing datasets dataframe
        if not self.datasets.empty:
            if removed_urls:   # remove deleted datasets from self.datasets
                self.datasets = self.datasets[~self.datasets['url'].isin(removed_urls)]
                logger.info(f"Removed {len(removed_urls)} datasets from DB.")
            # find which rows are already in the datasets based on url - don't add them again
            merged_df = pd.merge(self.datasets["url"], new_datasets, on="url", how='outer', indicator=True)
            new_datasets = merged_df.query("_merge == 'right_only'").drop('_merge', axis=1).reset_index(drop=True)
        return new_datasets, removed_urls

    async def enrich_new_datasets_metadata(self, new_datasets: pd.DataFrame) -> None:
        """Enrich metadata of new_datasets and update self.datasets with the enriched data."""
        # file for storing intermediate results - it must be csv, because json doesn't have an append option
        # but later we want to use json because of faster loading time
        tmp_file_name = self._data_config["datasets_path"].replace(".json", f"_tmp.csv")
        if not self.datasets.empty:  # if datasets is not empty, put current state of it in tmp file
            self.datasets.to_csv(tmp_file_name, index=False, header=True)
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
        self.datasets = pd.read_csv(tmp_file_name, sep=",", converters={col: pd.eval for col in self.list_columns})
        if os.path.exists(tmp_file_name):  # remove temporary file, it is not needed anymore
            os.remove(tmp_file_name)

        for col in self.list_columns:  # validate that each list column truly contains a list - otherwise set it as empty list
            self.datasets[col] = self.datasets[col].apply(lambda x: x if isinstance(x, list) else [])

        # metadata cleaning for the enhanced datasets
        process_spatial_and_temporal_coverage(self.datasets)
        clean_metadata(self.datasets, self.client, self._data_processing_config["categories"], self._state_dir)

        # save after metadata cleaning as json
        self.datasets.to_json(self._data_config["datasets_path"], orient="split", force_ascii=False)

    async def update_datasets(self) -> tuple[pd.DataFrame, list]:
        """Update self.datasets and find and return new or updated datasets and removed datasets.

        Update datasets based on the last downloaded datasets_raw table. Remove datasets that are not present,
        update existing datasets or add new datasets, then enrich new or existing datasets' metadata using an LLM.

        Returns:
            a tuple: (pd.DataFrame, list), where the dataframe contains new or updated datasets in hte same format as
            self.dataset. The list contains a list of removed datasets URLs.
        """
        new_datasets, removed_urls = self.get_new_datasets()
        await self.enrich_new_datasets_metadata(new_datasets)
        new_datasets = self.datasets[self.datasets["url"].isin(new_datasets["url"])]
        return new_datasets, removed_urls

    async def _create_metadata_for_chunk(self, new_datasets:  pd.DataFrame):
        """Create metadata dict for a chunk of new datasets asynchronously."""
        new_rows = [self._enrich_datasets_metadata_async(row) for _, row in new_datasets.iterrows()]
        return await asyncio.gather(*new_rows)

    async def _enrich_datasets_metadata_async(self, row: Series) -> dict:
        """Asynchronous wrapper for _enrich_datasets_metadata."""
        # this method is needed for the async code to work properly
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self._enrich_datasets_metadata, row)

    def _enrich_datasets_metadata(self, row: Series) -> dict:
        """Enrich metadata of a dataset represented by the given row.

        The metadata categories that are considered are: keywords, themes, categories, spatial_coverages and
        temporal_coverage. If no (ior not enough) metadata is assigned to the dataset in a specific metadata category,
        new metadata are generated by an LLM."""
        categories = self._data_processing_config["categories"]
        other_category = self._data_processing_config["other_category"]
        generated_metadata = enrich_metadata(row, self.client, categories, other_category)

        keywords = row["keywords"] if row["keywords"] is not None else []
        keywords = list(set(keywords + generated_metadata["keywords"]))
        themes = row["themes"] if row["themes"] is not None else []
        categories = row["categories"] if row["categories"] is not None else []
        categories = list(set(categories + generated_metadata["categories"]))
        spatial_coverage = row["spatial_coverage"] if row["spatial_coverage"] is not None else []
        spatial_coverage = list(set(spatial_coverage + generated_metadata["spatial_coverage"]))
        temporal_coverage = row["temporal_coverage"] if row["temporal_coverage"] is not None else []
        temporal_coverage = list(set(temporal_coverage + generated_metadata["temporal_coverage"]))

        metadata = {
            "title": row["title"],
            "description": row["description"],
            "url": row["url"],
            "keywords": keywords,
            "themes": themes,
            "provider": row["provider"],
            "categories": categories,
            "spatial_coverage": spatial_coverage,
            "temporal_coverage": temporal_coverage,
            "full_description_hash": row["full_description_hash"]
        }
        return metadata
