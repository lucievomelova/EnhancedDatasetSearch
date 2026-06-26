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

from data_processing.data_catalog import DataCatalog
from data_processing.NKOD.distributions import download_distribution_info
from data_processing.NKOD.spatial_and_temporal_data import add_metadata_to_datasets_from_sparql
from llama_index.core import Document
from ollama_client import OllamaClient
from pandas import Series

from data_processing.NKOD.metadata import  enrich_metadata, clean_metadata, preprocess_temporal_coverage
from utils import setup_logger, dataset_detail_url
from data_processing.NKOD.utils import drop_irrelevant_columns, merge_keywords_and_themes_rows, split_dataframe, download_df

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

        self._download_new_data: bool = download_new_data
        """Indicates if new datasets_raw file should be downloaded and processed or not."""

        self.config = config
        self.client = OllamaClient(self.config["llm"])

        self.distributions = download_distribution_info(config["data"]["distributions"]["path"],
                                                        config["data"]["distributions"]["url"],
                                                        config["data_processing"]["distribution_column_mapping"],
                                                        self._download_new_data)

        self._data_processing_config = config["data_processing"]
        self._state_dir = config["state_dir"]
        self._column_mapping: dict = self._data_processing_config["column_mapping"]

        self.datasets_raw: pd.DataFrame
        """Dataset of datasets - contains information about each dataset on NKOD."""

        self.datasets_raw_transformed: pd.DataFrame
        """Transformed datasets_raw - columns are renamed, rows belonging to the same dataset are merged into one,
        spatial and temporal coverage is added and initial preprocessing is applied."""

        self.datasets: pd.DataFrame
        """Dataset of datasets containing also dataset metadata, with renamed columns based on column mapping.
        Obtained by processing datasets_raw."""

        self._data_config = {
            "datasets_raw_path": config["data"]["datasets_raw"]["path"],
            "datasets_raw_url": config["data"]["datasets_raw"]["url"],
            "datasets_transformed_path": config["data"]["datasets_raw"]["path"].replace(".csv", f"_transformed.csv"),
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

    def _load_datasets(self) -> None:
        """Load datasets file, which contains the preprocessed and enriched data about all datasets in NKOD."""

        # for checking if temporary file exists
        tmp_file_name = self._data_config["datasets_path"].replace(".json", f"_tmp.csv")

        if os.path.exists(self._data_config["datasets_path"]):
            logger.info("Loading datasets file.")
            self.datasets = pd.read_json(self._data_config["datasets_path"], orient="split")
        elif os.path.exists(tmp_file_name):  # tmp file exists -> loading of new datasets was interrupted -> continue
            logger.info("Loading datasets from tmp file.")
            self.datasets = pd.read_csv(tmp_file_name, sep=",", converters={col: pd.eval for col in self.list_columns})
        else:
            logger.warning("Datasets file not found.")
            self.datasets = pd.DataFrame()
            return

        for col in self.list_columns:
            self.datasets[col] = self.datasets[col].apply(lambda lst: [x for x in lst if pd.notna(x) and x is not None and x != np.nan])

        self.all_keywords = set(self.datasets["keywords"].explode().dropna().unique())
        self.all_themes = set(self.datasets["themes"].explode().dropna().unique())
        self.all_providers = set(self.datasets["provider"].dropna().unique())
        self.all_spatial_coverages = set(self.datasets["spatial_coverage"].explode().dropna().unique())
        self.all_temporal_coverages = set(self.datasets["temporal_coverage"].explode().dropna().unique())

    def _load_datasets_raw(self, download_new_data: bool) -> None:
        """Load the raw NKOD dataset of datasets."""
        # TODO this is just for debugging, old file should not be used
        if not download_new_data or os.path.exists(self._data_config["datasets_raw_path"]):
            logger.info("Using old datasets_raw file.")
            self.datasets_raw = pd.read_csv(self._data_config["datasets_raw_path"], sep=",", dtype="string")
            return
        else:
            today = datetime.today().date()
            if os.path.exists(self._data_config["datasets_raw_path"]):
                logger.info("File containing datasets_raw exists.")
                mod_time = os.path.getmtime(self._data_config["datasets_raw_path"])
                mod_datetime = datetime.fromtimestamp(mod_time)
                if mod_datetime.date() != today:
                    # if the file is outdated, rename it and keep it as a backup
                    old_file_name = self._data_config["datasets_path"].replace(".json", f"_old.json")
                    os.rename(self._data_config["datasets_raw_path"], old_file_name)
            self.datasets_raw = download_df(self._data_config["datasets_raw_path"], self._data_config["datasets_raw_url"])
        logger.info("Raw dataset info loaded, starting preprocessing.")

    def _transform_datasets_raw(self, download_new_data: bool) -> None:
        """Apply initial transformations on the raw dataset."""
        # TODO this is just for debugging, old file should not be used
        if not download_new_data and os.path.exists(self._data_config["datasets_transformed_path"]):
            logger.info("Using old datasets_raw_transformed file.")
            self.datasets_raw_transformed = pd.read_csv(self._data_config["datasets_transformed_path"], sep=",", dtype="string")
            return

        self.datasets_raw_transformed = self.datasets_raw.rename(columns=self._column_mapping)
        self.datasets_raw_transformed = merge_keywords_and_themes_rows(self.datasets_raw_transformed)
        self.datasets_raw_transformed = drop_irrelevant_columns(self.datasets_raw_transformed,
                                                                self._data_processing_config["irrelevant_columns"])
        # add category column, now empty for each dataset
        self.datasets_raw_transformed["categories"] = [[] for _ in range(len(self.datasets_raw_transformed))]
        clean_metadata(self.datasets_raw_transformed,
                       None,
                       self._data_processing_config["categories"],
                       self.config["llm"]["model_name"],
                       self._state_dir)
        add_metadata_to_datasets_from_sparql(self.config, self.datasets_raw_transformed)

        self.datasets_raw_transformed.to_csv(self._data_config["datasets_transformed_path"], index=False, header=True)

        # metadata cleaned -> find all unique keywords and themes, which will be used for metadata enrichment
        self._all_keywords_raw = set(self.datasets_raw_transformed["keywords"].explode().dropna().unique())
        logger.info(f"Number of unique keywords in transformed datasets_raw: {len(self._all_keywords_raw)}")

    def get_new_datasets(self) -> pd.DataFrame:
        """Get a dataframe of new or updated datasets and update self.datasets with the new data."""
        # find new or updated datasets by comparing new and old datasets if old datasets exist
        removed_urls = None  # for tracking which urls were present before but are not present now
        if not self.datasets.empty:
            merged_df = pd.merge(self.datasets_raw_transformed, self.datasets["url"], on="url", how='outer', indicator=True)
            new_datasets = merged_df[merged_df['_merge'] == 'left_only'][self.datasets_raw_transformed.columns]
            removed_urls = merged_df[merged_df['_merge'] == 'right_only']["url"].tolist()
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
        return new_datasets

    async def enrich_new_datasets_metadata(self, new_datasets: pd.DataFrame) -> None:
        """Enrich metadata of new_datasets and update self.datasets with the enriched data."""

        # file for storing intermediate results - it must be csv, because json doesn't have an append option
        # but later we want to use json because of faster loading time
        tmp_file_name = self._data_config["datasets_path"].replace(".json", f"_tmp.csv")
        print(self.datasets.columns)
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
        preprocess_temporal_coverage(self.datasets)
        clean_metadata(self.datasets, self.client, self._data_processing_config["categories"],
                       self.config["llm"]["model_name"], self._state_dir)

        # save after metadata cleaning as json
        self.datasets.to_json(self._data_config["datasets_path"], orient="split", force_ascii=False)

    async def update_datasets(self) -> None:
        """Update datasets based on the last downloaded datasets_raw table.

        Remove datasets that are not present, update existing datasets or add new datasets, then enrich new
        or existing datasets' metadata using an LLM."""
        new_datasets = self.get_new_datasets()
        await self.enrich_new_datasets_metadata(new_datasets)

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

    @staticmethod
    def _create_documents(datasets: pd.DataFrame) -> list[Document]:
        """Create llama index Documents from the dataframe. Each row will be used to create one Document.

        The Document text will be: title + description + keywords + themes + categories + provider. All
        columns will also be stored in the metadata of the Document (even those that will be part of the text)."""
        documents = []
        descriptions = datasets["description"]
        datasets = datasets.where(datasets.notna(), None)
        metadata_df = datasets.drop(columns="description").to_dict(orient="records")
        logger.info(f"Creating {len(datasets)} llamaindex Documents.")

        for description, metadata in zip(descriptions, metadata_df):
            text = f"""
                {metadata['title']}
                {description}\n
                Poskytovatel: {metadata['provider']}
                Klíčová slova: {metadata['keywords']}
                Témata: {metadata['themes']}
                Kategorie: {metadata['categories']}
                Prostorové pokrytí": {metadata['spatial_coverage']}
                Časové pokrytí: {metadata['temporal_coverage']}
            """
            document = Document(text=text, metadata=metadata, id_=metadata["url"])
            documents.append(document)

        logger.info("Documents created.")
        return documents

    def get_dataset_by_url(self, url: str) -> dict | None:
        """Get extended dataset info by URL."""
        dataset_row = self.datasets[self.datasets['url'] == url]
        if dataset_row.empty:  # try also the url used on dataset detail page
            detail_urls = self.datasets['url'].apply(lambda u: dataset_detail_url(self.config, u))
            dataset_row = self.datasets[detail_urls == url]
        if dataset_row.empty:
            return None  # still no result -> dataset with the given URL not found

        row = dataset_row.iloc[0]
        return {
            'title': row['title'],
            'url': row['url'],
            'text': row['description'] if pd.notna(row['description']) else "",
            'distributions': self.distributions[self.distributions['dataset_url'] == row['url']].to_dict(orient='records'),
            'categorization_metadata': {
                'keywords': row['keywords'] if 'keywords' in row and isinstance(row['keywords'], list) else [],
                'themes': row['themes'] if 'themes' in row and isinstance(row['themes'], list) else [],
                'categories': row['categories'] if 'categories' in row and isinstance(row['categories'], list) else [],
                'spatial_coverage': row['spatial_coverage'] if 'spatial_coverage' in row and isinstance(row['spatial_coverage'], list) else [],
                'temporal_coverage': row['temporal_coverage'] if 'temporal_coverage' in row and isinstance(row['temporal_coverage'], list) else [],
                'provider': row['provider'] if 'provider' in row and not pd.isna(row['provider']) else '',
            }
        }

    def get_filters_with_counts(self) -> dict:
        if not self.filters_with_counts:
            self.filters_with_counts = {
                col_name: {"title": title, "value_counts": self.datasets[col_name].explode().value_counts(dropna=True).to_dict()}
                for (col_name, title) in zip(self._filter_columns, self._filter_column_names)
            }
        return self.filters_with_counts
