import os

import pandas as pd

from EnhancedDatasetSearch.data_catalog import DataCatalog
from EnhancedDatasetSearch.utils import dataset_detail_url, setup_logger

logger = setup_logger(__name__)


class NkodDataCatalog(DataCatalog):
    """Class representing the NKOD.

    After the data processing pipeline finishes, this class loads the stored datasets file and distributions file.
    The datasets dataframe is the processed and enhanced metadata dataset representing the NKOD datasets.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.config: dict = config

        self.datasets: pd.DataFrame
        """The dataset of datasets metadata that represents the NKOD in our system."""

        self._datasets_path: str = config["data"]["datasets"]["path"]
        """Path where the datasets file is stored."""

        self.distributions: pd.DataFrame
        """Dataset of distributions - links each dataset with all its available distributions."""

        self._distributions_path: str = config["data"]["distributions"]["path"]
        """Path where the datasets file is stored."""

        self._filter_columns: list = ["keywords", "themes", "categories", "spatial_coverage", "temporal_coverage", "provider"]
        """Columns that can be used for search result filtering."""

        self.filters_with_counts: dict = {}
        """Dictionary that stores occurrence counts of each unique value in every filter category."""

        self._load_datasets()
        self._load_distributions()

    def _load_datasets(self) -> None:
        """Load datasets file, which contains the preprocessed and enriched metadata for all datasets in the NKOD."""
        if os.path.exists(self._datasets_path):
            logger.info("Loading datasets file.")
            self.datasets = pd.read_json(self._datasets_path, orient="split")
        else:
            logger.error("Datasets file not found.")
            raise FileNotFoundError("Datasets file not found - to obtain it you must run DataProcessingPipeline.")

        self.all_keywords = set(self.datasets["keywords"].explode().dropna().unique())
        self.all_themes = set(self.datasets["themes"].explode().dropna().unique())
        self.all_providers = set(self.datasets["provider"].dropna().unique())
        self.all_spatial_coverages = set(self.datasets["spatial_coverage"].explode().dropna().unique())
        self.all_temporal_coverages = set(self.datasets["temporal_coverage"].explode().dropna().unique())

    def _load_distributions(self):
        """Load distributions file."""
        if os.path.exists(self._distributions_path):
            logger.info("Loading distributions file.")
            self.distributions = pd.read_json(self._distributions_path, orient="split")
        else:
            logger.error("Distributions file not found.")


    def get_dataset_by_url(self, url: str) -> dict | None:
        """Get complete dataset info by URL.

        Returns: a dictionary with keys:
            title: dataset title,
            url: dataset URL,
            text: description text of the dataset,
            distributions: list of distributions, each as a dict with keys: title, url, file_format
            categorization_metadata: dict with keywords, themes, categories, spatial_coverage,
            temporal_coverage, provider
        """
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
            'distributions': self.distributions[self.distributions['dataset_url'] == row['url']].to_dict(orient="records"),
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
        """Get a dict of metadata filters and the number of occurrences of each metadata value.

        Each filter column is a key, value is another dict with two keys: title and value_counts. Title is the filter
        category title, value_counts is a dict, where key is each unique value, value is number of occurrences."""
        if not self.filters_with_counts:
            self.filters_with_counts = {
                col_name: {
                    "title": col_name.replace("_", " ").capitalize(),
                    "value_counts": self.datasets[col_name].explode().value_counts(dropna=True).to_dict()
                }
                for col_name in self._filter_columns
            }
        return self.filters_with_counts
