import pandas as pd
from llama_index.core import Document

from EnhancedDatasetSearch.utils import setup_logger

logger = setup_logger(__name__)


class DataCatalog:
    """Class representing a data catalog."""
    def __init__(self):
        self.datasets: pd.DataFrame = pd.DataFrame()
        """
        Dataset of datasets - contains metadata about each dataset in the catalog. One row represents one dataset.
        
        For each dataset, we have the following information: title, description, url, 
        keywords, themes, provider, categories, region, time_periods
        """

        self.distributions: pd.DataFrame = pd.DataFrame()
        """Dataset of distributions - links each dataset with all its available distributions."""

        self.all_keywords: set
        """Set of all keywords present in the datasets metadata."""

        self.all_themes: set
        """Set of all themes present in the datasets metadata."""

        self.all_categories: set
        """Set of all categories."""

        self.all_providers: set
        """Set of all providers of datasets at NKOD."""

        self.all_spatial_coverages: set
        """Set of all spatial_coverages used in the datasets."""

        self.all_temporal_coverages: set
        """Set of all temporal coverages used in the datasets."""

        self.all_categories_with_other_category: set
        """Set of all categories including "other" category used when a dataset does not belong into any category."""

        self._filter_columns: list
        self._filter_column_names: list
        self.filters_with_counts: dict


    async def update_datasets(self) ->  None:
        pass

    def prepare_documents_for_upload(self, datasets: pd.DataFrame) -> list[Document]:
        """Get the list of llamaindex documents that should be uploaded to the knowledge base."""
        if datasets.empty:
            return []
        documents = self._create_documents(datasets)
        return documents


    @staticmethod
    def _create_documents(datasets: pd.DataFrame) -> list[Document]:
        """Create llama index Documents from the dataframe. Each row will be used to create one Document."""
        pass

    def get_dataset_by_url(self, url: str) -> dict | None:
        """Get extended dataset info by URL."""
        pass

    def get_filters_with_counts(self) -> dict:
        """Get dict of filter columns.

        Each column is a key, value is another dict with two keys: title and value_counts. Title is the filter
        category title, value_counts is a dict, where key is each unique value, value is number of occurrences."""
        pass