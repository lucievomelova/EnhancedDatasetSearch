import pandas as pd


class DataCatalog:
    """Class representing a generic data catalog."""
    def __init__(self, config: dict):
        self.config = config

        self.datasets: pd.DataFrame
        """The dataset of datasets metadata that represents the data catalog used in our system."""

        self.distributions: pd.DataFrame
        """Dataset of distributions - links each dataset with all its available distributions."""

        self._filter_columns: list
        """Columns that can be used for search result filtering."""

        self.filters_with_counts: dict
        """Dictionary that stores occurrence counts of each unique value in every filter category."""

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


    def get_dataset_by_url(self, url: str) -> dict | None:
        """Get dataset info by URL."""
        pass

    def get_filters_with_counts(self) -> dict:
        """Get a dict of metadata filters and the number of occurrences of each metadata value.

        Each filter column is a key, value is another dict with two keys: title and value_counts. Title is the filter
        category title, value_counts is a dict, where key is each unique value, value is number of occurrences."""
        pass


