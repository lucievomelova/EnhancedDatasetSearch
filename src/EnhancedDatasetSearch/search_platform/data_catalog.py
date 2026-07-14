from abc import ABC, abstractmethod

import pandas as pd
from EnhancedDatasetSearch.data_processing.knowledge_graph import KnowledgeGraph


class DataCatalog(ABC):
    """Class representing a generic data catalog."""
    def __init__(self, config: dict, knowledge_graph: KnowledgeGraph):
        self.config = config

        self.datasets: pd.DataFrame
        """The dataset of datasets metadata that represents the data catalog used in our system."""

        self.distributions: pd.DataFrame
        """Dataset of distributions - links each dataset with all its available distributions."""

        self._filter_columns: list
        """Columns that can be used for search result filtering."""

        self.filters_with_counts: dict
        """Dictionary that stores occurrence counts of each unique value in every filter category."""

        self.knowledge_graph: KnowledgeGraph = knowledge_graph
        """Knowledge graph of the data catalog - contains dataset and metadata nodes. This knowledge is used to 
        detect similar datasets."""

        self.distributions: pd.DataFrame = pd.DataFrame()
        """Dataset of distributions - links each dataset with all its available distributions."""

        self.all_keywords: set
        """Set of all keywords present in the datasets metadata."""

        self.all_themes: set
        """Set of all themes present in the datasets metadata."""

        self.all_categories: set
        """Set of all categories (including "other" category)."""

        self.all_providers: set
        """Set of all providers of datasets at NKOD."""

        self.all_spatial_coverages: set
        """Set of all spatial_coverages used in the datasets."""

        self.all_temporal_coverages: set
        """Set of all temporal coverages used in the datasets."""

    @abstractmethod
    def get_dataset_by_url(self, url: str) -> dict | None:
        """Get dataset info by URL."""

    @abstractmethod
    def get_dataset_url_by_title(self, title: str) -> str | None:
        """Get dataset URL by its title. If multiple datasets have the same title, the first match will be returned."""

    @abstractmethod
    def get_filters_with_counts(self) -> dict:
        """Get a dict of metadata filters and the number of occurrences of each metadata value."""

    @abstractmethod
    def get_similar_datasets(self, dataset_url: str) -> dict[str, list[tuple[str, float]]]:
        """Get similar datasets based on the knowledge graph."""