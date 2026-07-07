from abc import ABC, abstractmethod

import pandas as pd

from EnhancedDatasetSearch.utils import setup_logger

logger = setup_logger(__name__)


class DatasetProcessingPipeline(ABC):
    """Class representing the generic dataset processing pipeline.

    This pipeline processes all datasets in the data catalog.
    The main output of this pipeline is the self.datasets file, that is stored in memory. It is the transformed
    and enhanced metadata dataset of the data catalog, where one row represents one dataset."""
    def __init__(self):
        self.datasets: pd.DataFrame = pd.DataFrame()
        """
        Dataset of datasets - contains metadata about each dataset in the catalog. One row represents one dataset.
        
        For each dataset, we have the following information: title, description, url, 
        keywords, themes, provider, categories, region, time_periods
        """

    @abstractmethod
    async def update_datasets(self) -> tuple[pd.DataFrame, list]:
        """Update self.datasets and find and return new or updated datasets and removed datasets.

        Returns:
            a tuple: (pd.DataFrame, list), where the dataframe contains new or updated datasets in hte same format as
            self.dataset. The list contains a list of removed datasets URLs.
        """
