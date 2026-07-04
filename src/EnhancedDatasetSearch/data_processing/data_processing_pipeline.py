import pandas as pd
from llama_index.core import Document

from EnhancedDatasetSearch.utils import setup_logger

logger = setup_logger(__name__)


class DataProcessingPipeline:
    """Class representing a data catalog."""
    def __init__(self):
        self.datasets: pd.DataFrame = pd.DataFrame()
        """
        Dataset of datasets - contains metadata about each dataset in the catalog. One row represents one dataset.
        
        For each dataset, we have the following information: title, description, url, 
        keywords, themes, provider, categories, region, time_periods
        """


    async def update_datasets(self) -> (pd.DataFrame, list):
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
