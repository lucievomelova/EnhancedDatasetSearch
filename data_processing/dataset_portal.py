import asyncio
import pandas as pd
from data_processing.NKOD.generic_data_catalog import DataCatalog
from data_processing.metadata import create_documents
from llama_index.core import Document

from utils import setup_logger

logger = setup_logger(__name__)

class DatasetPortal:
    """Representing the Dataset Portal."""
    def __init__(self, data_catalog: DataCatalog):
        data_catalog.init()
        self.data_catalog: DataCatalog = data_catalog

        """
        Dataset of datasets - contains all information about each dataset in the catalog. One row represents one dataset.
        
        For each dataset, we have the following information: title, description, url, 
        keywords, themes, provider, categories, region, time_periods
        """

        # self.all_keywords: list = list(self.datasets["keywords"].explode().dropna().unique())
        # """List of all keywords present in the datasets metadata."""
        #
        # self.all_themes: list = list(self.datasets["themes"].explode().dropna().unique())
        # """List of all themes present in the datasets metadata."""

    def get_new_datasets(self) -> list[Document]:
        """Get the list of new datasets as llamaindex documents."""
        # new_datasets = asyncio.run(self.data_catalog.get_new_datasets())
        new_datasets = self.data_catalog.datasets  # upload all datasets to db
        documents = create_documents(new_datasets)
        return documents

    def get_dataset_by_url(self, url: str) -> dict | None:
        """Get dataset info by URL."""
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
                'region': row['region'] if isinstance(row['region'], list) else [],
                'time_periods': row['time_periods'] if isinstance(row['time_periods'], list) else [],
                'provider': row['provider'] if 'provider' in row and not pd.isna(row['provider']) else '',
            }
        }
