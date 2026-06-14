import asyncio

import pandas as pd
from data_processing.metadata import create_documents
from llama_index.core import Document
from utils import setup_logger, dataset_detail_url

logger = setup_logger(__name__)


class DataCatalog:
    def __init__(self):
        self.datasets: pd.DataFrame = pd.DataFrame()
        """
        Dataset of datasets - contains all information about each dataset in the catalog. One row represents one dataset.
        
        For each dataset, we have the following information: title, description, url, 
        keywords, themes, provider, categories, region, time_periods
        """

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

    async def get_new_datasets(self) -> pd.DataFrame | None:
        pass

    def prepare_documents_for_upload(self, new_datasets: pd.DataFrame) -> list[Document]:
        """Get the list of llamaindex documents that should be uploaded to the knowledge base."""
        if new_datasets.empty:
            return []
        documents = create_documents(new_datasets)
        return documents

    def get_dataset_by_url(self, url: str) -> dict | None:
        """Get extended dataset info by URL."""
        dataset_row = self.datasets[self.datasets['url'] == url]
        logger.info(url)
        if dataset_row.empty:  # try also the url used on dataset detail page
            dataset_row = self.datasets[dataset_detail_url(self.config, url) == url]
        if dataset_row.empty:
            return None  # still no result -> dataset with the given URL not found

        row = dataset_row.iloc[0]
        return {
            'title': row['title'],
            'url': row['url'],
            'text': row['description'] if pd.notna(row['description']) else "",
            'metadata': {
                'keywords': row['keywords'] if 'keywords' in row and isinstance(row['keywords'], list) else [],
                'themes': row['themes'] if 'themes' in row and isinstance(row['themes'], list) else [],
                'categories': row['categories'] if 'categories' in row and isinstance(row['categories'], list) else [],
                'spatial_coverage': row['spatial_coverage'] if 'spatial_coverage' in row and isinstance(row['spatial_coverage'], list) else [],
                'temporal_coverage': row['temporal_coverage'] if 'temporal_coverage' in row and isinstance(row['temporal_coverage'], list) else [],
                'provider': row['provider'] if 'provider' in row and not pd.isna(row['provider']) else '',
            }
        }
