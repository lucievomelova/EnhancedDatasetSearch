import pandas as pd
from data_processing.metadata import create_documents
from llama_index.core import Document


class DataCatalog:
    def __init__(self):
        self.datasets: pd.DataFrame = pd.DataFrame()
        """
        Dataset of datasets - contains all information about each dataset in the catalog. One row represents one dataset.
        
        For each dataset, we have the following information: title, description, url, 
        keywords, themes, provider, categories, region, time_periods
        """

        self.all_keywords: list | None = None
        """List of all keywords present in the datasets metadata."""

        self.all_themes: list | None = None
        """List of all themes present in the datasets metadata."""

        self.all_categories: list
        """List of all categories."""

        self.all_providers: list | None = None
        """List of all providers of datasets at NKOD."""

        self.all_spatial_coverages: list | None = None
        """List of all spatial_coverages used in the datasets."""

        self.all_temporal_coverages: list | None = None
        """List of all temporal coverages used in the datasets."""

        self.all_categories_with_other_category: list
        """List of all categories including "other" category used when a dataset does not belong into any category."""

    async def get_new_datasets(self) -> pd.DataFrame | None:
        pass

    def init(self):
        pass

    def prepare_documents_for_upload(self) -> list[Document]:
        """Get the list of llamaindex documents that should be uploaded to the knowledge base."""
        self.init()
        # new_datasets = asyncio.run(self.get_new_datasets())
        new_datasets = self.datasets  # upload all datasets to db
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

