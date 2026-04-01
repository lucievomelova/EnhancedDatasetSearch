import pandas as pd


class DataCatalog:
    def __init__(self):
        self.datasets: pd.DataFrame = pd.DataFrame()
        """
        Dataset of datasets - contains all information about each dataset in the catalog. One row represents one dataset.
        
        For each dataset, we have the following information: title, description, url, 
        keywords, themes, provider, categories, region, time_periods
        """

        self.all_keywords: list
        """List of all keywords present in the datasets metadata."""

        self.all_themes: list
        """List of all themes present in the datasets metadata."""

    async def get_new_datasets(self) -> pd.DataFrame | None:
        pass

    def init(self):
        pass
