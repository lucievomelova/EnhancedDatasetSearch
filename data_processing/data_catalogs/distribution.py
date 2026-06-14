from data_processing.data_catalogs.df_utils import download_df
import pandas as pd

def download_distribution_info(path: str, url: str, column_mapping: dict) -> pd.DataFrame:
    """Download distribution table from NKOD."""
    distribution_df = download_df(path, url)
    columns_to_keep = list(column_mapping.keys())
    print(column_mapping)
    distribution_df = distribution_df[columns_to_keep]
    distribution_df = distribution_df.rename(columns=column_mapping)
    return distribution_df


