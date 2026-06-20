import os

from data_processing.data_catalogs.df_utils import download_df
import pandas as pd

def download_distribution_info(path: str, url: str, column_mapping: dict, download_new_data: bool) -> pd.DataFrame:
    """Download distribution table from NKOD."""
    if download_new_data or not os.path.exists(path):
        distribution_df = download_df(path, url)
    else:
        distribution_df = pd.read_csv(path, sep=",", dtype="string")
    columns_to_keep = list(column_mapping.keys())
    distribution_df = distribution_df[columns_to_keep]
    distribution_df = distribution_df.rename(columns=column_mapping)
    return distribution_df


