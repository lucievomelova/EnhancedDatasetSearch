import os

import pandas as pd

from data_processing.NKOD.utils import download_df


def download_distribution_info(path: str, url: str, column_mapping: dict, download_new_data: bool) -> pd.DataFrame:
    """Download distribution table from NKOD."""
    if not download_new_data and os.path.exists(path):
        distribution_df = pd.read_csv(path, sep=",", dtype="string")
    else:
        distribution_df = download_df(path, url)

    columns_to_keep = list(column_mapping.keys())
    distribution_df = distribution_df[columns_to_keep]
    distribution_df = distribution_df.rename(columns=column_mapping)
    return distribution_df
