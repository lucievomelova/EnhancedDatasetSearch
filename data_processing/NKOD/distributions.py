import os
from datetime import datetime

from data_processing.NKOD.utils import download_df
import pandas as pd

def download_distribution_info(path: str, url: str, column_mapping: dict, download_new_data: bool) -> pd.DataFrame:
    """Download distribution table from NKOD."""
    mod_time = os.path.getmtime(path)
    mod_datetime = datetime.fromtimestamp(mod_time)
    today = datetime.today().date()
    if os.path.exists(path) and mod_datetime.date() == today:  # distributions file is up to date
        distribution_df = pd.read_csv(path, sep=",", dtype="string")
    else:
        distribution_df = download_df(path, url)

    columns_to_keep = list(column_mapping.keys())
    distribution_df = distribution_df[columns_to_keep]
    distribution_df = distribution_df.rename(columns=column_mapping)
    return distribution_df


