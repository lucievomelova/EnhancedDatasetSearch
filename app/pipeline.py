"""
Search pipeline:
    1. Query Preprocessing
    2. Search
    3. Result Postprocessing
    4. Context

There will be a RAG database containing info about all datasets. Every day, the new datove_sady and distribuce csvs
will be downloaded and if there are changes detected in some datasets at NKOD, their info will be deleted from DB and
then added again.

"""
import re
from datetime import datetime
import os

import ollama
import pandas as pd
import requests
import logging
from query_prepocessing import query_preprocessing
from search import search
from result_postprocessing import result_postprocessing

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SearchPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.data_df = self._load_df(**config["data"]["datasets"])
        self.distribution_df = self._load_df(**config["data"]["distribution"])

    @staticmethod
    def _load_df(path: str, url: str) -> pd.DataFrame:
        """Load dataset from csv and return it as a pandas dataframe.
        Check if csv file exists and is up to date - if not, download it again, load it and return the loaded df.
        """
        response = requests.get(url)
        if os.path.exists(path):
            mod_time = os.path.getmtime(path)
            mod_datetime = datetime.fromtimestamp(mod_time)
            today = datetime.today().date()
        if not os.path.exists(path) or mod_datetime.date() != today:
            with open(path, "wb") as f:
                logger.info(f"Downloading {path.split("/")[-1]}.")
                f.write(response.content)
        df = pd.read_csv(path, sep=",")
        logger.info("Dataset info loaded.")
        return df


    def run(self, query: str) -> pd.DataFrame | None:
        """Run the search pipeline for the given query and return the results as a DataFrame."""

        expanded_query = query_preprocessing(query)
        results = search(expanded_query, self.data_df)
        results = result_postprocessing(results)
        # return just nazev and popis columns
        if results is not None:
            # results["datová_sada"] = results["datová_sada"].apply(lambda url: f'<a href="{url}" target="_blank">link</a>')
            return results[['název', 'popis', "datová_sada"]]
        return None
