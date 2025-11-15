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

import numpy as np
import ollama
import pandas as pd
import requests
import logging

import yaml

from rag import load_documents_to_rag_db
from query_prepocessing import query_preprocessing
from search import search
from result_postprocessing import result_postprocessing

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SearchPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.data_df = self._load_df(**config["data"]["datasets"])
        self._merge_dataset_rows_into_one_row()
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

    def _merge_dataset_rows_into_one_row(self) -> None:
        """Merge rows about the same dataset into one so that we have one row per dataset in data_df.

        Right now, there is a separate row for the same dataset for each keyword, theme, location, legal regulation
        and category associated with it. We want a single row per dataset with all keywords merged into one column.
        """

        columns_with_duplicates = ["klíčová_slova", "prostorové_pokrytí", "téma", "právní_předpis", "kategorie_hvd_název"]
        # columns that are not needed - e.g. tema_IRI when we also have column tema, je_součástí_IRI is always empty
        columns_to_be_dropped = ["kategorie_hvd_IRI", "je_součástí_IRI", "periodicita_aktualizace_IRI", "téma_IRI", "poskytovatel_IRI"]
        self.data_df = self.data_df.drop(columns=columns_to_be_dropped)

        for col in columns_with_duplicates:
            sub_df = self.data_df.groupby('datová_sada')[col].apply(lambda x: list(set(x))).reset_index()
            self.data_df = self.data_df.drop(columns=[col])
            self.data_df = pd.merge(self.data_df, sub_df, on='datová_sada', how='left')

        self.data_df = self.data_df.drop_duplicates(subset=['datová_sada'])

        self.data_df = self.data_df.applymap(lambda x: None if isinstance(x, list) and len(x) == 1 and pd.isna(x[0]) else x)
        self.data_df = self.data_df.replace(np.nan, None)
        logger.info(f"Number of rows: {self.data_df.shape[0]}.")

    def run(self, query: str) -> pd.DataFrame | None:
        """Run the search pipeline for the given query and return the results as a DataFrame."""

        load_documents_to_rag_db(self.config["rag"], self.data_df)
        expanded_query = query_preprocessing(query)
        results = search(expanded_query, self.data_df)
        results = result_postprocessing(results)
        # return just nazev and popis columns
        if results is not None:
            # results["datová_sada"] = results["datová_sada"].apply(lambda url: f'<a href="{url}" target="_blank">link</a>')
            return results[['název', 'popis', "datová_sada"]]
        return None


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

search_pipeline = SearchPipeline(config)
search_pipeline.run("Hi.")