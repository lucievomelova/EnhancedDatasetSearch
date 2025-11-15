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

from app.nkod_datasets import get_informative_dataset
from rag import load_documents_to_rag_db
from query_prepocessing import query_preprocessing
from search import search
from result_postprocessing import result_postprocessing

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SearchPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.informative_df = get_informative_dataset(config)


    def run(self, query: str) -> pd.DataFrame | None:
        """Run the search pipeline for the given query and return the results as a DataFrame."""

        load_documents_to_rag_db(self.config["rag"], self.informative_df)
        expanded_query = query_preprocessing(query)
        results = search(expanded_query, self.informative_df)
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