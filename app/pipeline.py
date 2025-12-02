"""
Search pipeline:
    1. Query Preprocessing
    2. Search
    3. Result Postprocessing
    4. Context
"""
import pandas as pd
import logging

import yaml

from nkod_datasets import NKOD
from rag import RAG
from query_prepocessing import query_preprocessing
from search import search
from result_postprocessing import result_postprocessing

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SearchPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.dataset_portal = NKOD(config)
        self.super_df = self.dataset_portal.super_df

    def run(self, query: str) -> pd.DataFrame | None:
        """Run the search pipeline for the given query and return the results as a DataFrame."""

        rag = RAG(self.config['rag'])
        rag.load_documents(self.dataset_portal.get_new_datasets())

        expanded_query = query_preprocessing(query)
        results = search(rag.index, expanded_query, self.super_df)
        results = result_postprocessing(results)
        # return just nazev and popis columns
        if results is not None:
            # results["datová_sada"] = results["datová_sada"].apply(lambda url: f'<a href="{url}" target="_blank">link</a>')
            return results[['název', 'popis', "datová_sada"]]
        return None


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

search_pipeline = SearchPipeline(config)
search_pipeline.run("Praha a její okolí.")