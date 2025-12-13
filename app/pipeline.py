"""
Search pipeline:
    1. Query Preprocessing
    2. Search
    3. Result Postprocessing
    4. Context
"""
from typing import List, Dict

import pandas as pd
import logging

import yaml
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

from custom_ollama_embedding import CustomOllamaEmbedding
from data_processing.nkod_datasets import NKOD
from data_processing.rag import RAG
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
        Settings.llm = Ollama(model=self.config['rag']['llm']['model_name'])
        Settings.embed_model = CustomOllamaEmbedding(
            model_name=self.config['rag']['embedding']['model_name'],
            base_url=self.config['rag']['embedding']['base_url'],
            embed_batch_size=self.config['rag']['embedding']['embed_batch_size'],
        )

    def run(self, query: str) -> List[Dict] | None:
        """Run the search pipeline for the given query and return the results as a DataFrame."""

        rag = RAG(self.config['rag'])
        rag.load_documents(self.dataset_portal.get_new_datasets())

        expanded_query = query_preprocessing(query)
        nodes = search(rag.index, rag.document_store, expanded_query)
        # results = result_postprocessing(results)
        # return just nazev and popis columns
        if nodes is not None:
            return nodes
        return None


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

search_pipeline = SearchPipeline(config)
search_pipeline.run("Praha a její okolí.")