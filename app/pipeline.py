"""
Search pipeline:
    1. Query Preprocessing
    2. Search
    3. Result Postprocessing
    4. Context
"""
import asyncio
from typing import List, Dict

import pandas as pd
import logging

import yaml
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

from context import Agent
from custom_ollama_embedding import CustomOllamaEmbedding
from data_processing.nkod_datasets import NKOD
from data_processing.rag import RAG
from query_prepocessing import query_preprocessing, detect_user_intent
from result_postprocessing import result_postprocessing
from search import SearchEngine



logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.DEBUG)


class SearchPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.dataset_portal = NKOD(config)
        self.super_df = self.dataset_portal.super_df
        self.llm = Ollama(model=self.config['rag']['llm']['model_name'], context_window=self.config['rag']['llm']['context_length'])
        Settings.llm = self.llm
        Settings.embed_model = CustomOllamaEmbedding(
            model_name=self.config['rag']['embedding']['model_name'],
            base_url=self.config['rag']['embedding']['base_url'],
            embed_batch_size=self.config['rag']['embedding']['embed_batch_size'],
        )

    async def run(self, query: str) -> List[Dict] | None:
        """Run the search pipeline for the given query and return the results as a DataFrame."""

        rag = RAG(self.config['rag'])
        rag.load_documents(self.dataset_portal.get_new_datasets())

        intent, alternative_queries = query_preprocessing(query)
        search_engine = SearchEngine(rag.index, rag.document_store)
        search_results = await search_engine.search(query, alternative_queries)
        agent = Agent(rag.index, search_engine.retriever, self.llm)
        await agent.run_chatbot()
        # results = result_postprocessing(results)
        # return just nazev and popis columns
        nodes = result_postprocessing(query, search_results, intent)
        if nodes is not None:
            return nodes
        return None


# with open("config.yaml", "r") as f:
#     config = yaml.safe_load(f)
#
# search_pipeline = SearchPipeline(config)
# asyncio.run(search_pipeline.run("Praha a její okolí."))