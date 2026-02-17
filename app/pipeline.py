"""
Search pipeline:
    1. Query Preprocessing
    2. Search
    3. Result Postprocessing
    4. Context
"""
import asyncio
import logging

import yaml
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from app.context import Agent
from data_processing.nkod_datasets import NKOD
from data_processing.database import Database
from app.query_prepocessing import query_preprocessing
from app.result_postprocessing import result_postprocessing, format_search_results
from app.search import SearchEngine
from app.result_postprocessing import rerank_with_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.DEBUG)


class SearchPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.dataset_portal = NKOD(config)
        self.llm = Ollama(model=self.config['rag']['llm']['model_name'], context_window=self.config['rag']['llm']['context_length'])
        Settings.llm = self.llm
        Settings.embed_model = OllamaEmbedding(
            model_name=self.config['rag']['embedding']['model_name'],
            base_url=self.config['rag']['embedding']['base_url'],
            embed_batch_size=self.config['rag']['embedding']['embed_batch_size'],
        )
        self.database = Database(self.config['rag'])


    async def run(self, query: str) -> list[dict] | None:
        """Run the search pipeline for the given query and return the results as a DataFrame."""

        intent, extended_query = query_preprocessing(query,
                                                          self.config['rag']["data_processing"]["categories"],
                                                          self.config['rag']["data_processing"]["other_category"])
        search_engine = SearchEngine(self.database.index, self.database.document_store)
        search_results = await search_engine.search(extended_query)
        # agent = Agent(self.database.index, search_engine.retriever, self.llm)
        # await agent.run_chatbot()
        # nodes = result_postprocessing(query, extended_query, search_results, intent)
        # nodes = rerank_with_llm(self.llm, query, search_results, intent)
        nodes = format_search_results(search_results)
        if nodes is not None:
            return nodes
        return None


# with open("config.yaml", "r") as f:
#     config = yaml.safe_load(f)
#
# search_pipeline = SearchPipeline(config)
# asyncio.run(search_pipeline.run("datasety o cukrovce"))