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

from data_processing.nkod_datasets import NKOD
from data_processing.database import Database
from app.query_prepocessing import query_preprocessing
from app.result_postprocessing import PostProcessor
from app.search import SearchEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.DEBUG)


class SearchPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.dataset_portal = NKOD(config)
        self.dataset_portal.init()
        self.llm = Ollama(model=self.config['rag']['llm']['model_name'], context_window=self.config['rag']['llm']['context_length'])
        Settings.llm = self.llm
        Settings.embed_model = OllamaEmbedding(
            model_name=self.config['rag']['embedding']['model_name'],
            base_url=self.config['rag']['embedding']['base_url'],
            embed_batch_size=self.config['rag']['embedding']['embed_batch_size'],
        )
        self.database = Database(self.config['rag'], self.config["state_dir"])


    async def run(self, query: str) -> list[dict[str, str | list | None]] | None:
        """Run the search pipeline for the given query and return the results as a DataFrame."""

        intent, extended_query = query_preprocessing(query,
                                                          self.config['rag']["data_processing"]["categories"],
                                                          self.config['rag']["data_processing"]["other_category"])
        search_engine = SearchEngine(self.config["rag"]["pipeline_config"]["search"], self.database.index, self.database.document_store)
        search_results = await search_engine.search(extended_query, extended_query)
        postprocessor = PostProcessor(self.config["rag"]["pipeline_config"]["postprocessing"])
        nodes = postprocessor.run(query, extended_query, search_results, intent)
        if nodes is not None:
            return nodes
        return None


# with open("config.yaml", "r") as f:
#     config = yaml.safe_load(f)
#
# search_pipeline = SearchPipeline(config)
# asyncio.run(search_pipeline.run("volby, prezidentské volby, volby 2023, volba prezidenta"))