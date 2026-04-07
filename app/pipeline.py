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

from data_processing.NKOD.nkod_data_catalog import NkodDataCatalog
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
        self.data_catalog = NkodDataCatalog(config)
        self.data_catalog.init()
        self.database = Database(self.config, self.config["state_dir"])
        self.index = self.database.index


    async def run(self, query: str) -> list[dict[str, str | list | None]] | None:
        """Run the search pipeline for the given query and return the results as a DataFrame."""

        intent, extended_query = query_preprocessing(query,
                                                          self.config["data_processing"]["categories"],
                                                          self.config["data_processing"]["other_category"])
        search_engine = SearchEngine(self.config["pipeline_config"]["search"], self.index, self.database.document_store)
        search_results = await search_engine.search(query, extended_query)
        postprocessor = PostProcessor(self.config["pipeline_config"]["postprocessing"])
        nodes = postprocessor.run(query, extended_query, search_results, intent)
        if nodes is not None:
            return nodes
        return None


# with open("config.yaml", "r") as f:
#     config = yaml.safe_load(f)
#
# search_pipeline = SearchPipeline(config)
# asyncio.run(search_pipeline.run("volby, prezidentské volby, volby 2023, volba prezidenta"))