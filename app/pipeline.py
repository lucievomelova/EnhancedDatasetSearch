"""
Search pipeline:
    1. Query Preprocessing
    2. Search
    3. Result Postprocessing
    4. Context
"""
import logging

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

from data_processing.data_catalogs.nkod import NkodDataCatalog
from data_processing.database import Database
from app.query_prepocessing import QueryPreprocessor
from app.result_postprocessing import PostProcessor
from app.retrieve import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.DEBUG)


class SearchPipeline:
    """The search pipeline class that handles searching the knowledge abse for relevant datasets

    The pipeline has the following steps:
    1. Query preprocessing: process the user query to extract user intent and extend the query.
    2. Retrieve: retrieve relevant datasets' documents from the database based on the preprocessed query.
    3. Result postprocessing: postprocess the retrieved documents into a list of search results."""

    def __init__(self, config: dict, llm: Ollama):
        self.config = config
        self.data_catalog = NkodDataCatalog(config)
        self.database = Database(self.config, self.config["state_dir"])
        self.index = self.database.index
        Settings.llm = llm
        self.query_preprocessor = QueryPreprocessor(config)
        self.retriever = Retriever(self.config["pipeline_config"]["search"],
                                self.index,
                                self.database.document_store)
        self.postprocessor = PostProcessor(self.config["pipeline_config"]["postprocessing"], self.data_catalog)


    async def run(self, query: str) -> list[dict[str, str | list | None]] | None:
        """Run the search pipeline for the given query and return the results as a DataFrame."""

        intent, extended_query = self.query_preprocessor.run(query,
                                                             self.config["data_processing"]["categories"],
                                                             self.config["data_processing"]["other_category"])
        search_results = await self.retriever.run(query, extended_query)
        nodes = self.postprocessor.run(query, extended_query, search_results, intent)
        if nodes is not None:
            return nodes
        return None


# with open("config.yaml", "r") as f:
#     config = yaml.safe_load(f)
#
# search_pipeline = SearchPipeline(config)
# asyncio.run(search_pipeline.run("volby, prezidentské volby, volby 2023, volba prezidenta"))