import logging

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

from EnhancedDatasetSearch.data_catalog import DataCatalog
from EnhancedDatasetSearch.search.query_prepocessing import QueryPreprocessor
from EnhancedDatasetSearch.search.result_postprocessing import PostProcessor
from EnhancedDatasetSearch.search.retrieve import Retriever
from EnhancedDatasetSearch.database import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.DEBUG)


class SearchPipeline:
    """The search pipeline class that handles searching the knowledge abse for relevant datasets

    The pipeline has the following steps:
    1. Query preprocessing: process the user query to extract user intent and extend the query.
    2. Retrieve: retrieve relevant datasets' documents from the database based on the preprocessed query.
    3. Result postprocessing: postprocess the retrieved documents into a ranked list of search results.
    """

    def __init__(self, config: dict, llm: Ollama, data_catalog: DataCatalog, database: Database):
        self.config = config
        Settings.llm = llm
        self.query_preprocessor = QueryPreprocessor(config)
        self.retriever = Retriever(self.config["pipeline_config"]["search"], database)
        self.postprocessor = PostProcessor(self.config["pipeline_config"]["postprocessing"], data_catalog)

    async def run(self, query: str, applied_filters: dict | None = None) -> list[dict[str, str | list | None]]:
        """Run the search pipeline for the given query and return the results as a DataFrame."""
        _, extended_query = self.query_preprocessor.run(
            query,
            applied_filters,
            self.config["data_processing"]["categories"],
            self.config["data_processing"]["other_category"]
        )
        search_results = await self.retriever.run(query, extended_query, applied_filters)
        nodes = self.postprocessor.run(query, search_results)
        if nodes is not None:
            return nodes
        return []
