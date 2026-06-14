import json

from data_processing.data_catalogs.data_catalog import DataCatalog
import data_processing.knowledge_graph as kg
from llama_index.core.agent import AgentWorkflow
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool

from app.pipeline import SearchPipeline
from utils import setup_logger, render_template, dataset_detail_url, get_nkod_url

logger = setup_logger(__name__)


class Chatbot:
    def __init__(self, config: dict, search_pipeline: SearchPipeline, llm: FunctionCallingLLM):
        self.config = config
        self.search_pipeline = search_pipeline
        self.data_catalog = search_pipeline.data_catalog
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=config["chatbot"]["llm"]["memory_limit"])
        tools = self.create_tools()
        system_prompt = render_template("chatbot/chat.j2")
        self.agent = FunctionAgent(
            llm=llm,
            tools=tools,
            verbose=True,
            memory=self.memory,
            system_prompt=system_prompt,
        )


    def create_tools(self) -> list[FunctionTool]:
        search_in_data_catalog = FunctionTool.from_defaults(
            async_fn=self._tool_search,
            name="search_in_data_catalog",
            description="Search the data catalog for most relevant datasets based on th search query." +
                        "Input should be an effective search query about what datasets the user is looking for."
        )
        get_list_of_all_values_for_metadata_category = FunctionTool.from_defaults(
            fn=self._tool_get_list_of_all_values_for_metadata_category,
            name="get_list_of_all_values_for_metadata_category",
            description="Get a list of all values occurring in the data catalog for a given metadata category. " +
            "Possible category values are: keywords, themes, categories, providers, spatial_coverages, temporal_coverages."
        )
        extract_data_catalog_information = FunctionTool.from_defaults(
            fn=self._tool_read_info_file,
            name="get_general_information_from_file",
            description="Get general information about how the data catalog works from an information file. "+
            "Specify the type of information needed, possible values are:\n" +
            "- metadata: to get information about the metadata structure.\n" +
            "- NKOD: to get general information about the Czech national open data catalog NKOD."
        )
        get_dataset_info = FunctionTool.from_defaults(
            fn=self._tool_get_dataset_info,
            name="get_dataset_info",
            description="Get dataset information based on the dataset URL. "+
            "Returns a dict, where key is the type of information (e.g. dataset title, description or some metadata) " +
            "and value is the associated information."
        )

        get_similar_datasets = FunctionTool.from_defaults(
            fn=self._tool_get_similar_datasets,
            name="get_similar_datasets",
            description="Get datasets similar to the given dataset (based on its URL). "+
            "Optionally, a type of similarity can be specified - 'description', 'keywords' or themes'. " +
            "If no category is specified, returns a dict with similarity type as key and similar datasets as value. " +
            "Otherwise returns a list of similar datasets for the specified category."
        )

        get_nkod_url = FunctionTool.from_defaults(
            fn=self._tool_get_nkod_url,
            name="get_nkod_url",
            description="Get link to the dataset detail page on NKOD on the Czech Dataset Portal."
        )

        return [search_in_data_catalog,
                get_list_of_all_values_for_metadata_category,
                extract_data_catalog_information,
                get_dataset_info,
                get_similar_datasets,
                get_nkod_url]

    async def react_to_message(self, user_message: str) -> str:
        """Run the chatbot in a loop."""
        error_message = "There was an error during answer generation. Try rewriting your message or start a new chat."
        logger.info("Reacting.")
        try:
            response = str(await self.agent.run(user_msg=user_message, memory=self.memory))
            logger.info(f"Response: {response}")
            if not response:  # there was an error during response generation
                return error_message
            return response
        except Exception as e:
            logger.error(f"Error in chatbot response generation: {str(e)}", exc_info=True)
            return error_message

    def _tool_get_list_of_all_values_for_metadata_category(self, metadata_category: str) -> list[str] | str:
        """Get a list of all values for a given metadata category.

        Possible category values are: keywords, themes, categories, providers, spatial_coverages, temporal_coverages."""
        possible_categories = ["keywords", "themes", "categories", "region", "time_periods"]
        try:
            if metadata_category == "categories":
                return self.data_catalog.all_categories_with_other_category
            return getattr(self.data_catalog, f"all_{metadata_category}")
        except AttributeError:
            return f"Invalid metadata category, possible values are: {possible_categories}"

    def _tool_read_info_file(self, info_type: str) -> str:
        """Read the content of an information file and return it as a string."""
        try:
            with open(f"/infromation_texts/{info_type}.md", "r") as f:
                return f.read()
        except FileNotFoundError:
            return "Invalid information type. No such information file. Possible values are: metadata, NKOD."

    async def _tool_search(self, query: str) -> str | list:
        """Search the data catalog for relevant datasets based on the query.

        Returns:
            a of dicts, where each item represents one dataset. For each dataset there is its title and URL."""
        try:
            search_results = await self.search_pipeline.run(query)
            formatted_results = self._keep_only_title_and_url(search_results)
            return formatted_results
        except Exception as e:
            logger.error(f"Error during search in data catalog: {str(e)}", exc_info=True)
            return "An error occurred during search in data catalog."

    def _tool_get_dataset_info(self, dataset_url: str) -> dict | str:
        """Get detailed information about a dataset based on its URL."""
        try:
            dataset_info = self.data_catalog.get_dataset_by_url(dataset_url)
            if dataset_info is None:
                return "No dataset found with the given URL."
            dataset_info["url"] = dataset_detail_url(self.config, dataset_url)
            return dataset_info
        except Exception as e:
            logger.error(f"Error during getting dataset info: {str(e)}", exc_info=True)
            return "An error occurred during retrieving dataset information."

    def _tool_get_similar_datasets(self, dataset_url: str, similarity_type: str | None = None) -> list | dict | str:
        """Get similar datasets for a given dataset based on its URL."""
        try:
            similar_datasets = kg.get_similar_datasets(dataset_url, self.config["data_processing"]["knowledge_graph"]["top_k"], similarity_type)
            results = {}
            for sim_type, url_score_list in similar_datasets.items():
                results_per_sim_type = []
                for url, _ in url_score_list:
                    sim_dataset = self.search_pipeline.data_catalog.get_dataset_by_url(url)
                    results_per_sim_type.append({
                        "title": sim_dataset["title"],
                        "url": dataset_detail_url(self.config, sim_dataset["url"])
                    })
                results[sim_type] = results_per_sim_type

            if similarity_type is not None:
                similar_datasets = results[similarity_type]
            return similar_datasets
        except Exception as e:
            logger.error(f"Error during getting similar datasets: {str(e)}", exc_info=True)
            return "An error occurred during retrieving similar datasets."

    def _tool_get_nkod_url(self, dataset_url: str) -> str:
        """Get the NKOD url for the given dataset URL."""
        return get_nkod_url(self.config, dataset_url)

    def _keep_only_title_and_url(self, search_results: list[dict] | None) -> list[dict[str, str]]:
        """Keep only title of a dataset and its url to the dataset_detail page.

        Returns:
            Processed search results as a list of dicts."""
        if search_results is None:
            return []
        processed_results = []
        for result in search_results:
            processed_results.append({
                "title": result["title"],
                "url": dataset_detail_url(self.config, result["url"])
            })
        return processed_results
