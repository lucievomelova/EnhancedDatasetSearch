from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool

from EnhancedNkodDatasetSearch.search_platform.data_catalog import DataCatalog
from EnhancedNkodDatasetSearch.search_platform.search_pipeline.pipeline import SearchPipeline
from EnhancedNkodDatasetSearch.utils import dataset_detail_url, render_template, setup_logger

logger = setup_logger(__name__)


class Chatbot:
    """The chatbot class.

    This class handles the chatbot backend logic - message generation and tool calling.
    The chatbot is powered by llama_index FunctionAgent."""

    def __init__(
            self,
            config: dict,
            search_pipeline: SearchPipeline,
            data_catalog: DataCatalog,
            llm: FunctionCallingLLM,
            language: str):
        self.config: dict = config
        self.search_pipeline: SearchPipeline = search_pipeline
        self.data_catalog: DataCatalog = data_catalog
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=config["search_platform"]["llm"]["memory_limit"])
        self.language: str = language
        tools = self.create_tools()
        system_prompt = render_template("chat.j2", {"language": language})
        self.agent = FunctionAgent(
            llm=llm,
            tools=tools,
            verbose=True,
            memory=self.memory,
            system_prompt=system_prompt,
        )

    def create_tools(self) -> list[FunctionTool]:
        """Create tools for the chatbot."""
        search_in_data_catalog = FunctionTool.from_defaults(
            async_fn=self._tool_search,
            name="search_in_data_catalog",
            description="Search in the data catalog - retrieve relevant datasets based on the provided search query."
        )
        get_list_of_all_values_for_metadata_category = FunctionTool.from_defaults(
            fn=self._tool_get_list_of_all_values_for_metadata_category,
            name="get_list_of_all_values_for_metadata_category",
            description="Get a list of all values occurring in the data catalog for a given metadata category. " +
            "Possible category values are: themes, categories, providers, spatial_coverages, temporal_coverages."
        )
        get_data_catalog_information = FunctionTool.from_defaults(
            fn=self._tool_read_info_file,
            name="get_general_information_from_file",
            description="Get general information about how the data catalog works from an information file. "+
            "Specify the type of information needed, possible values are:\n" +
            "- metadata: information about the metadata structure.\n" +
            "- data_catalog: general information about the used data catalog.\n" +
            "- app: information about this system."
        )
        get_dataset_info = FunctionTool.from_defaults(
            fn=self._tool_get_dataset_info,
            name="get_dataset_info",
            description="Get dataset information based on the dataset URL or title. "+
            "Returns a dict, where key is the type of metadata (e.g. dataset title or description) " +
            "and value is the associated metadata."
        )

        return [
            search_in_data_catalog,
            get_list_of_all_values_for_metadata_category,
            get_data_catalog_information,
            get_dataset_info
        ]

    async def react_to_message(self, user_message: str) -> str:
        """Run the chatbot in a loop."""
        error_message = "There was an error during answer generation. Try rewriting your message or start a new chat."
        logger.info(f"Generating a response, language: {self.language}.")
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

        Possible category values are: themes, categories, providers, spatial_coverages, temporal_coverages."""
        logger.info(f"Tool call: get_list_of_all_values_for_metadata_category, metadata_category: {metadata_category}")
        possible_categories = ["themes", "categories", "region", "time_periods", "providers"]
        try:
            return getattr(self.data_catalog, f"all_{metadata_category}")
        except AttributeError:
            return f"Invalid metadata category, possible values are: {possible_categories}"

    def _tool_read_info_file(self, info_type: str) -> str:
        """Read the content of an information file and return it as a string."""
        logger.info(f"Tool call: read_info_file, file: {info_type}")
        try:
            with open(f"llm_inputs/nkod_context/{info_type}.md", "r") as f:
                content = f.read()
                return content
        except FileNotFoundError as e:
            logger.error(f"Error while trying to read information file: {str(e)}", exc_info=True)
            return "Invalid information file. Possible values are: metadata, data_catalog, app."

    async def _tool_search(self, query: str) -> str | list:
        """Search the data catalog for relevant datasets based on the query.

        Returns:
            a of dicts, where each item represents one dataset. For each dataset there is its title and URL."""
        logger.info(f"Tool call: search, query: {query}")
        try:
            search_results = await self.search_pipeline.run(query)
            formatted_results = self._keep_only_title_and_url(search_results)
            return formatted_results
        except Exception as e:
            logger.error(f"Error during search in data catalog: {str(e)}", exc_info=True)
            return "An error occurred during search in data catalog."

    def _tool_get_dataset_info(self, url: str | None = None, title: str | None = None) -> dict | str:
        """Get detailed information about a dataset based on its URL."""
        logger.info(f"Tool call: get_dataset_info, url: {url}, title: {title}")
        try:
            if url is None:
                url = self.data_catalog.get_dataset_url_by_title(title)
            dataset_info = self.data_catalog.get_dataset_by_url(url)
            if dataset_info is None:
                return "No dataset found with the given URL."
            dataset_info["url"] = dataset_detail_url(self.config, url)
            return dataset_info
        except Exception as e:
            logger.error(f"Error during getting dataset info: {str(e)}", exc_info=True)
            return "An error occurred during retrieving dataset information. You must specify title or URL."

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
