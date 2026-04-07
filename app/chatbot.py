import json

from data_processing.NKOD.data_catalog import DataCatalog
from llama_index.core.agent import AgentWorkflow
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool

from app.pipeline import SearchPipeline
from utils import setup_logger, render_template

logger = setup_logger(__name__)


class Chatbot:
    def __init__(self, config: dict, search_pipeline: SearchPipeline, llm: FunctionCallingLLM):
        self.search_pipeline = search_pipeline
        self.data_catalog = search_pipeline.data_catalog
        self.dataset_detail_url = config["url"] + "dataset"
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
            async_fn=self.search_pipeline.run,
            name="search_in_data_catalog",
            description="Search the data catalog for most relevant datasets based on th search query."
        )
        get_list_of_all_values_for_metadata_category = FunctionTool.from_defaults(
            fn=self._tool_get_list_of_all_values_for_metadata_category,
            name="get_list_of_all_values_for_metadata_category",
            description="Get a list of all values occurring in the data catalog for a given metadata category." +
            "Possible category values are: keywords, themes, categories, providers, spatial_coverages, temporal_coverages."
        )
        extract_data_catalog_information = FunctionTool.from_defaults(
            fn=self._tool_read_info_file,
            name="get_general_information_from_file",
            description="Get general information about how the data catalog works from an information file."+
            "Specify the type of information needed, possible values are:\n" +
            "- metadata: to get information about the metadata structure.\n" +
            "- NKOD: to get general information about the Czech national open data catalog NKOD."
        )
        keep_only_title_and_url = FunctionTool.from_defaults(
            fn=self._tool_keep_only_title_and_url,
            name="keep_only_title_and_url",
            description="Keep only title of returned datasets and their url link to the dataset_detail page."+
            "This tool is used after search in data catalog is performed, to format the response to the user."
        )

        return [search_in_data_catalog,
                get_list_of_all_values_for_metadata_category,
                extract_data_catalog_information,
                keep_only_title_and_url]

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

    def _tool_keep_only_title_and_url(self, search_results: list[dict]) -> str:
        """Keep only title of a dataset and its url to the dataset_detail page.

        Returns:
            Processed search results, already converted to json string."""
        processed_results = []
        for result in search_results:
            processed_results.append({
                "title": result["title"],
                "url": self.dataset_detail_url + "/" + result["url"]
            })
        return json.dumps(processed_results)
