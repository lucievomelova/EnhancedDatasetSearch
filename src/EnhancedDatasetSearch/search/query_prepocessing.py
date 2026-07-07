import json

import httpx
from jinja2 import Environment, FileSystemLoader

from EnhancedDatasetSearch.ollama_client import OllamaClient
from EnhancedDatasetSearch.utils import setup_logger

logger = setup_logger(__name__)
env = Environment(loader=FileSystemLoader('llm_inputs/prompts'))
intro_template = env.get_template("intro.j2")
intro_prompt = intro_template.render()


def _metadata_filters_selected(applied_filters: dict | None) -> bool:
    """Check if any metadata filters were selected."""
    if not applied_filters:
        return False
    for filter_category, filters in applied_filters.items():
        if filters:
            return True  # at least one list not empty
    return False  # all lists empty


class QueryPreprocessor:
    """Search query preprocessor.

    Based on the config, the preprocessor will:
    1. Detect intent behind the user query. The detection is done by an LLM.
    2. Extend te user query to increase the chance of finding relevant datasets. The extension is done by an LLM.

    The steps are performed only if they are set to true in the preprocessing config."""
    def __init__(self, config: dict) -> None:
        self.config: dict = config
        self.client: OllamaClient = OllamaClient(config["pipeline_config"]["llm"], timeout=config["pipeline_config"]["preprocessing"]["timeout"])

    def run(self, user_query: str, applied_filters: dict | None, categories: list[str], other_category: str) -> tuple[dict | None, None]:
        """Preprocess the search query.

        Preprocessing steps:
        1. Detect intent behind the search query.
        2. Extend the search query.
        The steps are performed only if they are set to true in the config.
        """
        logger.info(f"Preprocessing user query: {user_query} with metadata filters: {applied_filters}")

        if self.config["pipeline_config"]["preprocessing"]["detect_intent"]:
            intent = self.detect_user_intent(user_query, applied_filters, categories, other_category)
        else:
            intent = None
        if self.config["pipeline_config"]["preprocessing"]["extend_query"]:
            extended_query = self.extend_user_query(user_query, applied_filters, intent)
        else:
            extended_query = None
        return intent, extended_query

    def extend_user_query(self, user_query: str, applied_filters: dict | None, intent: dict | None) -> str:
        """Extend the search query to increase the chance of finding relevant datasets."""
        template = env.get_template("rewrite_query.j2")
        prompt = template.render(intro=intro_prompt,
                                 user_query=user_query,
                                 applied_filters=applied_filters,
                                 filters_selected=_metadata_filters_selected(applied_filters),
                                 intent_detected=True if intent is not None else False,
                                 categories_intent=intent["categories"] if intent is not None else [],
                                 spatial_intent=intent["spatial_coverage"] if intent is not None else [],
                                 temporal_intent=intent["temporal_coverage"] if intent is not None else [])
        logger.info(f"Extending user query: {user_query}")

        try:
            extended_query = self.client.get_llm_response(prompt)
        except httpx.ReadTimeout as e:
            logger.error(f"Error while extending query: {e}, the original user query will be used instead.")
            extended_query = user_query

        logger.info(f"Extended query: {extended_query}")
        return extended_query

    def detect_user_intent(
            self,
            user_query: str,
            applied_filters: dict | None,
            categories: list[str],
            other_category: str
    ) -> dict[str, list]:
        """Detect the user intent from the query using LLM.

        Intent detection is done for the following metadata categories:
        categories, spatial_coverage and temporal_coverage."""
        logger.info("Detecting intent for query: %s", user_query)

        num_categories = 2
        template = env.get_template("detect_user_intent.j2")
        prompt = template.render(intro=intro_prompt,
                                 user_query=user_query,
                                 applied_filters=applied_filters,
                                 filters_selected=_metadata_filters_selected(applied_filters),
                                 num_categories=num_categories,
                                 categories=", ".join(categories),
                                 other_category=other_category)
        try:
            response = self.client.get_llm_response(prompt)
            if "```" in response:
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[len("json"):].strip()
            intent = json.loads(response)
        except (httpx.ReadTimeout, json.decoder.JSONDecodeError) as e:
            logger.error(f"Error while detecting user intent for query: {e}, this step will be skipped.")
            intent = None
        logger.info(f"Detected intent: {intent}")
        return intent
