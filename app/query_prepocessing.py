import json

import httpx
from ollama import Client
from ollama_client import OllamaClient
from utils import setup_logger
from jinja2 import Environment, FileSystemLoader

logger = setup_logger(__name__)
env = Environment(loader=FileSystemLoader('prompts'))
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
    def __init__(self, config: dict) -> None:
        self.config = config
        self.client = OllamaClient(config["llm"], timeout=config["pipeline_config"]["preprocessing"]["timeout"])

    def run(self, user_query: str, applied_filters: dict | None, categories: list[str], other_category: str) -> (dict, str):
        """Preprocess the user query."""
        logger.info(f"Preprocessing user query: {user_query} with metadata filters: {applied_filters}")
        intent = self.detect_user_intent(user_query, applied_filters, categories, other_category)
        extended_query = self.extend_user_query(user_query, applied_filters, intent)
        return intent, extended_query

    def extend_user_query(self, user_query: str, applied_filters: dict | None, intent: dict) -> str:
        template = env.get_template("rewrite_query.j2")
        prompt = template.render(intro=intro_prompt,
                                 user_query=user_query,
                                 applied_filters=applied_filters,
                                 filters_selected=_metadata_filters_selected(applied_filters),
                                 categories_intent=intent["categories"],
                                 spatial_intent=intent["spatial_coverage"],
                                 temporal_intent=intent["temporal_coverage"])
        logger.info(f"Extending user query: {user_query}")

        try:
            extended_query = self.client.get_llm_response(prompt)
        except httpx.ReadTimeout as e:
            logger.error(f"Error while extending query: {e}, the original user query will be used isntead.")
            extended_query = user_query

        logger.info(f"Extended query: {extended_query}")
        return extended_query

    def detect_user_intent(
            self,
            user_query: str,
            applied_filters: dict | None,
            categories: list[str],
            other_category: str
    ) -> dict[str, str]:
        """Detect the user intent from the query using LLM."""
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
        except httpx.ReadTimeout as e:
            logger.error(f"Error while detecting user intent query: {e}, this step will be skipped.")
            response = {
                "categories": [],
                "spatial_coverage": [],
                "temporal_coverage": []
            }
        if "```" in response:
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[len("json"):].strip()
        intent = json.loads(response)
        logger.info(f"Detected intent: {intent}")

        return intent
