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

class QueryPreprocessor:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.client = OllamaClient(config["llm"], timeout=config["pipeline_config"]["preprocessing"]["timeout"])

    def run(self, user_query: str, categories: list[str], other_category: str) -> (dict, str):
        """Preprocess the user query."""

        logger.info("Preprocessing user query: %s", user_query)
        intent = self.detect_user_intent(user_query, categories, other_category)
        extended_query = self.extend_user_query(user_query)
        return intent, extended_query


    def extend_user_query(self, user_query: str) -> str:
        template = env.get_template("rewrite_query.j2")
        prompt = template.render(intro=intro_prompt, user_query=user_query)
        logger.info(f"Extending user query: {user_query}")

        try:
            extended_query = self.client.get_llm_response(prompt)
        except httpx.ReadTimeout as e:
            logger.error(f"Error while extending query: {e}, the original user query will be used isntead.")
            extended_query = user_query

        logger.info(f"Extended query: {extended_query}")
        return extended_query


    def detect_user_intent(self, user_query: str, categories: list[str], other_category: str) -> dict[str, str]:
        """Detect the user intent from the query using LLM."""
        logger.info("Detecting intent for query: %s", user_query)

        num_categories = 2
        template = env.get_template("detect_user_intent.j2")
        prompt = template.render(intro=intro_prompt,
                                 user_query=user_query,
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
