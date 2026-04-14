import json

import ollama
from ollama import Client
from utils import setup_logger
from jinja2 import Environment, FileSystemLoader

logger = setup_logger(__name__)
env = Environment(loader=FileSystemLoader('prompts'))
intro_template = env.get_template("intro.j2")
intro_prompt = intro_template.render()

client = Client(
        host='http://localhost:11434',
        timeout=10
    )

def query_preprocessing(user_query: str, categories: list[str], other_category: str) -> (str, list):
    """Preprocess the user query."""

    logger.info("User query: %s", user_query)
    intent = detect_user_intent(user_query, categories, other_category)
    # alternative_queries = get_alternative_queries(user_query, intent)
    # alternative_queries = []
    extended_query = extend_user_query(user_query)

    return intent, extended_query


def extend_user_query(user_query: str) -> str:
    template = env.get_template("rewrite_query.j2")
    prompt = template.render(intro=intro_prompt, user_query=user_query)
    logger.info(f"Extending user query: {user_query}")

    # extended_query = ollama.generate(model='mistral-small3.2', prompt=prompt).response
    extended_query = client.generate(model='mistral-small3.2', prompt=prompt).response


    logger.info(f"Extended query: {extended_query}")
    return extended_query


def get_alternative_queries(
        user_query: str,
        intent: dict[str, str],
        min_number_of_alternative_queries: int = 1,
        max_number_of_alternative_queries: int = 3) -> list[str]:
    """Get alternative search queries to the given user query."""

    template = env.get_template("alternative_queries.j2")
    prompt = template.render(intro=intro_prompt,
                             user_query=user_query,
                             categories_intent=", ".join(intent["categories"]),
                             regions_intent=", ".join(intent["regions"]),
                             time_periods_intent=", ".join(intent["time_periods"]),
                             min_num=min_number_of_alternative_queries,
                             max_num=max_number_of_alternative_queries)

    logger.info("Creating alternative queries for: %s", user_query)
    alternative_queries = ollama.generate(model='mistral-small3.2', prompt=prompt).response

    logger.info(f"Alternative queries:")
    alternative_queries = [q.strip() for q in alternative_queries.split('\n') if q.strip()]
    for i in range(len(alternative_queries)):
        logger.info(f"{i+1}. {alternative_queries[i]}")
    return alternative_queries


def detect_user_intent(user_query: str, categories: list[str], other_category: str) -> dict[str, str]:
    """Detect the user intent from the query using LLM."""
    logger.info("Detecting intent for query: %s", user_query)

    num_categories = 2
    template = env.get_template("detect_user_intent.j2")
    prompt = template.render(intro=intro_prompt,
                             user_query=user_query,
                             num_categories=num_categories,
                             categories=", ".join(categories),
                             other_category=other_category)

    response = ollama.generate(model='mistral-small3.2', prompt=prompt).response
    if "```" in response:
        response = response.split("```")[1]
        if response.startswith("json"):
            response = response[len("json"):].strip()
    intent = json.loads(response)
    logger.info(f"Detected intent: {intent}")

    return intent