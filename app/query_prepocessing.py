import json
import re
from typing import Dict

import ollama
from utils import setup_logger

logger = setup_logger(__name__)


def query_preprocessing(user_query: str, categories: list[str], other_category: str) -> (str, list):
    """Preprocess the user query."""

    logger.info("User query: %s", user_query)
    intent = detect_user_intent(user_query, categories, other_category)
    # alternative_queries = get_alternative_queries(user_query, intent)
    alternative_queries = []

    return intent, alternative_queries


def get_alternative_queries(
        user_query: str,
        user_intent: Dict[str, str],
        min_number_of_alternative_queries: int = 3,
        max_number_of_alternative_queries: int = 5) -> list[str]:
    """Get alternative search queries to the given user query."""

    system_query = f"""
    You are a helpful AI assistant for a dataset catalog search engine. 
    Your task is to expand a user search query so that the retrieval of relevant datasets 
    is more effective. Use synonyms, related terms, and broader concepts.
    
    You know that the user is looking for data with the following intent:
    * Categories: {user_intent['categories']}
    * Geographical regions: {user_intent['regions']}
    * Time Periods: {user_intent['time_periods']}
    
    Provide {min_number_of_alternative_queries}-{max_number_of_alternative_queries} alternative 
    search queries that capture the essence of the user's intent. 
    
    Return only the alternative queries without any additional commentary. 
    At least two of the alternative queries must be a full sentence.
    Return queries in the user's language, each on a separate line.

    The user query:
    """

    logger.info("Creating alternative queries for: %s", user_query)
    alternative_queries = ollama.generate(model='mistral-small3.2', prompt=f'{system_query}{user_query}').response

    logger.info(f"Alternative queries:")
    alternative_queries = [q.strip() for q in alternative_queries.split('\n') if q.strip()]
    for i in range(len(alternative_queries)):
        logger.info(f"{i+1}. {alternative_queries[i]}")
    return alternative_queries


def detect_user_intent(user_query: str, categories: list[str], other_category: str) -> Dict[str, str]:
    """Detect the user intent from the query using LLM."""
    logger.info("Detecting intent for query: %s", user_query)

    num_categories = 2

    prompt = f"""
    You are an AI assistant for a dataset catalog search engine. Your task is to identify the user's intent 
    behind their search query.
    
    The user query is: {user_query}

    ## Category
    Classify the query intent into at most {num_categories} of the following predefined 
    categories: {', '.join(categories)}. You can choose up to {num_categories} categories if all are equally relevant. 
    But if one category is clearly more relevant than all other, choose only that one. 
    If none of the provided categories is appropriate, classify it into one category called {other_category}.
        
    ## Region
    Your task is to answer if the user is looking for data for a specific geographical region? 
    e.g. a city, region, places with specific geographical attributes (like rivers, mountains, etc.). 
    If yes, return the place name or names separated by commas.

    ## Time Period
    Your task is to answer if the user is looking for data from a specific time period or date range?
    e.g. historical data from a specific decade or year, data from the last year, data from winter months, etc.
    If yes, return the time period or date range.
    
    Return the intent in the following JSON format:
    {{
        "categories": [list of categories (in Czech)],
        "regions": [list of regions (in Czech)],
        "time_periods": [list of time periods (in Czech)]
    }}
    
    Return only the final JSON object. Do NOT wrap the output in markdown. Do NOT use ```json or ``` fences.
    The whole output must be directly parseable by json.loads().    
    """

    response = ollama.generate(model='mistral-small3.2', prompt=prompt).response
    if "```" in response:
        response = response.split("```")[1]
        if response.startswith("json"):
            response = response[len("json"):].strip()
    intent = json.loads(response)
    logger.info(f"Detected intent: {intent}")

    return intent