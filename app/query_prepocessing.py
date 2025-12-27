import re
from typing import Dict

import ollama
from utils import setup_logger

logger = setup_logger(__name__)


def query_preprocessing(user_query: str) -> (str, list):
    """Preprocess the user query."""

    logger.info("User query: %s", user_query)
    intent = detect_user_intent(user_query)
    alternative_queries = get_alternative_queries(user_query, intent)

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
    * Place (e.g. a city, region, geographical area): {user_intent['place']}
    * Discipline (a field, area of expertise): {user_intent['discipline']}
    * Time (e.g. a year, specific time range): {user_intent['time']}
    
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


def detect_user_intent(user_query: str) -> Dict[str, str]:
    """Detect the user intent from the query using LLM."""
    logger.info("Detecting intent for query: %s", user_query)

    general_intent_explanation = """
    You are an AI assistant for a dataset catalog search engine. Your task is to identify the user's intent 
    behind their search query."""

    ending = """  Otherwise return "no". You have to be 100% sure that you are right, return "no" if you are not 
    completely sure. Do not include any additional commentary. The user query is: """

    intents = {
        "place": """
        Your task is to answer if the user is looking for data for a specific place? 
        e.g. a city, region, places with specific geographical attributes (like rivers, mountains, etc.). 
        If yes, return the place name or names separated by commas.""",

        "discipline": """    
        Your task is to answer if the user is looking for data from a specific field, discipline or area of expertise? 
        e.g. healthcare, transportation, education, environment, etc.
        If yes, return the discipline name or names separated by commas.""",

        "time": """
        Your task is to answer if the user is looking for data from a specific time period or date range?
        e.g. historical data from a specific decade or year, data from the last year, data from winter months, etc.
        If yes, return the time period or date range."""
    }
    results = {}
    for name, intent in intents.items():
        query = f"{general_intent_explanation}{intent}{user_query}{ending}"
        response = ollama.generate(model='mistral-small3.2', prompt=query).response
        logger.info(f"Detected {name} intent: {response}")
        results[name] = response

    return results