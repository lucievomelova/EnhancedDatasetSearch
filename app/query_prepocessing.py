import re
import ollama
from utils import setup_logger

logger = setup_logger(__name__)


def query_preprocessing(user_query: str) -> (str, list):
    """Preprocess the user query."""

    logger.info("User query: %s", user_query)
    alternative_queries = get_alternative_queries(user_query)

    return user_query, alternative_queries


def get_alternative_queries(
        user_query: str,
        min_number_of_alternative_queries: int = 3,
        max_number_of_alternative_queries: int = 5) -> list[str]:
    """Get alternative search queries to the given user query."""

    system_query = f"""
    You are a helpful AI assistant for a dataset catalog search engine. 
    Your task is to expand a user search query so that the retrieval of relevant datasets 
    is more effective. Use synonyms, related terms, and broader concepts. Provide 
    {min_number_of_alternative_queries}-{max_number_of_alternative_queries} alternative 
    search queries that capture the essence of the user's intent. Return only the alternative queries 
    without any additional commentary. Return queries in the user's language. 

    Return the alternative queries ach on a separate line.

    The user query:
    """

    logger.info("Creating alternative queries for: %s", user_query)
    alternative_queries = ollama.generate(model='mistral-small3.2', prompt=f'{system_query}{user_query}').response

    logger.info(f"Alternative queries:")
    alternative_queries = [q.strip() for q in alternative_queries.split('\n') if q.strip()]
    for i in range(len(alternative_queries)):
        logger.info(f"{i+1}. {alternative_queries[i]}")
    return alternative_queries



# not used currently
def expand_with_synonyms_without_stopwords(user_query: str) -> list:
    """Expand the user query with synonyms using LLM, without stopwords."""
    llm_query = """
    Expand the given user query using synonyms and related terms so that the search results are more comprehensive.
    Return just the expanded query in natural language without any additional commentary.
    Expand the following user query: 
    """
    expanded_query = ollama.generate(model='mistral-small3.2', prompt=f'{llm_query} + {user_query}').response

    STOPWORDS = {'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'on', 'for', 'with', 'from', ",", "."}
    words = re.findall(r'\b\w+\b', expanded_query.lower())
    words = [w for w in words if w not in STOPWORDS]
    words = list(set(words))  # remove duplicate words
    logger.info("Expanded query: %s", words)
    return words