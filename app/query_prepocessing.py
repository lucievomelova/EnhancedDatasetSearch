import re
import ollama
from utils import setup_logger

logger = setup_logger(__name__)


def query_preprocessing(user_query: str) -> list:
    """Preprocess the user query."""

    logger.info("User query: %s", user_query)

    # preprocessing with LLM - expand query
    llm_query = """
    Expand the given user query using synonyms and related terms so that the search results are more comprehensive.
    Return just the expanded query in natural language without any additional commentary.
    Expand the following user query: 
    """
    expanded_query = ollama.generate(model='tinyllama:1.1b', prompt=f'{llm_query} + {user_query}').response

    STOPWORDS = {'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'on', 'for', 'with', 'from', ",", "."}
    words = re.findall(r'\b\w+\b', expanded_query.lower())
    words = [w for w in words if w not in STOPWORDS]
    words = list(set(words))  # remove duplicate words
    logger.info("Expanded query: %s", words)
    return words