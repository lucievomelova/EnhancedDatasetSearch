from typing import Dict, List

import ollama
from utils import setup_logger
import ast

logger = setup_logger(__name__)


def _make_list_from_results(results: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
    # take each value from the results dict and make one list from it
    result_list = []
    urls = set()
    for res_list in results.values():
        for item in res_list:
            if item['url'] not in urls:
                print(item['url'])
                urls.add(item['url'])
                result_list.append(item)
    return result_list


def result_postprocessing(user_query: str, results: Dict[str, List[Dict[str, str]]], intent: Dict[str, str], k: int = 10) -> List[Dict[str, str]] | None:
    """Post-process search results."""
    if not results:
        logger.info("No results found.")
        return None

    results = _make_list_from_results(results)
    logger.info(f"Post-processing search results.")
    query = f"""You are a helpful AI assistant for a dataset catalog search engine.
    Your task is to rerank the following search results based on their relevance to the user's original query:
    Original user query: {user_query}.
    The original query was expanded int oa few alternative queries. Here are the search results from all queries combined:
    {results}
    Each result contains the title, url and description text of a dataset.
    
    Some queries may have returned the same datasets, you can detect these by comparing their "URL" - if it's the 
    same, both datasets are the same. Rerank the datasets and return the most relevant ones, but 
    each dataset at most once.
    
    You know that the user is looking for data with the following intent:
    * Place (e.g. a city, region, geographical area): {intent['place']}
    * Discipline (a field, area of expertise): {intent['discipline']}
    * Time (e.g. a year, specific time range): {intent['time']}
    
    Rerank the datasets based on their relevance to the original user query and detected intent. 
    
    Do not change the title, url or text of any dataset.
    
    Return at most {k} most relevant datasets as a python list of dictionaries with title, url, text and explanation keys. 
    Be absolutely sure
    to return each dataset AT MOST ONCE, even if it appeared in results for multiple queries. 
    If the URL is the same, consider it the same dataset. You cannot return a dataset with the same URL more than once.

    In the explanation key, provide a brief explanation (1-2 sentences) why this result is relevant to the user's query.
    
    Return just the list, no additional commentary or markdown formatting. The result will be converted to a 
    python list directly, so it must contain ONLY the list.
    """

    reranked_results = ollama.generate(model='mistral-small3.2', prompt=f'{query}').response

    # keep only results in []
    reranked_results = reranked_results[reranked_results.find('['):reranked_results.rfind(']')+1]
    results_as_list = ast.literal_eval(reranked_results)

    logger.info(f"Reranked results: {results_as_list}.")
    return results_as_list


class Reranker:
    """Post-process and rerank search results based on relevance to the original query."""
    def __init__(self):
        pass
