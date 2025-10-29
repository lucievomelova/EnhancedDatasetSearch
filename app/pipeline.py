"""
Search pipeline:
    1. Query Preprocessing
    2. Search
    3. Result Postprocessing
    4. Context

There will be a RAG database containing info about all datasets. Every day, the new datove_sady and distribuce csvs
will be downloaded and if there are changes detected in some datasets at NKOD, their info will be deleted from DB and
then added again.

"""
import re
from datetime import datetime
import os

import ollama
import pandas as pd
import requests
import logging
from query_prepocessing import query_preprocessing
from search import search
from result_postprocessing import result_postprocessing

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_todays_file(filename: str, url: str) -> pd.DataFrame:
    """Check if file exists and is up to date - if not, download it again, load it and return the loaded df."""
    response = requests.get(url)
    if os.path.exists(filename):
        mod_time = os.path.getmtime(filename)
        mod_datetime = datetime.fromtimestamp(mod_time)
        today = datetime.today().date()
    if not os.path.exists(filename) or mod_datetime.date() != today:
        with open(filename, "wb") as f:
            f.write(response.content)
    df = pd.read_csv(filename, sep=",")
    return df

def run(query: str) -> pd.DataFrame | None:
    """Run the search pipeline for the given query and return the results as a DataFrame."""
    # load current data files from NKOD
    data_url = "https://data.gov.cz/soubor/datov%C3%A9-sady.csv"
    data_filename = "data/datove_sady.csv"
    data_df = download_todays_file(data_filename, data_url)

    distribution_url = "https://data.gov.cz/soubor/distribuce.csv"
    distribution_filename = "data/distribuce.csv"
    distribution_df = download_todays_file(distribution_filename, distribution_url)
    logger.info("Loaded data files.")
    expanded_query = query_preprocessing(query)
    results = search(expanded_query, data_df)
    results = result_postprocessing(results)
    # return just nazev and popis columns
    if results is not None:
        # results["datová_sada"] = results["datová_sada"].apply(lambda url: f'<a href="{url}" target="_blank">link</a>')
        return results[['název', 'popis', "datová_sada"]]
    return None


# def query_preprocessing(user_query: str) -> list:
#     """Preprocess the user query."""
#
#     logger.info("User query: %s", user_query)
#
#     # preprocessing with LLM - expand query
#     llm_query = """
#     Expand the given user query using synonyms and related terms so that the search results are more comprehensive.
#     Return just the expanded query in natural language without any additional commentary.
#     Expand the following user query:
#     """
#     expanded_query = ollama.generate(model='tinyllama:1.1b', prompt=f'{llm_query} + {user_query}').response
#
#     STOPWORDS = {'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'on', 'for', 'with', 'from', ",", "."}
#     words = re.findall(r'\b\w+\b', expanded_query.lower())
#     words = [w for w in words if w not in STOPWORDS]
#     words = list(set(words))  # remove duplicate words
#     logger.info("Expanded query: %s", words)
#     return words
#
#
# def search(expanded_query: list, data_df: pd.DataFrame, limit: int = 100) -> pd.DataFrame | None:
#     """Search the data for the given query."""
#
#     logger.info("Searching.")
#     if expanded_query:
#         def _match_score(row):
#             text = f"{row['název']} {row['popis']}".lower()
#             return sum(1 for w in expanded_query if w in text)
#         data_df['score'] = data_df.apply(_match_score, axis=1)
#         results = data_df[data_df['score'] > 0].sort_values(by='score', ascending=False)[:limit]
#         logger.info("Searching complete, found %s relevant results.", results.shape[0])
#         return results
#     return None
#
#
# def result_postprocessing(results: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
#     """Postprocess the search results and return the most relevant datasets."""
#
#     results = results[:limit]
#     logger.info("Results: %s", results)
#     return results


if __name__ == "__main__":
    run()