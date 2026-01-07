import json
from pprint import pprint
from typing import List

import pandas as pd
from llama_index.core import Document
from pandas import Series
import ollama

from utils import setup_logger

logger = setup_logger(__name__)


def create_documents(extended_df: pd.DataFrame) -> list[Document]:
    """Create a llama index Document from a dataframe row.

    The Document text will be: title + description + keywords + themes + categories + provider. All
    columns will also be stored in the metadata of the Document (even those that will be part of the text)."""
    documents = []
    descriptions = extended_df["description"]
    extended_df = extended_df.where(extended_df.notna(), None)
    metadata_df = extended_df.drop(columns="description").to_dict(orient="records")

    for description, metadata in zip(descriptions, metadata_df):
        text = f"""
            {metadata['title']}
            {description}

            Poskytovatel: {metadata['provider']}
            Klíčová slova: {metadata['keywords']}
            Témata: {metadata['themes']}
            Kategorie: {metadata['categories']}
            Region" : {metadata['region']}
            Časová období: {metadata['time_period']}
        """
        document = Document(text=text, metadata=metadata)
        documents.append(document)
    return documents


def enrich_metadata(row: Series, all_keywords: list, all_themes: list, categories: list, other_category: str) -> dict:
    """Enrich the metadata of the datasets in the extended dataframe using LLM."""
    keywords = row['klíčová_slova']
    title = row['název']
    themes = row['téma']
    text = row['popis']
    provider = row['poskytovatel']
    remaining_keywords = 3 - len(keywords) if keywords is not None else 3
    num_categories = 2
    num_themes = 2

    prompt = f"""You are a helpful AI assistant at a dataset portal. Your task is to generate metadata 
    for the given dataset based on its description. You should assign relevant keywords and a theme, classify the 
    dataset into categories, and detect any associated regions and time periods. Generated words must be in Czech.
    
    
    ## Keywords
    Your task is to assign up to {remaining_keywords} relevant keywords that represent the most 
    important concepts in the dataset. Use the most relevant keywords 
    from this list, which is the list of keywords already used for other datasets: {', '.join(all_keywords)}. If none
    of the provided keywords are appropriate, you can generate 1 new keyword based on the description.
    
    
    ## Theme
    Your task is to assign at most {num_themes} relevant themes that best categorizes the dataset. 
    Pick the most relevant themes from this list, which is the list of all themes already used for other 
    datasets: {', '.join(all_themes)}. Only include themes that are really relevant. If none of the provided themes
    is appropriate, you can generate 1 new theme based on the description.
    
    The list of possible themes contains also categories (from the following section). You can pick only one theme that
    is also a category, but only if it is clearly the most relevant theme for the dataset. The remaining themes have 
    to be different from the categories. 
    
    After you pick the themes, remove those that are already in the "Themes" part of the dataset description. Do NOT 
    add anything else. If you can't come up with any relevant theme or if all themes are removed, return an empty list.
    
    
    ## Category
    Classify the dataset into at most {num_categories} of the following predefined 
    categories: {', '.join(categories)}. You can choose up to {num_categories} categories if all are equally relevant. 
    But if one category is clearly more relevant than all other, choose only that one. 
    If none of the provided categories is appropriate, classify it into one category called {other_category}.
    
    
    ## Geographical Region
    It is possible that the dataset is associated with a geographical region. 
    Your task is to detect this region if applicable. The types of regions can be: a city, a village, 
    a part of a country, a geographical area (e.g. mountains, rivers, etc.) or a czech region ("kraj" in Czech).
    
    There can be more than one region associated with the dataset. For example, if the dataset is about a small village 
    in a specific region, both the village and the region should be returned. Or if the dataset is focused on one 
    part of Prague (e.g. Prague 10), return also Prague as a whole.
    
    Ignore if the dataset is associated only with Czech Republic as a whole - all datasets are associated with it, so 
    it isn't relevant. Do not include "Czech Republic" or variants of it in your response.
    
    
    ## Time Period
    It is possible that the dataset is associated with a specific time period. 
    Your task is to detect this time period if applicable. The types of time periods can be: a specific year, 
    a range of years, a decade, a specific month every year (e.g. February), or any other relevant time frame. 
    The time period is usually mentioned in the dataset title.
    
    There can be more than one time period associated with the dataset. For example, if the dataset contains data for
    each March in 2022-2025, return BOTH the range (2022-2025) and March.
    Do NOT assign time periods that correspond to some type of frequency - e.g. "annual", "monthly", etc.
    
    Here is the complete description of the dataset:
    Title: {title}
    Description: {text}
    Themes: {", ".join(themes) if themes is not None else "-"}
    Keywords: {", ".join(keywords) if keywords is not None else "-"}
    Provider: {provider}
    
    Return the metadata in the following JSON format:
    {{
        "keywords": [list of keywords (in Czech)],
        "themes": [list of themes (in Czech)],
        "categories": [list of categories (in Czech)],
        "regions": [list of regions (in Czech)],
        "time_periods": [list of time periods (in Czech)]
    }}
    
    Return only the final JSON object. Do NOT wrap the output in markdown. Do NOT use ```json or ``` fences.
    The whole output must be directly parseable by json.loads().
    """

    metadata_str = ollama.generate(model='mistral-small3.2',
                                   prompt=prompt,
                                   options = {
                                       "temperature": 0
                                   }
                                   ).response
    if "```" in metadata_str:
        metadata_str = metadata_str.split("```")[1]
        if metadata_str.startswith("json"):
            metadata_str = metadata_str[len("json"):].strip()
    metadata = json.loads(metadata_str)
    return metadata
