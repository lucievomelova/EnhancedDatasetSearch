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
    metadata_df = extended_df.drop(columns="description").to_dict(orient="records")

    for description, metadata in zip(descriptions, metadata_df):
        text = f"{metadata['title']}\n{description}\n\n{metadata['provider']}\n{metadata['keywords']}\n{metadata['themes']}\n{metadata['categories']}"
        document = Document(text=text, metadata=metadata)
        documents.append(document)
    return documents


def assign_keywords(row: Series, all_keywords: list) -> List:
    """Assign or generate dataset keywords (using LLM) if no keywords are already present."""
    keywords = row['klíčová_slova']
    title = row['název']
    themes = row['téma']
    text = row['popis']
    provider = row['poskytovatel']
    remaining_keywords = 3 - len(keywords) if keywords is not None else 3
    if remaining_keywords <= 0:
        return keywords
    prompt = f"""You are a helpful AI assistant at a dataset portal. You will be given a dataset 
    description with metadata. The description describes the dataset in natural language, it also contains the title 
    and other information about the dataset - relevant themes, categories, provider. Your task is to assign up to 
    {remaining_keywords} keywords in the "keywords" part. Preferably, you should use the most relevant keywords from this list, 
    which is the list of all keywords already used for other datasets: {', '.join(all_keywords)}.     
     
    If none of the provided keywords are appropriate or if you are completely sure there is another important keyword 
    that would be used to describe the dataset, you can generate new keywords based on the description. 
    But be sure to:
    * generate only relevant keywords in the same language as the description
    * not include keywords that are already present in keywords, themes or categories.
    * not make up anything. All keywords must be based on the description
    * make the keywords as simple as possible - do not use complex phrases
    * don't use too complicated or specific words
    * generate all keywords in the default word form - e.g. verbs in infinitive, nouns in singular and first case
    * take inspiration from the provided list of keywords.
    
    Good examples of keywords are: city or town names, parts of a country, common terms related to the dataset topic,
    area of expertise, industry terms, etc.
    
    Only provide keywords that are really relevant, you have to be completely sure they are relevant.
    Do not return keywords that are already part of the existing keywords, themes or providers.
    
    Don't include any additional commentary, just return the keywords as a comma-separated list. If you can't 
    come up with any relevant keywords, return an empty response.
    
    Here is the full description:
    {title}
    {text}
    Themes: {", ".join(themes) if themes is not None else "-"}
    Keywords: {", ".join(keywords) if keywords is not None else "-"}
    Provider: {provider}
    """
    new_keywords = ollama.generate(model='mistral-small3.2', prompt=prompt).response
    logger.info(f"Title: {title},\n Old keywords: {keywords},\n New keywords: {new_keywords}")

    if new_keywords:
        new_keywords_list = [kw.strip() for kw in new_keywords.split(',') if kw.strip()]
        combined_keywords = keywords + new_keywords_list if keywords is not None else new_keywords_list
        logger.info(f"Title: {title},\n Old keywords: {keywords},\n New keywords: {combined_keywords}")
        return combined_keywords
    return []


def assign_theme(row: Series, all_themes: list) -> List:
    """Assign or generate the dataset theme (with LLM) if no theme is already present."""
    title = row['název']
    keywords = row['klíčová_slova']
    themes = row['téma']
    text = row['popis']
    prompt = f"""You are a helpful AI assistant at a dataset portal. You will be given a dataset 
    description with metadata. The description describes the dataset in natural language, it also contains the title 
    and other information about the dataset - relevant keywords, categories, provider. Your task is to assign one theme 
    in the empty "theme" part. Preferably, you should use the most relevant theme from this list, 
    which is the list of all themes already used for other datasets: {', '.join(all_themes)}.     

    If none of the provided themes is appropriate, you can generate a new theme based on the description. 
    But be sure to:
    * generate only a relevant theme in the same language as the description
    * not make up anything. The generated theme must be based on the dataset description
    * do not overly complex phrases
    * generate the theme in the default word form - e.g. verbs in infinitive, nouns in singular and first case
    * take inspiration from the provided list of themes.

    Good examples of themes are: discipline or area of expertise, name of the industry that is described.

    Don't include any additional commentary, just return one theme. If you can't 
    come up with a relevant theme, return an empty response.

    Here is the full description:    
    {title}
    {text}
    Keywords: {", ".join(keywords) if keywords is not None else "-"}
    """
    new_theme = ollama.generate(model='mistral-small3.2', prompt=prompt).response
    logger.info(f"Title: {title},\n Old themes: {themes},\n New theme: {new_theme}")

    if new_theme:
        new_theme_list = [theme.strip() for theme in new_theme.split(',') if theme.strip()]
        return new_theme_list
    return []


def classify_into_domain(row: Series) -> list:
    """Classify the dataset into one of the predefined domains."""
    domains = ["Science", "Technology", "Engineering", "Mathematics", "Healthcare", "Law", "Economics",
               "Infrastructure", "Business", "Education", "Social Sciences", "Culture", "History", "Environment",
               "Geography", "Agriculture", "Demography", "Auxiliary Dataset"]
    title = row['název']
    keywords = row['klíčová_slova']
    themes = row['téma']
    text = row['popis']
    prompt = f"""You are a helpful AI assistant at a dataset portal. You will be given a dataset 
    description with metadata. The description describes the dataset in natural language, it also contains the title 
    and other information about the dataset - relevant keywords, categories, provider. Your task is to 
    classify the dataset into the following domains: {', '.join(domains)}.

    If none of the provided domains is appropriate, classify it as Other. You can choose up to two domains if 
    both are equally relevant. But if one domain is clearly more relevant than all other, choose only that one.
    Don't include any additional commentary, just return the domain as a string or domains as a comma separated list. 

    Here is the full description:    
    {title}
    {text}
    Keywords: {", ".join(keywords) if keywords is not None else "-"}
    Themes: {", ".join(themes) if themes is not None else "-"}
    """

    detected_domains = ollama.generate(model='mistral-small3.2', prompt=prompt).response
    if detected_domains:
        detected_domains_list = [domain.strip() for domain in detected_domains.split(',') if domain.strip()]
        return detected_domains_list
    return []


def detect_region(row: Series) -> list:
    """Detect the region associated with the dataset."""
    title = row['název']
    keywords = row['klíčová_slova']
    themes = row['téma']
    text = row['popis']
    prompt = f"""You are a helpful AI assistant at a dataset portal. You will be given a dataset 
    description with metadata. The description describes the dataset in natural language, it also contains the title 
    and other information about the dataset - relevant keywords, categories, provider. It is possible that 
    the dataset is associated with a geographical region. Your task is to detect this region if applicable.
    
    The types of regions can be: a city, a village, a part of a country, a geographical area (e.g. mountains, rivers, etc.)
    or a czech region ("kraj" in Czech). It has to be a geographical region. 
    
    There can be more than one region associated with the dataset. For example, if the dataset is about a small village 
    in a specific region, both the village and the region should be returned. Or if the dataset is focused on one 
    part of Prague (e.g. Prague 10), return also Prague as a whole.
    
    Ignore if the dataset is associated only with Czech Republic as a whole - all datasets are associated with it, so 
    it isn't relevant. Do not include "Czech Republic" or variants of it in your response.

    If no region is associated with the dataset, return -.
    Don't include any additional commentary, just return the regions as a comma separated list.

    Here is the full description:    
    {title}
    {text}
    Keywords: {", ".join(keywords) if keywords is not None else "-"}
    Themes: {", ".join(themes) if themes is not None else "-"}
    """

    detected_regions = ollama.generate(model='mistral-small3.2', prompt=prompt).response
    if detected_regions:
        detected_regions_list = [region.strip() for region in detected_regions.split(',') if region.strip()]
        return detected_regions_list
    return []


def detect_time_period(row: Series) -> list:
    """Detect the time_period associated with the dataset."""
    title = row['název']
    keywords = row['klíčová_slova']
    themes = row['téma']
    text = row['popis']
    prompt = f"""You are a helpful AI assistant at a dataset portal. You will be given a dataset 
    description with metadata. The description describes the dataset in natural language, it also contains the title 
    and other information about the dataset - relevant keywords, categories, provider. It is possible that 
    the dataset is associated with a specific time period. Your task is to detect this time period if applicable.

    The types of time periods can be: a specific year, a range of years, a decade, a century, 
    or any other relevant time frame.

    There can be more than one time period associated with the dataset. For example, if the dataset contains data for
    years 2022-2025, return both the range and each individual year.
    
    It is possible that no time period is associated with the dataset. In that case return -.
    Don't include any additional commentary, just return the time periods as a comma separated list.

    Here is the full description:    
    {title}
    {text}
    Keywords: {", ".join(keywords) if keywords is not None else "-"}
    Themes: {", ".join(themes) if themes is not None else "-"}
    """

    detected_time_periods = ollama.generate(model='mistral-small3.2', prompt=prompt).response
    if detected_time_periods:
        detected__time_periods_list = [time.strip() for time in detected_time_periods.split(',') if time.strip()]
        return detected__time_periods_list
    return []