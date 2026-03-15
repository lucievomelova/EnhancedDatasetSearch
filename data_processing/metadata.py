import json
import os
from pprint import pprint

import pandas as pd
from llama_index.core import Document
from pandas import Series
import ollama
from jinja2 import Environment, FileSystemLoader

from data_processing.keywords import get_representatives
from utils import setup_logger

logger = setup_logger(__name__)
env = Environment(loader=FileSystemLoader('prompts'))
intro_template = env.get_template("intro.j2")
intro_prompt = intro_template.render()
return_json_template = env.get_template("return_json.j2")
return_json_instructions = return_json_template.render()


def preprocess_comma_separated_words(word_sequence: str, model_name: str, state_dir: str) -> list[str]:
    """Preprocess words containing commas.

    Some keywords or themes like this truly contain commas, but others are actually multiple keywords
    that were formatted incorrectly. Use an LLM to identify and split those."""
    state_file = state_dir + "/comma_separated_keywords.json"
    if os.path.exists(state_file):  # check if we already processed some keywords
        with open(state_file, "r") as f:
            already_processed_keywords = json.load(f)
            if word_sequence in already_processed_keywords:
                return already_processed_keywords[word_sequence]

    template = env.get_template("keywords_commas.j2")
    prompt = template.render(keyword=word_sequence, return_json_instructions=return_json_instructions)
    response = ollama.generate(model=model_name,
                               prompt=prompt, stream=False, format="json",
                               options={"temperature": 0}).response

    current_new_keywords = json.loads(response)[word_sequence]  # split sequence into keywords
    logger.info(current_new_keywords)
    if not os.path.exists(state_file):  # add to file with already processed keywords
        with open(state_file, "w") as f:
            json.dump({word_sequence: current_new_keywords}, f, indent=4, ensure_ascii=False)
    else:
        with open(state_file, "r") as f:
            content = json.load(f)
            content.update({word_sequence: current_new_keywords})
        with open(state_file, "w") as f:
            json.dump(content, f, indent=4, ensure_ascii=False)
    return current_new_keywords


def clean_metadata(df: pd.DataFrame, categories: list[str], model_name: str, state_dir: str) -> None:
    """Clean the metadata of the datasets in the extended dataframe."""
    # some keywords or themes might be incorrectly formatted and contain commas separating multiple keywords/themes
    df["keywords"] = df["keywords"].apply(lambda x: preprocess_comma_separated_words(x, model_name, state_dir) if "," in x else x)
    df["themes"] = df["themes"].apply(lambda x: preprocess_comma_separated_words(x, model_name, state_dir) if "," in x else x)

    all_themes = list(df["themes"].explode().dropna().unique())

    # we don't want reoccurring words in different metadata parts - we would just process more metadata unnecessarily
    # the order of significance is categories > themes > keywords
    categories_from_themes = df["themes"].apply(lambda themes: [c for c in themes if c in categories])
    categories_from_keywords = df["keywords"].apply(lambda keywords: [c for c in keywords if c in categories])
    df["categories"] = df["categories"].apply(lambda x: list(set(x)))  # remove duplicates
    df["categories"] = df.apply(lambda row: list(set(row["categories"] + categories_from_themes.loc[row.name] +
                                categories_from_keywords.loc[row.name])), axis=1)

    themes_from_keywords = df["keywords"].apply(lambda keywords: [t for t in keywords if t in all_themes])

    df["themes"] = df.apply(lambda row: list(set(row["themes"] + themes_from_keywords.loc[row.name])), axis=1)
    df["themes"] = df["themes"].apply(lambda themes: [t for t in themes if t not in categories])
    df["keywords"] = df["keywords"].apply(lambda keywords: [k for k in keywords if k not in categories + all_themes])


def add_cluster_representatives_to_metadata(df: pd.DataFrame, state_dir: str) -> None:
    """Add keyword cluster representatives to the metadata of the datasets in the extended dataframe. """
    # explode keywords so we can assign a keywords representative to each of them, then group them back together
    exploded = df['keywords'].explode().reset_index()
    exploded.columns = ['original_index', 'keyword']

    # assign representatives
    keyword_cluster_representatives = get_representatives(state_dir)
    # we will use inverted representatives mapping so that the lookup is faster
    inverted_keyword_cluster_representatives = {
        keyword: representative
        for representative, keywords in keyword_cluster_representatives.items()
        for keyword in keywords
    }
    exploded['representative'] = exploded['keyword'].map(inverted_keyword_cluster_representatives)
    exploded = exploded.dropna(subset=['representative'])
    representatives_series = exploded.groupby('original_index')['representative'].apply(list)
    df['keyword_cluster_representatives'] = representatives_series.reindex(df.index, fill_value=[])


def create_documents(extended_df: pd.DataFrame) -> list[Document]:
    """Create llama index Documents from the dataframe. Each row will be used to create one Document.

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
            Hlavní klíčová slova: {metadata['keyword_cluster_representatives']}
            Témata: {metadata['themes']}
            Kategorie: {metadata['categories']}
            Region" : {metadata['region']}
            Časová období: {metadata['time_period']}
        """
        document = Document(text=text, metadata=metadata, id_=metadata["url"])
        documents.append(document)

    logger.info("Documents created.")
    return documents


def enrich_metadata(row: Series, all_keywords: list, all_themes: list, categories: list, other_category: str) -> dict:
    """Enrich the metadata of the datasets in the extended dataframe using LLM."""
    keywords = row['keywords']
    title = row['title']
    themes = row['themes']
    description = row['description']
    provider = row['provider']
    remaining_keywords = 3 - len(keywords) if keywords is not None else 3
    num_categories = 2
    num_themes = 2

    template = env.get_template("enrich_metadata.j2")
    prompt = template.render(intro=intro_prompt,
                             remaining_keywords=remaining_keywords,
                             all_keywords=", ".join(all_keywords),
                             all_themes=", ".join(all_themes),
                             categories=", ".join(categories),
                             num_categories=num_categories,
                             other_category=other_category,
                             num_themes=num_themes,
                             title=title,
                             description=description,
                             themes=", ".join(themes) if themes is not None else "-",
                             keywords=", ".join(keywords) if keywords is not None else "-",
                             provider=provider,
                             return_json_instructions=return_json_instructions)

    metadata_str = ollama.generate(model='mistral-small3.2',
                                   prompt=prompt,
                                   format="json",
                                   options = {
                                       "temperature": 0
                                   }
                                   ).response

    metadata = json.loads(metadata_str)
    logger.info(metadata)
    return metadata
