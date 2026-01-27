import json
from pprint import pprint
from typing import List

import pandas as pd
from llama_index.core import Document
from pandas import Series
import ollama
from jinja2 import Environment, FileSystemLoader

from utils import setup_logger

logger = setup_logger(__name__)
env = Environment(loader=FileSystemLoader('prompts'))
intro_template = env.get_template("intro.j2")
intro_prompt = intro_template.render()
return_json_template = env.get_template("return_json.j2")
return_json_instructions = return_json_template.render()


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
    return metadata
