from typing import List

from llama_index.core import Document
from pandas import Series
import ollama

from utils import setup_logger

logger = setup_logger(__name__)


def create_document_from_row(row: Series) -> Document:
    """Create a llama index Document from a dataframe row.

    The Document text will be: title + description + keywords + themes + categories + provider. All
    columns will also be stored in the metadata of the Document (even those that will be part of the text)."""
    # columns - datová_sada, název, popis, poskytovatel, klíčová_slova, prostorové_pokrytí, téma,
    # periodicita_aktualizace, právní_předpis, kategorie_hvd_název

    keyword_list = _enrich_keywords(row) if row['klíčová_slova'] is None else row['klíčová_slova']
    provider = f"Poskytovatel: {row['poskytovatel']}" if row['poskytovatel'] is not None else '-'
    keywords = "Klíčová slova:" + ', '.join(keyword_list) if keyword_list is not None else '-'
    themes = "Témata:" + ', '.join(row['téma']) if row['téma'] is not None else '-'
    categories = "Kategorie:" + ', '.join(row['kategorie_hvd_název']) if row['kategorie_hvd_název'] is not None else '-'
    text = f"{row['název']}\n{row['popis']}\n\n{provider}\n{keywords}\n{themes}\n{categories}"

    metadata = {
        "title": row['název'],
        "url": row['datová_sada'],
        "keywords": row['klíčová_slova'],  # list of keywords
        "provider": row['poskytovatel'],
        "themes": row['téma'],  # list of themes
        "legal_regulations": row['právní_předpis'],  # list of legal regulations
        "categories": row['kategorie_hvd_název'],  # list of categories
    }
    return Document(text=text, metadata=metadata, doc_id=row['datová_sada'])


def _enrich_keywords(row: Series) -> List:
    """Enrich the dataset keywords with LLM if no keywords are already present."""
    system_query = """You are a helpful AI assistant for enriching dataset descriptions. You will be given a dataset 
    description with metadata. The description describes the dataset in natural language, it also contains the title 
    and other information about the dataset - relevant themes, categories, provider. Your task is to add up to 3 new 
    keywords in the empty "keywords" part by generating relevant keywords based on the description. 
    Be sure to:
    * generate only relevant keywords in the same language as the description
    * not include keywords that are already present in keywords, themes or categories.
    * not make up anything. All keywords must be taken directly from the description
    * make the keywords as simple as possible - do not use complex phrases
    * don't use too complicated or specific words - we want to use the generated keywords for search enhancement, so
    too complex keywords will not be useful
    * generate all keywords in the same form - e.g. verbs in infinitive, nouns in singular and first case
    
    Good examples of keywords are: city or town names, parts of a country, common terms related to the dataset topic,
    area of expertise, industry terms, etc.
    
    Don't include any additional commentary, just return up to 3 keywords as a comma-separated list. If you can't 
    come up with any relevant keywords, return an empty response.
    
    Here is the full description:
    """
    title = row['název']
    keywords = row['klíčová_slova']
    themes = row['téma']
    text = row['popis']
    prompt = f"""{system_query}{title}\n{text}\n
    Keywords: {", ".join(keywords) if keywords is not None else "-"}\n
    Themes: {", ".join(themes) if themes is not None else "-"}\n"""
    new_keywords = ollama.generate(model='mistral-small3.2', prompt=prompt).response
    logger.info(f"Title: {title},\n Old keywords: {keywords},\n New keywords: {new_keywords}")

    if new_keywords:
        new_keywords_list = [kw.strip() for kw in new_keywords.split(',') if kw.strip()]
        combined_keywords = keywords.extend(new_keywords_list) if keywords is not None else new_keywords_list
        return combined_keywords
    return []
