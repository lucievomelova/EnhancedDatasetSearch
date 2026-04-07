import json
import os
import re

import pandas as pd
from llama_index.core import Document
from pandas import Series
import ollama
from jinja2 import Environment, FileSystemLoader
from polyleven import levenshtein

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

def extract_year_from_date(time_periods: list[str]) -> list[str]:
    """Extract year and possibly month from time_periods.

    The LLM sometimes assigns concrete dates to time periods, which is not desirable. Other times, it assigns
    just month+year, which we also do not want."""
    new_time_periods = set()
    date_pattern1 = r'^\d{2}[/\.\-]\d{2}[/\.\-](\d{4})$'
    date_pattern2 = r'^(\d{4})[/\.\-]\d{2}[/\.\-]\d{2}$'
    month_pattern = r'(0[1-9]|1[0-2])/(\d{4})'
    month_str_pattern = r'(leden|únor|březen|duben|květen|červen|červenec|srpen|září|říjen|listopadu|prosinec)\s+(\d{4})'

    month_mapping = {
        '01': 'leden', '02': 'únor', '03': 'březen', '04': 'duben',
        '05': 'květen', '06': 'červen', '07': 'červenec', '08': 'srpen',
        '09': 'září', '10': 'říjen', '11': 'listopad', '12': 'prosinec'
    }
    for time_period in time_periods:
        match = re.search(date_pattern1, time_period)
        if match:
            new_time_periods.add(match.group(1))
            continue

        match = re.search(date_pattern2, time_period)
        if match:
            new_time_periods.add(match.group(1))
            continue

        match = re.search(month_pattern, time_period)
        if match:
            new_time_periods.add(match.group(2)) # year
            month = match.group(1)
            new_time_periods.add(month_mapping[month])  # month
            continue

        match = re.search(month_str_pattern, time_period)
        if match:
            new_time_periods.add(match.group(2)) # year
            new_time_periods.add(match.group(1))  # month
            continue
        new_time_periods.add(time_period)
    return list(new_time_periods)

def year_range_to_year_list(time_periods: list[str]) -> list[str]:
    """Extract concrete years from a range of years.

    If the time period is e.g. 2022-2025, we want to have a list [2022, 2023, 2024, 2025]"""
    pattern =  r'(\d{4})\s*-\s*(\d{4})'
    new_time_periods = []
    for time_period in time_periods:
        match = re.search(pattern, time_period)
        if match:
            start_year = int(match.group(1))
            end_year = int(match.group(2))
            if end_year < start_year:
                start_year, end_year = end_year, start_year
            year_range = list(range(start_year, end_year + 1))
            new_time_periods.extend([str(year) for year in year_range])
        else:
            new_time_periods.append(time_period)
    return new_time_periods


def preprocess_temporal_coverage(datasets: pd.DataFrame) -> None:
    """Preprocess temporal coverage column."""
    all_periods = datasets["temporal_coverage"].explode().dropna().unique()
    logger.info(f"Preprocessing {len(all_periods)} time periods.")
    datasets["temporal_coverage"] = datasets["temporal_coverage"].apply(
        lambda temporal_coverage: extract_year_from_date(temporal_coverage)
    )
    all_periods = datasets["temporal_coverage"].explode().dropna().unique()
    logger.info(f"Extracted years from time periods. Number of unique time periods: {len(all_periods)}")

    datasets["temporal_coverage"] = datasets["temporal_coverage"].apply(
        lambda temporal_coverage: year_range_to_year_list(temporal_coverage)
    )
    all_periods = datasets["temporal_coverage"].explode().dropna().unique()
    logger.info(f"Expanded year ranges into individual years. Number of unique time periods: {len(all_periods)}")




def preprocess_keywords_and_themes(datasets: pd.DataFrame, model_name: str, state_dir: str) -> None:
    """Preprocess datasets keywords and themes to make data preprocessing and searching more effective.

    Because there are a lot of keywords and themes, they need extra preprocessing. Some contain typos,
    some are present only once, etc. We will preprocess them in order to decrease their amount, so that
    subsequent operations on them are faster and more effective."""

    columns = ["keywords", "themes"]
    for col in columns:
        logger.info(f"Preprocessing {col}.")
        # some keywords or themes might be incorrectly formatted and contain commas separating multiple keywords/themes
        datasets[col] = datasets[col].apply(lambda x: preprocess_comma_separated_words(x, model_name, state_dir) if "," in x else x)
        # strip whitespaces from beginning and end of each word
        datasets[col] = datasets[col].apply(lambda words: [w.strip() for w in words])

        # remove "." and "," from end of words (except for Sb., where its supposed to be)
        datasets[col] = datasets[col].apply(
            lambda words: [w[:-1] if (w.endswith(".") and not w.endswith("Sb.")) or w.endswith(",") else w for w in words]
        )

        # find words that differ just by capitalization
        all_words = datasets[col].explode().dropna().unique()
        logger.info(f"Number of words {len(all_words)}.")
        logger.info(f"Merging words that differ just by capitalization.")

        words_df = pd.DataFrame({
            'word': all_words,
            'word_lower': [k.lower() for k in all_words]
        })

        capitalization_df = words_df.groupby('word_lower')['word'].apply(list)
        capitalization_df = capitalization_df[capitalization_df.apply(len) > 1]  # keep rows which have multiple variants
        capitalization_df = capitalization_df.reset_index().rename(columns={'word': 'words'})
        if not capitalization_df.empty:

            # if there's an all caps word, then the word is probably an abbreviation -> we want to keep it uppercase
            capitalization_df['has_all_caps_word'] = capitalization_df['words'].apply(
                lambda words: any(w.isupper() and w.isalpha() for w in words)
            )
            capitalization_df["word_default"] = capitalization_df.apply(
                lambda row: row["word_lower"].upper() if row["has_all_caps_word"] else row["word_lower"], axis=1
            )
            # switch all words by their word_default in datasets df where applicable
            word_mapping = dict(zip(capitalization_df["word_lower"], capitalization_df["word_default"]))
            datasets[col] = datasets[col].apply(
                lambda words: [word_mapping.get(w.lower(), w) for w in words]
            )

        # use levensthein distance to find very similar words
        word_counts = datasets[col].explode().dropna().value_counts()
        all_words = datasets[col].explode().dropna().unique()
        logger.info(f"Merged. Number of words: {len(all_words)}")
        logger.info("Finding and merging similar words using levenshtein distance.")
        similarity_dict = {}  # key is the word to be replaced, value is the word it will be replaced by
        for i in range(len(all_words)):
            for j in range(i + 1, len(all_words)):
                w1, w2 = all_words[i].lower(), all_words[j].lower()
                if levenshtein(w1, w2) <= 2:
                    count1, count2 = int(word_counts[all_words[i]]), int(word_counts[all_words[j]])
                    len_of_shorter_word = min(len(w1), len(w2))
                    # if they differ in the last chars, we will consider them as same
                    if w1[:len_of_shorter_word - 1] == w1[:len_of_shorter_word - 1]:
                        if count1 < count2:
                            similarity_dict[w1] = w2
                        else:
                            similarity_dict[w2] = w1
                    # or if one of the words occurs only once, its probably a typo -> we will consider them the same
                    elif count1 == 1:
                        similarity_dict[w1] = w2
                    elif count2 == 1:
                        similarity_dict[w2] = w1

        # replace by similar words if applicable
        datasets[col] = datasets[col].apply(lambda words: [similarity_dict.get(w, w) for w in words])

        datasets[col] = datasets[col].apply(lambda words: [w for w in words if w is not None])
        all_words = datasets[col].explode().dropna().unique()
        logger.info(f"Merged. Number of words: {len(all_words)}")

def replace_nonfrequent_keywords_with_cluster_representatives(datasets: pd.DataFrame, model_name: str, state_dir: str) -> None:
    """Some keywords occur only once - replace tehm with other representative keywords."""

    word_counts = datasets["keywords"].explode().value_counts()
    single_occurence_words = word_counts[word_counts <= 1].index.tolist()
    logger.info(f"Replacing words that occur only once with their keyword cluster representatives ({len(single_occurence_words)}).")

    # assign representatives
    keyword_cluster_representatives = get_representatives(state_dir, model_name)
    # we will use inverted representatives mapping so that the lookup is faster
    inverted_keyword_cluster_representatives = {
        keyword: representative
        for representative, keywords in keyword_cluster_representatives.items()
        for keyword in keywords
    }
    datasets["keywords"] = datasets["keywords"].apply(
        lambda keywords: [inverted_keyword_cluster_representatives.get(k, k) if k in single_occurence_words else k for k in keywords]
    )

    word_counts = datasets["keywords"].explode().value_counts()
    logger.info(f"Replaced. Number of words: {len(word_counts)}")
    # now look at word count again and remove any remaining keywords that are single_occurrence
    single_occurence_words = word_counts[word_counts <= 1].index.tolist()
    datasets["keywords"] = datasets["keywords"].apply(
        lambda keywords: [k for k in keywords if k not in single_occurence_words]
    )

    all_words = datasets["keywords"].explode().dropna().unique()
    logger.info(f"Removed single occurrence keywords. Number of words: {len(all_words)}")


def clean_metadata(df: pd.DataFrame, categories: list[str], model_name: str, state_dir: str) -> None:
    """Clean the metadata of the datasets in the extended dataframe."""

    # we don't want reoccurring words in different metadata parts - we would just process more metadata unnecessarily
    # the order of significance is categories > themes > keywords
    all_themes = list(df["themes"].explode().dropna().unique())
    categories_from_themes = df["themes"].apply(lambda themes: [c for c in themes if c in categories])
    categories_from_keywords = df["keywords"].apply(lambda keywords: [c for c in keywords if c in categories])
    df["categories"] = df["categories"].apply(lambda x: list(set(x)))  # remove duplicates
    df["categories"] = df.apply(lambda row: list(set(row["categories"] + categories_from_themes.loc[row.name] +
                                categories_from_keywords.loc[row.name])), axis=1)

    themes_from_keywords = df["keywords"].apply(lambda keywords: [t for t in keywords if t in all_themes])

    df["themes"] = df.apply(lambda row: list(set(row["themes"] + themes_from_keywords.loc[row.name])), axis=1)
    df["themes"] = df["themes"].apply(lambda themes: [t for t in themes if t not in categories])
    df["keywords"] = df["keywords"].apply(lambda keywords: list(set([k for k in keywords if k not in categories + all_themes])))

    preprocess_keywords_and_themes(df, model_name, state_dir)


def create_documents(datasets: pd.DataFrame) -> list[Document]:
    """Create llama index Documents from the dataframe. Each row will be used to create one Document.

    The Document text will be: title + description + keywords + themes + categories + provider. All
    columns will also be stored in the metadata of the Document (even those that will be part of the text)."""
    documents = []
    descriptions = datasets["description"]
    datasets = datasets.where(datasets.notna(), None)
    metadata_df = datasets.drop(columns="description").to_dict(orient="records")
    logger.info(f"Creating {len(datasets)} llamaindex Documents.")

    for description, metadata in zip(descriptions, metadata_df):
        text = f"""
            {metadata['title']}
            {description}

            Poskytovatel: {metadata['provider']}
            Klíčová slova: {metadata['keywords']}
            Témata: {metadata['themes']}
            Kategorie: {metadata['categories']}
            Prostorové pokrytí": {metadata['spatial_coverage']}
            Časové pokrytí: {metadata['temporal_coverage']}
        """
        document = Document(text=text, metadata=metadata, id_=metadata["url"])
        documents.append(document)

    logger.info("Documents created.")
    return documents


def enrich_metadata(row: Series, all_keywords: list, all_themes: list, all_categories: list, other_category: str) -> dict:
    """Enrich the metadata of the datasets in the extended dataframe using LLM."""
    keywords = row['keywords']
    title = row['title']
    themes = row['themes']
    description = row['description']
    provider = row['provider']
    categories = row['categories']
    spatial_coverage = row['spatial_coverage']
    temporal_coverage = row['temporal_coverage']

    num_remaining = {
        "keywords": 3 - len(keywords) if keywords is not None else 3,
        "categories": 1 - len(categories) if categories is not None else 2,
        "themes": 2 - len(themes) if themes is not None else 2
    }
    generate_keywords = True if num_remaining["keywords"] > 0 else False
    generate_themes = True if num_remaining["themes"] > 0 else False
    generate_categories = True if num_remaining["categories"] > 0 else False

    generate_spatial_coverage = False if len(spatial_coverage) > 0 else True
    generate_temporal_coverage = False if len(temporal_coverage) > 0 else True

    template = env.get_template("enrich_metadata.j2")
    prompt = template.render(intro=intro_prompt,
                             remaining_keywords=num_remaining["keywords"],
                             generate_keywords=generate_keywords,
                             all_keywords=", ".join(all_keywords),
                             all_themes=", ".join(all_themes),
                             all_categories=", ".join(all_categories),
                             remaining_categories=num_remaining["categories"],
                             generate_categories=generate_categories,
                             other_category=other_category,
                             remaining_themes=num_remaining["themes"],
                             generate_themes=generate_themes,
                             generate_spatial_coverage=generate_spatial_coverage,
                             generate_temporal_coverage=generate_temporal_coverage,
                             title=title,
                             description=description,
                             themes=", ".join(themes) if themes is not None else "-",
                             keywords=", ".join(keywords) if keywords is not None else "-",
                             provider=provider,
                             return_json_instructions=return_json_instructions)
    retry = 0
    metadata_keys = ["keywords", "themes", "categories", "spatial_coverage", "temporal_coverage"]
    if (generate_keywords is False and generate_themes is False and generate_categories is False and
            generate_spatial_coverage is False and generate_temporal_coverage is False):
        metadata = {}
        for k in metadata_keys:
            metadata[k] = []
        return metadata

    while True:
        metadata_str = ollama.generate(model='mistral-small3.2',
                                   prompt=prompt,
                                   format="json",
                                   options = {
                                       "temperature": 0
                                   }
                                   ).response
        metadata = {}
        try:
            metadata = json.loads(metadata_str)
            if all(k in metadata_keys for k in metadata):  # check that all returned keys are actually metadata keys
                # check that the model did not generate more metadata than we specified
                for k in num_remaining.keys():
                    if len(metadata[k]) > num_remaining[k]:
                        retry += 1
                        continue
                for k in metadata_keys:  # fill in missing values with empty lists
                    if k not in metadata:
                        metadata[k] = []
                break
            retry += 1  # the result is missing a key, retry
        except json.decoder.JSONDecodeError as e:
            retry += 1
            logger.error(f"Error: {e}. Retrying...")
            if retry >= 3:
                # if the llm keeps making mistakes, fill problematic metadata categories with emty list and
                # return it to avoid blocking the pipeline
                for k in metadata_keys:
                    if k not in metadata:
                        metadata[k] = []
                break

    return metadata
