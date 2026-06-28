"""File containing functions for NKOD datasets' metadata processing."""

import json
import os
import re
from datetime import datetime

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from pandas import Series
from polyleven import levenshtein

from ollama_client import OllamaClient
from utils import setup_logger

logger = setup_logger(__name__)
env = Environment(loader=FileSystemLoader('prompts'))
intro_template = env.get_template("intro.j2")
intro_prompt = intro_template.render()
return_json_template = env.get_template("return_json.j2")
return_json_instructions = return_json_template.render()


def preprocess_comma_separated_words(client: OllamaClient, word_sequence: str, state_dir: str) -> list[str]:
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
    response = client.get_llm_json_response(prompt, num_retry_attempts=1)
    if response is None:
        return [word_sequence]  # return unprocessed sequence if the LLM couldn't process it

    current_new_keywords = response[word_sequence]  # split sequence into keywords
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


def extract_year_from_temporal_coverage(temporal_coverages: list[str]) -> list[str]:
    """Extract year from temporal_coverage.

    The LLM sometimes assigns concrete dates to time periods, which is not desirable. Other times, it assigns
    just month+year, which we also do not want."""
    new_temporal_coverages = set()
    date_pattern1 = r'^\d{1,2}[/\.\-]\s*\d{1,2}[/\.\-]\s*(\d{4})$'
    date_pattern2 = r'^(\d{4})[/\.\-]\d{1,2}[/\.\-]\d{1,2}$'
    month_pattern = r'(0[1-9]|1[0-2])/(\d{4})'
    month_str_pattern = r'(leden|únor|březen|duben|květen|červen|červenec|srpen|září|říjen|listopad|prosinec)\s+(\d{4})'

    month_mapping = {
        '01': 'leden', '02': 'únor', '03': 'březen', '04': 'duben',
        '05': 'květen', '06': 'červen', '07': 'červenec', '08': 'srpen',
        '09': 'září', '10': 'říjen', '11': 'listopad', '12': 'prosinec'
    }
    for temporal_coverage in temporal_coverages:
        match = re.search(date_pattern1, temporal_coverage)
        if match:
            new_temporal_coverages.add(match.group(1))
            continue

        match = re.search(date_pattern2, temporal_coverage)
        if match:
            new_temporal_coverages.add(match.group(1))
            continue

        match = re.search(month_pattern, temporal_coverage)
        if match:
            new_temporal_coverages.add(match.group(2)) # year
            month = match.group(1)
            new_temporal_coverages.add(month_mapping[month])  # month
            continue

        match = re.search(month_str_pattern, temporal_coverage)
        if match:
            new_temporal_coverages.add(match.group(2)) # year
            new_temporal_coverages.add(match.group(1))  # month
            continue
        new_temporal_coverages.add(temporal_coverage)
    return list(new_temporal_coverages)


def simplify_temporal_coverage(temporal_coverages: list[str]) -> list[str]:
    """Simplify temporal coverage to contain just year or range of years if possible.

    Sometimes the LLM assigns something like "year 2024" as temporal coverage or assigns a range of years like
    2022-now. Here we try to remove or replace some parts of these temporal coverages and also unify format of
    year ranges."""
    new_temporal_coverages = set()
    for temporal_coverage in temporal_coverages:
        if "rok " in temporal_coverage:
            temporal_coverage = temporal_coverage.replace("rok ", "")
        if "současnost" in temporal_coverage:
            temporal_coverage = temporal_coverage.replace("současnost", str(datetime.today().year))
        if "dnes" in temporal_coverage:
            temporal_coverage = temporal_coverage.replace("dnes", str(datetime.today().year))
        if "-" in  temporal_coverage:
            temporal_coverage = temporal_coverage.replace("-", "–")
        if "–" in temporal_coverage and " – " not in temporal_coverage:
            temporal_coverage = temporal_coverage.replace("–", " – ")

        new_temporal_coverages.add(temporal_coverage)
    return list(new_temporal_coverages)


def process_spatial_and_temporal_coverage(datasets: pd.DataFrame) -> None:
    """Process spatial and temporal coverage column programatically (without using an LLM).

    The goal is to remove unusable data and unify format of data where possible."""
    logger.info(f"Processing spatial and temporal coverage.")
    datasets["temporal_coverage"] = datasets["temporal_coverage"].apply(
        lambda temporal_coverage: extract_year_from_temporal_coverage(temporal_coverage)
    )
    logger.info("Extracted years from temporal coverage.")
    datasets["temporal_coverage"] = datasets["temporal_coverage"].apply(
        lambda temporal_coverage: simplify_temporal_coverage(temporal_coverage)
    )
    logger.info("Temporal coverage simplified to contain just year or year range where applicable.")

    # we don't want spatial coverage in forms of "Czech Republic" or similar - that is useless information
    cz_spatial_coverage = ["ČESKÁ REPUBLIKA", "CZ", "CZECH REPUBLIC", "ČESKO", "CZECH"]
    datasets["spatial_coverage"] = datasets["spatial_coverage"].apply(
        lambda spatial_coverage: [s for s in spatial_coverage if s.upper() not in cz_spatial_coverage]
    )
    logger.info("Removed variants of \"Czech Republic\" from spatial coverage.")
    # many spatial coverages contain municipality or district word (obec, okres) - remvoe
    datasets["spatial_coverage"] = datasets["spatial_coverage"].apply(
        lambda spatial_coverage: [s.replace("obec ", "").replace("okres ", "") for s in spatial_coverage]
    )
    logger.info("Removed \"obec\" and \"okres\" from spatial coverage.")


def _unify_keyword_capitalization(datasets: pd.DataFrame, column: str) -> None:
    """Find words that differ just by capitalization and rewrite them into the same form."""
    all_words = datasets[column].explode().dropna().unique()
    logger.info(f"Number of words {len(all_words)}.")
    logger.info(f"Merging keywords that differ just by capitalization.")

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
        datasets[column] = datasets[column].apply(
            lambda words: [word_mapping.get(w.lower(), w) for w in words]
        )


def preprocess_keywords(datasets: pd.DataFrame, client: OllamaClient | None, state_dir: str) -> None:
    """Preprocess datasets keywords to make data preprocessing and searching more effective.

    Because there are a lot of keywords, they need extra preprocessing. Some contain typos,
    some are present only once, etc. We will preprocess them in order to decrease their amount, so that
    subsequent operations on them are faster and more effective."""
    if datasets.empty:
        return
    col = "keywords"
    datasets[col] = datasets[col].apply(lambda k: list(set(k)))  # remove possible duplicates from keywords
    logger.info(f"Preprocessing keywords.")
    if OllamaClient:
        # some keywords might be incorrectly formatted and contain commas separating multiple keywords/themes
        datasets[col] = datasets[col].apply(lambda x: preprocess_comma_separated_words(client, x, state_dir) if "," in x else x)
    # strip whitespaces from beginning and end of each word
    datasets[col] = datasets[col].apply(lambda words: [w.strip() for w in words])

    # remove "." and "," from end of words (except for Sb., where its supposed to be)
    datasets[col] = datasets[col].apply(
        lambda words: [w[:-1] if (w.endswith(".") and not w.endswith("Sb.")) or w.endswith(",") else w for w in words]
    )
    _unify_keyword_capitalization(datasets, col)

    # use Levenshtein distance to find very similar words
    word_counts = datasets[col].explode().dropna().value_counts()
    all_words = datasets[col].explode().dropna().unique()
    logger.info(f"Merged. Number of words: {len(all_words)}")
    logger.info("Finding and merging similar words using levenshtein distance.")
    similarity_dict = {}  # key is the word to be replaced, value is the word it will be replaced by
    for i in range(len(all_words)):
        for j in range(i + 1, len(all_words)):
            w1, w2 = all_words[i].lower(), all_words[j].lower()
            if levenshtein(w1, w2) <= 1:
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

    datasets[col] = datasets[col].apply(lambda words: [w for w in set(words) if w is not None])
    all_words = datasets[col].explode().dropna().unique()
    logger.info(f"Merged. Number of keywords: {len(all_words)}")


def clean_metadata(df: pd.DataFrame, client: OllamaClient | None, categories: list[str], state_dir: str) -> None:
    """Clean the datasets' metadata."""

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

    preprocess_keywords(df, client, state_dir)


def enrich_metadata(row: Series, client: OllamaClient, all_keywords: set, all_categories: set, other_category: str) -> dict:
    """Enrich the metadata of one dataset represented by the *row* using an LLM."""
    keywords = row['keywords']
    title = row['title']
    description = row['description']
    provider = row['provider']
    categories = row['categories']
    spatial_coverage = row['spatial_coverage']
    temporal_coverage = row['temporal_coverage']

    # number of remaining keywords and categories that should be generated by an LLM
    num_remaining = {
        "keywords": 3 - len(keywords) if keywords is not None else 3,
        "categories": 0 if categories is not None else 2,
    }
    # decide which metadata we want to include in the LLM prompt
    generate_keywords = True if num_remaining["keywords"] > 0 else False
    generate_categories = True if num_remaining["categories"] > 0 else False
    generate_spatial_coverage = False if len(spatial_coverage) > 0 else True
    generate_temporal_coverage = False if len(temporal_coverage) > 0 else True

    template = env.get_template("enrich_metadata.j2")
    prompt = template.render(intro=intro_prompt,
                             remaining_keywords=num_remaining["keywords"],
                             generate_keywords=generate_keywords,
                             all_keywords=", ".join(all_keywords),
                             all_categories=", ".join(all_categories),
                             remaining_categories=num_remaining["categories"],
                             generate_categories=generate_categories,
                             other_category=other_category,
                             generate_spatial_coverage=generate_spatial_coverage,
                             generate_temporal_coverage=generate_temporal_coverage,
                             title=title,
                             description=description,
                             keywords=", ".join(keywords) if keywords is not None else "-",
                             provider=provider,
                             return_json_instructions=return_json_instructions)

    metadata_keys = ["keywords", "categories", "spatial_coverage", "temporal_coverage"]
    if (generate_keywords is False and generate_categories is False and
            generate_spatial_coverage is False and generate_temporal_coverage is False):
        metadata = {}
        for k in metadata_keys:
            metadata[k] = []
        return metadata

    num_retry_attempts = 3
    retry = 0
    metadata = {}  # initialize metadata to empty dict
    while retry < 3:
        remaining_attempts = num_retry_attempts - retry
        metadata, retries = client.get_llm_json_response(prompt, num_retry_attempts=remaining_attempts)
        if all(k in metadata_keys for k in metadata):  # check that all returned keys are actually metadata keys
            for k in metadata_keys:  # fill in missing values with empty lists
                if k not in metadata:
                    metadata[k] = []
            # check that the model did not generate more metadata than we specified
            for k, num in num_remaining.items():
                if num > 0 and len(metadata[k]) > num_remaining[k]:
                    continue
            break
        retry += retries + 1 # the result is missing a key, retry

    # fill missing metadata categories (if there are any) with emty list and return it to avoid blocking the pipeline
    for k in metadata_keys:
        if k not in metadata:
            metadata[k] = []
    return metadata
