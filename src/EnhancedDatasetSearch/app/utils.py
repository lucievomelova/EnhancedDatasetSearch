from collections import defaultdict

import pandas as pd

from EnhancedDatasetSearch.data_processing.data_catalog import DataCatalog
from EnhancedDatasetSearch.data_processing.knowledge_graph import KnowledgeGraph


def get_common_metadata(metadata_category_list: list[str], metadata_a: dict, metadata_b: dict) -> dict[str, set[str]]:
    """Get common metadata between two datasets."""
    common_metadata = {}
    for cat in metadata_category_list:
        set_a = set(metadata_a[cat]) if metadata_a[cat] is not None else set()
        set_b = set(metadata_b[cat]) if metadata_b[cat] is not None else set()
        cat_title = cat.title().replace("_", " ")
        common_metadata[cat_title] = set_a.intersection(set_b)
    return common_metadata


def get_all_metadata(search_results: list[dict[str, str | list | None]]) -> dict[str, set[str]]:
    """Get all metadata appearing in search results.

    The retrieved metadata can be used for dynamic filtering on the search result page.
    Returns:
        a dict, where key is the metadata category and value is a set of all metadata values
        appearing in the search results for that category."""
    all_metadata = defaultdict(set)
    for result in search_results:
        metadata = result['categorization_metadata'] if result['categorization_metadata'] is not None else {}
        for metadata_category, metadata_values in metadata.items():
            if isinstance(metadata_values, str):
                all_metadata[metadata_category].add(metadata_values)
            elif isinstance(metadata_values, list):
                all_metadata[metadata_category].update(metadata_values)
    return all_metadata


def get_similar_datasets_with_preview_text(
        dataset_url: str,
        dataset_info: dict,
        data_catalog: DataCatalog,
        knowledge_graph: KnowledgeGraph
) -> dict:
    """Get datasets similar to the specified dataset with preview texts."""
    similar_datasets_raw = knowledge_graph.get_similar_datasets(dataset_url)

    similar_datasets = {}
    for sim_category, url_score_list in similar_datasets_raw.items():
        similar_datasets[sim_category] = []
        for url, _ in url_score_list:
            sim_dataset = data_catalog.get_dataset_by_url(url)
            if sim_dataset:
                text_preview = ""
                if pd.notna(sim_dataset['text']):
                    text_preview = sim_dataset['text']
                    if len(text_preview) > 200:  # too long description text preview -> take just first sentence
                        text_preview = sim_dataset['text'][:sim_dataset['text'].find(".") + 1]

                    # first sentence still too long or there is no "." char in the description
                    if len(text_preview) > 200 or len(text_preview) == 0:
                        index = sim_dataset['text'][:200].rfind(" ")
                        text_preview = sim_dataset['text'][:index]
                        if len(text_preview) > 0:
                            text_preview += "..."  # if the preview is not empty, show it was cut off by appending ...
                similar_dataset_info = {
                    'title': sim_dataset['title'],
                    'url': url,
                    'text_preview': text_preview,
                }
                if sim_category == "provider":
                    metadata_categories = ["keywords", "themes", "categories", "spatial_coverage", "temporal_coverage"]
                    common_metadata = get_common_metadata(metadata_categories,
                                                          dataset_info["categorization_metadata"],
                                                          sim_dataset["categorization_metadata"])
                    similar_dataset_info["common_metadata"] = common_metadata
                elif sim_category not in ["description", "provider"]:
                    common_metadata = get_common_metadata([sim_category],
                                                          dataset_info["categorization_metadata"],
                                                          sim_dataset["categorization_metadata"])
                    similar_dataset_info["common_metadata"] = common_metadata
                similar_datasets[sim_category].append(similar_dataset_info)

    similar_datasets["spatial coverage"] = similar_datasets.pop("spatial_coverage")
    similar_datasets["temporal coverage"] = similar_datasets.pop("temporal_coverage")
    return similar_datasets


def get_filters_for_results(results: list) -> dict:
    """Get metadata filters for search results."""
    filter_categories = ["keywords", "themes", "categories", "provider", "spatial_coverage", "temporal_coverage"]
    filters = {
        "keywords": {"title": "Keywords", "value_counts": dict()},
        "themes": {"title": "Themes", "value_counts": dict()},
        "categories": {"title": "Categories", "value_counts": dict()},
        "provider": {"title": "Provider", "value_counts": dict()},
        "spatial_coverage": {"title": "Spatial coverage", "value_counts": dict()},
        "temporal_coverage": {"title": "Temporal coverage", "value_counts": dict()},
    }
    if results:
        for dataset in results:
            categorization_metadata = dataset.get('categorization_metadata', {})
            for filter_category in filter_categories:
                result = categorization_metadata.get(filter_category, [])
                if isinstance(result, list):
                    for item in result:
                        filters[filter_category]["value_counts"][item] = filters[filter_category]["value_counts"].get(item, 0) + 1
                else:
                    filters[filter_category]["value_counts"][result] = filters[filter_category]["value_counts"].get(result, 0) + 1

    # sort by number of occurrences in search results
    for filter_category in filter_categories:
        filters[filter_category]["value_counts"] = dict(sorted(filters[filter_category]["value_counts"].items(), key=lambda x: x[1], reverse=True))

    # remove metadata category if no values are present
    filters = {category: info for category, info in filters.items() if len(info["value_counts"]) > 0}
    return filters
