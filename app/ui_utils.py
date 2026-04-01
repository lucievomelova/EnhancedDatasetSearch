from collections import defaultdict


def get_common_metadata(metadata_category_list: list[str], metadata_a: dict, metadata_b: dict) -> dict[str, set[str]]:
    """Get common metadata between two datasets."""
    common_metadata = {}
    for cat in metadata_category_list:
        set_a = set(metadata_a[cat]) if metadata_a[cat] is not None else set()
        set_b = set(metadata_b[cat]) if metadata_b[cat] is not None else set()
        common_metadata[cat] = set_a.intersection(set_b)
    return common_metadata


def get_all_metadata(search_results: list[dict[str, str | list | None]]) -> dict[str, set[str]]:
    """Get all metadata appearing in search results.

    The retrieved metadata can be used for dynamic filtering on the search result page.
    Returns:
        a dict, where key is the metadata category and value is a set of all metadata values
        appearing in the search results for that category."""
    all_metadata = defaultdict(set)
    for result in search_results:
        metadata = result['metadata'] if result['metadata'] is not None else {}
        for metadata_category, metadata_values in metadata.items():
            if isinstance(metadata_values, str):
                all_metadata[metadata_category].add(metadata_values)
            elif isinstance(metadata_values, list):
                all_metadata[metadata_category].update(metadata_values)
    return all_metadata

