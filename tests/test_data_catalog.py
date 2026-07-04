"""Unit tests for data catalog."""


def test_dataframes_exist(data_catalog):
    """Test that all needed dataframes are not None."""
    assert data_catalog.datasets is not None
    assert data_catalog.distributions is not None


def test_get_dataset_by_url_existing_url(data_catalog):
    """Test that get_dataset_by_url() works for an existing URL."""
    url = "http://example.com/ds1"
    dataset_info = data_catalog.get_dataset_by_url(url)
    assert dataset_info is not None
    assert dataset_info["url"] == url


def test_get_dataset_by_url_nonexistent_url(data_catalog):
    """Test that get_dataset_by_url() returns None for non-existent URL."""
    url = "http://nonexistent"
    dataset_info = data_catalog.get_dataset_by_url(url)
    assert dataset_info is None


def test_metadata_filters_contain_correct_values(data_catalog):
    """Test that get_filters_with_counts() contains keywords, themes, categories and providers with positive counts."""
    filters = data_catalog.get_filters_with_counts()
    for metadata_category in ["keywords", "themes", "categories", "provider"]:
        assert metadata_category in filters
        for count in filters[metadata_category]["value_counts"].values():
            assert count > 0


def test_metadata_filters_dont_contain_data_for_missing_metadata_category(data_catalog):
    """Test that get_filters_with_counts() do not contain values for metadata categories that are not present."""
    filters = data_catalog.get_filters_with_counts()
    for metadata_category in ["spatial_coverage", "temporal_coverage"]:
        assert metadata_category in filters  # the metadata category key must be present
        assert not filters[metadata_category]["value_counts"]  # value counts must be an empty dict


def test_distributions_transformed(data_catalog):
    """Test that the distributions dataframe contains expected columns."""
    transformed_columns = data_catalog.config["data_processing"]["distribution_column_mapping"].values()
    assert data_catalog.distributions is not None
    assert sorted(transformed_columns) == sorted(data_catalog.distributions.columns)

