"""Unit tests for database."""
from datetime import datetime

import pytest


@pytest.mark.dependency()
def test_load_new_documents(data_processing_pipeline, database):
    """Test that load_documents doesn't throw any errors."""
    datasets_documents = data_processing_pipeline.prepare_documents_for_upload(data_processing_pipeline.datasets)
    database.load_documents(datasets_documents)


@pytest.mark.dependency(depends=["test_load_new_documents"])
def test_live_tables_exist_after_load(config, database, connection):
    """Test that after load, live tables exist."""
    with connection.cursor() as cur:
        vector_table = "data_" + config["db"]["vector_table"]
        document_table = "data_" + config["db"]["document_table"]
        for table in [vector_table, document_table]:
            cur.execute("""
                        SELECT EXISTS (SELECT 1 FROM information_schema.tables 
                        WHERE table_schema = 'public' AND table_name = %s);
                        """,
                        (table,))
            result = cur.fetchone()[0]
            assert result


@pytest.mark.dependency(depends=["test_load_new_documents"])
def test_old_tables_exist_after_load(config, database, connection):
    """Test that after load, old live tables were renamed to _old."""
    with connection.cursor() as cur:
        vector_table_old = "data_" + config["db"]["vector_table"] + "_old"
        document_table_old = "data_" + config["db"]["document_table"] + "_old"
        for table in [vector_table_old, document_table_old]:
            cur.execute("""
                        SELECT EXISTS (SELECT 1 FROM information_schema.tables 
                        WHERE table_schema = 'public' AND table_name = %s);
                        """,
                        (table,))
            result = cur.fetchone()[0]
            assert result


@pytest.mark.dependency(depends=["test_load_new_documents"])
def test_staging_tables_dont_exist_after_load(config, database, connection):
    """Test that after load, staging tables do not exist anymore."""
    # we don't know the exact time that staging tables were created, but it was today,
    # so we will check for that and then put wildcard %
    time = datetime.now().strftime("%y_%m_%d")
    staging_suffix_partial = f"_{time}%"
    with connection.cursor() as cur:
        vector_table_old = "data_" + config["db"]["vector_table"] + staging_suffix_partial
        document_table_old = "data_" + config["db"]["document_table"] + staging_suffix_partial
        for table in [vector_table_old, document_table_old]:
            cur.execute("""
                        SELECT EXISTS (SELECT 1 FROM information_schema.tables 
                        WHERE table_schema = 'public' AND table_name LIKE %s);
                        """,
                        (table,))
            result = cur.fetchone()[0]
            assert not result


@pytest.mark.dependency(depends=["test_load_new_documents"])
def test_url_to_node_id_mapping(data_catalog, database):
    """Test that url_to_node_id_mapping contains mapping for all datasets."""
    urls = data_catalog.datasets["url"].tolist()
    for url in urls:
        assert url in database.url_to_node_id_mapping
