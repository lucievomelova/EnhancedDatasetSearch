"""Unit tests for knowledge graph."""

def test_create_kg(config, data_catalog, knowledge_graph):
    """Test that KG creation doesn't throw any errors."""
    knowledge_graph.create_kg(data_catalog.datasets)


def test_all_datasets_in_kg(config, driver):
    graph_name = config["data_processing"]["knowledge_graph"]["name"]
    query = f"MATCH (d:Dataset) WHERE d.graph = $graph_name RETURN count(d) AS count"
    with driver.session() as session:
        number_of_datasets = session.run(query, graph_name=graph_name).single()["count"]
        assert number_of_datasets == 4


def test_metadata_nodes_exist_in_kg(config, driver):
    """Test that metadata nodes exist in KG - at least one keyword, theme, category, provider."""
    graph_name = config["data_processing"]["knowledge_graph"]["name"]
    for node_type in ["Keyword", "Theme", "Category"]:
        rel_type = f"HAS_{node_type}".upper()
        query = f"MATCH (d:Dataset)-[:{rel_type}]->(n:{node_type}) WHERE d.graph = $graph_name RETURN count(DISTINCT n) AS count"
        with driver.session() as session:
            count = session.run(query, graph_name=graph_name).single()["count"]
            assert count > 0

    # provider has a different relationship name
    query = "MATCH (d:Dataset)-[:PROVIDED_BY]->(p:Provider) WHERE d.graph = $graph_name RETURN count(DISTINCT p) AS count"
    with driver.session() as session:
        count = session.run(query, graph_name=graph_name).single()["count"]
        assert count > 0


def test_all_keyword_nodes_exist_in_kg(config, data_catalog, driver):
    """Test that each keyword has a keyword node in the KG and no extra keyword nodes exist."""
    graph_name = config["data_processing"]["knowledge_graph"]["name"]
    # we have to use distinct, otherwise each keyword will be added once for each dataset it is connected to
    query = "MATCH (d:Dataset)-[:HAS_KEYWORD]->(k:Keyword) WHERE d.graph = $graph_name RETURN DISTINCT k.name AS name"
    with driver.session() as session:
        result = session.run(query, graph_name=graph_name)
        keywords = [record["name"].lower() for record in result]
        assert sorted(list(data_catalog.datasets["keywords"].explode().dropna().unique())) == sorted(keywords)


def test_no_similarity_edges_exist(config, driver):
    """Test that no SIMILAR relationships exist in the test KG."""
    graph_name = config["data_processing"]["knowledge_graph"]["name"]
    query = "MATCH (d1:Dataset)-[rel:SIMILAR]->(d2:Dataset) WHERE d1.graph = $graph_name AND d2.graph = $graph_name RETURN count(DISTINCT rel) AS count"
    with driver.session() as session:
        count = session.run(query, graph_name=graph_name).single()["count"]
        assert count == 0


def test_get_similar_datasets(config, driver, knowledge_graph):
    """Test that get_similar_datasets works."""
    similar_datasets = knowledge_graph.get_similar_datasets("http://example.com/ds2")
    assert len(similar_datasets["themes"]) > 0  # there is a similar dataset based on themes
    assert len(similar_datasets["description"]) == 0  # description similarity is not calculated for test KG
