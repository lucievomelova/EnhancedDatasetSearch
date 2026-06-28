"""File with functions performing Cypher (Neo4j querying language) queries."""

from collections import defaultdict

from neo4j import Session


def create_dataset_node(tx, metadata, graph_name):
    """Create a dataset node in the knowledge graph."""
    tx.run("""
        MERGE (d:Dataset {url: $url, graph: $graph})
        SET d.title = $title
        """,
           title=metadata["title"], url=metadata["url"], graph=graph_name)


def create_keyword_node(tx, dataset_url, keyword, graph_name):
    """Create a keyword node in the knowledge graph if it doesn't exist and connect it to the specified dataset node."""
    tx.run("""
        MATCH (d:Dataset {url: $url, graph: $graph})
        MERGE (k:Keyword {name: $kw})
        MERGE (d)-[:HAS_KEYWORD]->(k)
        """, url=dataset_url, kw=keyword, graph=graph_name)


def create_theme_node(tx, dataset_url, theme, graph_name):
    """Create a theme node in the knowledge graph if it doesn't exist and connect it to the specified dataset node."""
    tx.run("""
        MATCH (d:Dataset {url: $url, graph: $graph})
        MERGE (t:Theme {name: $theme})
        MERGE (d)-[:HAS_THEME]->(t)
        """, url=dataset_url, theme=theme, graph=graph_name)


def create_provider_node(tx, dataset_url, provider, graph_name):
    """Create a provider node in the knowledge graph if it doesn't exist and connect it to the specified dataset node."""
    tx.run("""
        MATCH (d:Dataset {url: $url, graph: $graph})
        MERGE (p:Provider {name: $provider})
        MERGE (d)-[:PROVIDED_BY]->(p)
        """, url=dataset_url, provider=provider, graph=graph_name)


def create_category_node(tx, dataset_url, category, graph_name):
    """Create a category node in the knowledge graph if it doesn't exist and connect it to the specified dataset node."""
    tx.run("""
        MATCH (d:Dataset {url: $url, graph: $graph})
        MERGE (c:Category {name: $category})
        MERGE (d)-[:HAS_CATEGORY]->(c)
        """, url=dataset_url, category=category, graph=graph_name)


def create_spatial_coverage_node(tx, dataset_url, spatial_coverage, graph_name):
    """Create a spatial_coverage node in the knowledge graph if it doesn't exist and connect it to the specified dataset node."""
    tx.run("""
        MATCH (d:Dataset {url: $url, graph: $graph})
        MERGE (r:SpatialCoverage {name: $spatial_coverage})
        MERGE (d)-[:HAS_SPATIAL_COVERAGE]->(r)
        """, url=dataset_url, spatial_coverage=spatial_coverage, graph=graph_name)


def create_temporal_coverage_node(tx, dataset_url, temporal_coverage, graph_name):
    """Create a temporal_coverage node in the knowledge graph if it doesn't exist and connect it to the specified dataset node."""
    tx.run("""
        MATCH (d:Dataset {url: $url, graph: $graph})
        MERGE (t:TemporalCoverage {name: $temporal_coverage})
        MERGE (d)-[:HAS_TEMPORAL_COVERAGE]->(t)
        """, url=dataset_url, temporal_coverage=temporal_coverage, graph=graph_name)


def create_description_similarity_edges(tx, dataset_url: str, similar_datasets: dict[str, float], graph_name: str):
    """Create similarity edges between datasets based on embedding similarity."""
    rows = [{"url": similar_dataset_url, "score": score} for similar_dataset_url, score in similar_datasets.items()]
    tx.run(
        """
        MATCH (d1:Dataset {url: $url, graph: $graph})
        UNWIND $rows AS row
        MATCH (d2:Dataset {url: row.url, graph: $graph})
        MERGE (d1)-[r:SIMILAR]-(d2)
        ON CREATE SET r.score = row.score
        ON MATCH SET r.score = row.score
        """,
        url=dataset_url, rows=rows, graph=graph_name
    )

def run_similarity_query(tx, dataset_url: str, kg_config: dict) -> list[tuple[str, float]]:
    """Run a query to get similar datasets based on a specific similarity type."""
    result = tx.run(
        """
        MATCH (d:Dataset {url: $url, graph: $graph})-[r:SIMILAR]-(similar:Dataset {graph: $graph})
        WHERE r.score IS NOT NULL
        RETURN similar.url AS url, r.score AS sim
        ORDER BY r.score DESC
        LIMIT $top_k
        """,
        url=dataset_url, top_k=kg_config["top_k"], graph=kg_config["name"]
    )
    return [(record["url"], record["sim"]) for record in result]


def get_similar_datasets_based_on_metadata_category(session: Session, dataset_url: str, metadata_type: str, kg_config: dict) -> list:
    """Retrieve datasets that share the most neighbors of the given metadata category with the specified dataset."""
    relationship = "HAS_" + metadata_type.upper()
    if relationship[-1] == "S":
        relationship = relationship[:-1]  # remove plural form of the metadata
    query = f"""
        MATCH (a:Dataset {{url: $url, graph: $graph}})-[:{relationship}]-(metadata_node)
        MATCH (metadata_node)-[:{relationship}]-(b:Dataset {{graph: $graph}})
        WHERE b <> a
        RETURN b.url as url, count(metadata_node) AS sharedMetadata,
               collect(DISTINCT metadata_node.name) AS sharedNeighborNames
        ORDER BY sharedMetadata DESC
        LIMIT {kg_config["top_k"]}
        """
    result = session.run(query, url=dataset_url, graph=kg_config["name"])
    return [(record["url"], record["sharedNeighborNames"]) for record in result]


def get_similar_datasets_from_the_same_provider(session: Session, dataset_url: str, kg_config: dict) -> list:
    """Retrieve datasets from the same provider that share the most of other metadata."""
    query = f"""
        MATCH (a:Dataset {{url: $url, graph: $graph}})-[:PROVIDED_BY]-(provider)
        MATCH (provider)-[:PROVIDED_BY]-(b:Dataset {{graph: $graph}})
        WHERE b <> a
        
        MATCH (a)-[rel_a]->(m)
        MATCH (b)-[rel_b]->(m)
        WHERE type(rel_a) STARTS WITH 'HAS_'
          AND type(rel_b) = type(rel_a)
        
        RETURN
            b.url AS url,
            count(DISTINCT m) AS sharedMetadata,
            collect(DISTINCT {{
                type: type(rel_a),
                name: coalesce(m.name, m.id)
            }}) AS sharedNeighborNames
        ORDER BY sharedMetadata DESC
        LIMIT {kg_config["top_k"]}
        """
    result = session.run(query, url=dataset_url, graph=kg_config["name"])
    return [(record["url"], record["sharedNeighborNames"]) for record in result]
