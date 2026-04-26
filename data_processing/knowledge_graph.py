import os
from collections import defaultdict

import pandas as pd
from neo4j import GraphDatabase, Session

from data_processing.database import Database
from utils import setup_logger


logger = setup_logger(__name__)


def ingest_dataset(tx, doc_id, description, metadata):
    tx.run("""
        MERGE (d:Dataset {id: $id, graph: 'dataset_graph_test'})
        SET d.title = $title,
            d.description = $description,
            d.url = $url
        """,
           id=doc_id, title=metadata["title"], description=description, url=metadata["url"])

def ingest_keyword(tx, dataset_id, keyword):
    tx.run("""
        MATCH (d:Dataset {id: $id, graph: 'dataset_graph_test'})
        MERGE (k:Keyword {name: $kw})
        MERGE (d)-[:HAS_KEYWORD]->(k)
        """, id=dataset_id, kw=keyword)

def ingest_theme(tx, dataset_id, theme):
    tx.run("""
        MATCH (d:Dataset {id: $id, graph: 'dataset_graph_test'})
        MERGE (t:Theme {name: $theme})
        MERGE (d)-[:HAS_THEME]->(t)
        """, id=dataset_id, theme=theme)

def ingest_provider(tx, dataset_id, provider):
    tx.run("""
        MATCH (d:Dataset {id: $id, graph: 'dataset_graph_test'})
        MERGE (p:Provider {name: $provider})
        MERGE (d)-[:PROVIDED_BY]->(p)
        """, id=dataset_id, provider=provider)

def ingest_category(tx, dataset_id, category):
    tx.run("""
        MATCH (d:Dataset {id: $id, graph: 'dataset_graph_test'})
        MERGE (c:Category {name: $category})
        MERGE (d)-[:HAS_CATEGORY]->(c)
        """, id=dataset_id, category=category)

def ingest_spatial_coverage(tx, dataset_id, spatial_coverage):
    tx.run("""
        MATCH (d:Dataset {id: $id, graph: 'dataset_graph_test'})
        MERGE (r:SpatialCoverage {name: $spatial_coverage})
        MERGE (d)-[:HAS_SPATIAL_COVERAGE]->(r)
        """, id=dataset_id, spatial_coverage=spatial_coverage)

def ingest_temporal_coverage(tx, dataset_id, temporal_coverage):
    tx.run("""
        MATCH (d:Dataset {id: $id, graph: 'dataset_graph_test'})
        MERGE (t:TemporalCoverage {name: $temporal_coverage})
        MERGE (d)-[:HAS_TEMPORAL_COVERAGE]->(t)
        """, id=dataset_id, temporal_coverage=temporal_coverage)


def create_description_similarity_edges(tx, dataset_url: str, similar_datasets: dict[str, float]):
    """Create similarity edges between datasets based on embedding similarity."""
    rows = [{"url": similar_dataset_url, "score": score} for similar_dataset_url, score in similar_datasets.items()]
    tx.run(
        """
        MATCH (d1:Dataset {url: $url, graph: 'dataset_graph_test'})
        UNWIND $rows AS row
        MATCH (d2:Dataset {url: row.url, graph: 'dataset_graph_test'})
        MERGE (d1)-[r:SIMILAR]-(d2)
        ON CREATE SET r.score = row.score
        ON MATCH SET r.score = row.score
        """,
        url=dataset_url,
        rows=rows
    )


def create_metadata_similarity_edges(tx, similar_datasets: dict[tuple[str, str], float], metadata_name: str):
    """Create similarity edges between datasets based on metadata similarity."""
    logger.info(f"Adding {len(similar_datasets)} metadata similarity edges for {metadata_name}.")
    batches = []
    batch_size = 10000
    batch = []
    for urls, score in similar_datasets.items():
        url1, url2 = urls
        batch.append({
            "from": url1,
            "to": url2,
            "score": score
        })
        if len(batch) >= batch_size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)

    for i in range(len(batches)):
        batch = batches[i]
        logger.info(f"Adding batch {i}/{len(batches)}.")
        tx.run(
            f"""
            UNWIND $edges AS e
            MATCH (d1:Dataset {{url: e.from, graph: 'dataset_graph_test'}})
            MATCH (d2:Dataset {{url: e.to, graph: 'dataset_graph_test'}})
            MERGE (d1)-[r:SIMILAR]-(d2)
            ON CREATE SET r.{metadata_name}_similarity = e.score
            ON MATCH SET r.{metadata_name}_similarity = e.score
            """,
            edges=batch
        )


def create_kg(datasets: pd.DataFrame, database: Database, kg_config: dict) -> None:
    """Create knowledge graph based on dataset metadata and description embedding similarity."""
    driver = GraphDatabase.driver(
        os.environ['NEO4J_URI'],
        auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD'])
    )
    logger.info(f"Creating knowledge graph from dataset metadata for {len(datasets)} datasets.")

    # delete old data
    with driver.session() as session:
        session.run("MATCH (n) WHERE n.graph = 'dataset_graph_test' DETACH DELETE n;")

    with driver.session() as session:
        session.run("CREATE CONSTRAINT dataset_id IF NOT EXISTS FOR (d:Dataset) REQUIRE d.id IS UNIQUE;")
        session.run("CREATE CONSTRAINT keyword_name IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE;")
        session.run("CREATE CONSTRAINT theme_name IF NOT EXISTS FOR (t:Theme) REQUIRE t.name IS UNIQUE;")
        session.run("CREATE CONSTRAINT provider_name IF NOT EXISTS FOR (p:Provider) REQUIRE p.name IS UNIQUE;")
        session.run("CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE;")
        session.run("CREATE CONSTRAINT spatial_coverage_name IF NOT EXISTS FOR (r:SpatialCoverage) REQUIRE r.name IS UNIQUE;")
        session.run("CREATE CONSTRAINT temporal_coverage_name IF NOT EXISTS FOR (t:TemporalCoverage) REQUIRE t.name IS UNIQUE;")

    with driver.session() as session:
        i = 0
        for index, row in datasets.iterrows():
            if i % 500 == 0:
                logger.info(f"{i}/{len(datasets)}")
            i += 1
            metadata = row.drop(columns="description")
            index += 200000
            session.execute_write(ingest_dataset, index, row["description"], metadata)

            if row["keywords"]:
                for keyword in row["keywords"]:
                    session.execute_write(ingest_keyword, index, keyword.title())
            if row["themes"]:
                for theme in row["themes"]:
                    session.execute_write(ingest_theme, index, theme.title())
            if row["categories"]:
                for category in row["categories"]:
                    session.execute_write(ingest_category, index, category.title())
            if row["spatial_coverage"]:
                for spatial_coverage in row["spatial_coverage"]:
                    session.execute_write(ingest_spatial_coverage, index, spatial_coverage.title())
            if row["temporal_coverage"]:
                for temporal_coverage in row["temporal_coverage"]:
                    session.execute_write(ingest_temporal_coverage, index, temporal_coverage.title())
            if row["provider"] is not None:
                session.execute_write(ingest_provider, index, row["provider"].title())

    add_similarity_edges(datasets, database, driver.session(), kg_config["similarity_threshold"], kg_config["top_k"])
    logger.info("Knowledge graph creation completed.")


def add_similarity_edges(datasets: pd.DataFrame, database: Database, session: Session, similarity_threshold: int, top_k: int) -> None:
    """Add similarity edges between datasets based on description embedding and metadata."""
    i = 0
    for _, row in datasets.iterrows():
        if i % 500 == 0:
            logger.info(f"Adding description similarity edges for dataset {i}")
        i += 1
        # description embedding similarity edges
        similar_datasets = database.get_similar_datasets_by_embedding(row["url"], similarity_threshold, top_k)
        session.execute_write(create_description_similarity_edges, row["url"], similar_datasets)


def _run_similarity_query(tx, dataset_url: str, top_k: int) -> list[tuple[str, float]]:
    """Run a query to get similar datasets based on a specific similarity type."""
    result = tx.run(
        """
        MATCH (d:Dataset {url: $url, graph: 'dataset_graph_test'})-[r:SIMILAR]-(similar:Dataset {graph: 'dataset_graph_test'})
        WHERE r.score IS NOT NULL
        RETURN similar.url AS url, r.score AS sim
        ORDER BY r.score DESC
        LIMIT $top_k
        """,
        url=dataset_url, top_k=top_k
    )
    return [(record["url"], record["sim"]) for record in result]


def _get_similar_datasets_based_on_metadata_category(session: Session, dataset_url: str, metadata_type: str, top_k: int):
    """Retrieve datasets that share the most neighbors of the given metadata category with the specified dataset."""
    relationship = "HAS_" + metadata_type.upper()
    if relationship[-1] == "S":
        relationship = relationship[:-1]  # remove plural form of the metadata
    query = f"""
        MATCH (a:Dataset {{url: $url, graph: 'dataset_graph_test'}})-[:{relationship}]-(metadata_node)
        MATCH (metadata_node)-[:{relationship}]-(b:Dataset {{graph: 'dataset_graph_test'}})
        WHERE b <> a
        RETURN b.url as url, count(metadata_node) AS sharedMetadata,
               collect(DISTINCT metadata_node.name) AS sharedNeighborNames
        ORDER BY sharedMetadata DESC
        LIMIT {top_k}
        """
    result = session.run(query, url=dataset_url)
    return [(record["url"], record["sharedNeighborNames"]) for record in result]


def _get_similar_datasets_based_on_all_metadata(session: Session, dataset_url: str, top_k: int):
    """Retrieve datasets that share the most neighbors when looking at all metadata types."""
    # look at all neighbors but exclude relationships of type "SIMILAR", because there datasets are neighbors directly
    query = """
        MATCH (a:Dataset {url: $url, graph: 'dataset_graph_test'})-[rel]-(metadata_node)
        WHERE type(rel) <> 'SIMILAR'
        MATCH (b:Dataset)-[rel2]-(metadata_node)
        WHERE b <> a
        RETURN b.url as url, count(metadata_node) AS sharedMetadata,
               collect(DISTINCT {name: metadata_node.name, metadata_category: type(rel)}) AS sharedNeighborNames
        ORDER BY sharedMetadata DESC
        LIMIT $top_k
        """
    result = session.run(query, url=dataset_url, top_k=top_k)
    similar_datasets = []
    for record in result:
        shared_metadata = defaultdict(list)
        for neighbor in record["sharedNeighborNames"]:
            metadata_category = neighbor["metadata_category"][4:].lower() + "s"  # remove HAS_, to lower, plural form
            if metadata_category in ["spatial_coverages", "temporal_coverages"]:
                metadata_category = metadata_category[:-1]  # remove plural form for spatial and temporal coverage
            shared_metadata[metadata_category].append(neighbor["name"])
        dataset = {
            "url": record["url"],
            "shared_metadata": shared_metadata
        }
        similar_datasets.append(dataset)
    return similar_datasets


def get_similar_datasets(dataset_url: str, top_k: int, similarity_type: str | None = None) -> dict[str, list[tuple[str, float]]]:
    """Get similar datasets based on the knowledge graph.
    
    Returns:
        A dictionary with keys: 'description', 'keywords', 'themes', 'overall'
        Each value is a list of (url, score) tuples.
    """
    driver = GraphDatabase.driver(
        os.environ['NEO4J_URI'],
        auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD'])
    )
    logger.info(f"Retrieving similar datasets based on knowledge graph.")
    similar_datasets_results = {}

    possible_types = ["description", "keywords", "themes"]
    if similarity_type is None:
        types = possible_types
    else:
        if similarity_type not in possible_types:
            types = possible_types
        else:
            types = [similarity_type]

    with driver.session() as session:
        # similar_datasets_results["overall"] = _get_similar_datasets_based_on_all_metadata(session, dataset_url, top_k)
        if "description" in types:
            similar_datasets_results["description"] = session.execute_read(_run_similarity_query, dataset_url, top_k)
            logger.info("Retrieved similar datasets based on description")
        # if "keywords" in types:
        #     similar_datasets_results["keywords"] = _get_similar_datasets_based_on_metadata_category(session, dataset_url, "keywords", top_k)
        #     logger.info("Retrieved similar datasets based on common keywords")
        if "themes" in types:
            similar_datasets_results["themes"] = _get_similar_datasets_based_on_metadata_category(session, dataset_url, "themes", top_k)
            logger.info("Retrieved similar datasets based on common themes")

    return similar_datasets_results
