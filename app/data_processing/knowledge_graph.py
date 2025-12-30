import os

from llama_index.core import Document
from neo4j import GraphDatabase
from utils import setup_logger


logger = setup_logger(__name__)


def ingest_dataset(tx, doc_id, description, metadata):
    tx.run("""
        MERGE (d:Dataset {id: $id})
        SET d.title = $title,
            d.description = $description,
            d.url = $url
        """,  id=doc_id, title=metadata["title"], description=description, url=metadata["url"])

def ingest_keyword(tx, dataset_id, keyword):
    tx.run("""
        MATCH (d:Dataset {id: $id})
        MERGE (k:Keyword {name: $kw})
        MERGE (d)-[:HAS_KEYWORD]->(k)
        """, id=dataset_id, kw=keyword)

def ingest_theme(tx, dataset_id, theme):
    tx.run("""
        MATCH (d:Dataset {id: $id})
        MERGE (t:Theme {name: $theme})
        MERGE (d)-[:HAS_THEME]->(t)
        """, id=dataset_id, theme=theme)

def ingest_provider(tx, dataset_id, provider):
    tx.run("""
        MATCH (d:Dataset {id: $id})
        MERGE (p:Provider {name: $provider})
        MERGE (d)-[:PROVIDED_BY]->(p)
        """, id=dataset_id, provider=provider)

def create_kg(datasets: list[Document]) -> None:
    """Create knowledge graph from list of llama index Documents."""
    driver = GraphDatabase.driver(
        os.environ['NEO4J_URI'],
        auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD'])
    )
    logger.info(f"Creating knowledge graph from dataset metadata for {len(datasets)} datasets.")

    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n;")
    with driver.session() as session:
        session.run("CREATE CONSTRAINT dataset_id IF NOT EXISTS FOR (d:Dataset) REQUIRE d.id IS UNIQUE;")
        session.run("CREATE CONSTRAINT keyword_name IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE;")
        session.run("CREATE CONSTRAINT theme_name IF NOT EXISTS FOR (t:Theme) REQUIRE t.name IS UNIQUE;")
        session.run("CREATE CONSTRAINT provider_name IF NOT EXISTS FOR (p:Provider) REQUIRE p.name IS UNIQUE;")

    with driver.session() as session:
        i = 1
        for ds in datasets:
            logger.info(f"{i}/{len(datasets)}")
            i += 1
            session.execute_write(ingest_dataset, ds.id_, ds.text, ds.metadata)
            if ds.metadata["keywords"] is not None:
                for keyword in ds.metadata["keywords"]:
                    session.execute_write(ingest_keyword, ds.id_, keyword.lower())
            if ds.metadata["themes"] is not None:
                for theme in ds.metadata["themes"]:
                    session.execute_write(ingest_theme, ds.id_, theme.lower())
            if ds.metadata["provider"] is not None:
                session.execute_write(ingest_provider, ds.id_, ds.metadata["provider"].lower())

    logger.info("Knowledge graph creation completed.")