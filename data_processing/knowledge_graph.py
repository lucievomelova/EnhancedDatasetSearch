import os

import pandas as pd
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

def ingest_category(tx, dataset_id, category):
    tx.run("""
        MATCH (d:Dataset {id: $id})
        MERGE (c:Category {name: $category})
        MERGE (d)-[:HAS_CATEGORY]->(c)
        """, id=dataset_id, category=category)

def ingest_region(tx, dataset_id, region):
    tx.run("""
        MATCH (d:Dataset {id: $id})
        MERGE (r:Region {name: $region})
        MERGE (d)-[:HAS_REGION]->(r)
        """, id=dataset_id, region=region)

def ingest_time_period(tx, dataset_id, time_period):
    tx.run("""
        MATCH (d:Dataset {id: $id})
        MERGE (t:TimePeriod {name: $time_period})
        MERGE (d)-[:HAS_TIME_PERIOD]->(t)
        """, id=dataset_id, time_period=time_period)

def create_kg(datasets: pd.DataFrame) -> None:
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
        session.run("CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE;")
        session.run("CREATE CONSTRAINT region_name IF NOT EXISTS FOR (r:Region) REQUIRE r.name IS UNIQUE;")
        session.run("CREATE CONSTRAINT time_period_name IF NOT EXISTS FOR (t:TimePeriod) REQUIRE t.name IS UNIQUE;")

    with driver.session() as session:
        i = 1
        for index, row in datasets.iterrows():
            logger.info(f"{i}/{len(datasets)}")
            i += 1
            metadata = row.drop(columns="description")
            session.execute_write(ingest_dataset, index, row["description"], metadata)
            if row["keywords"] is not []:
                for keyword in row["keywords"]:
                    session.execute_write(ingest_keyword, index, keyword.title())
            if row["themes"] is not []:
                for theme in row["themes"]:
                    session.execute_write(ingest_theme, index, theme.title())
            if row["categories"] is not []:
                for category in row["categories"]:
                    session.execute_write(ingest_category, index, category.title())
            if row["region"] is not []:
                for region in row["region"]:
                    session.execute_write(ingest_region, index, region.title())
            if row["time_period"] is not []:
                for time_period in row["time_period"]:
                    session.execute_write(ingest_time_period, index, time_period.title())
            if row["provider"] is not None:
                session.execute_write(ingest_provider, index, row["provider"].title())

    logger.info("Knowledge graph creation completed.")