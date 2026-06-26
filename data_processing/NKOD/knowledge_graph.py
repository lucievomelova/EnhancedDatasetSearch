import os

import pandas as pd
from neo4j import GraphDatabase, Driver

from data_processing.database import Database
from utils import setup_logger
from data_processing.knowledge_graph import KnowledgeGraph
from data_processing.NKOD.kg_queries import *

logger = setup_logger(__name__)

class NkodKnowledgeGraph(KnowledgeGraph):
    """Knowledge graph for the NKOD."""
    def __init__(self, kg_config: dict, database: Database):
        self.driver: Driver = GraphDatabase.driver(
            os.environ['NEO4J_URI'],
            auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD'])
        )
        self.kg_config = kg_config
        self.database = database

    def create_kg(self, datasets: pd.DataFrame) -> None:
        """Create knowledge graph based on dataset metadata and description embedding similarity."""
        logger.info(f"Creating knowledge graph from dataset metadata for {len(datasets)} datasets.")

        # delete old data and create constraints on nodes
        with self.driver.session() as session:
            session.run(f"MATCH (n) WHERE n.graph = $graph DETACH DELETE n;", graph=self.kg_config["name"])
            session.run("CREATE CONSTRAINT dataset_url IF NOT EXISTS FOR (d:Dataset) REQUIRE (d.url, d.graph) IS UNIQUE;")
            session.run("CREATE CONSTRAINT keyword_name IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE;")
            session.run("CREATE CONSTRAINT theme_name IF NOT EXISTS FOR (t:Theme) REQUIRE t.name IS UNIQUE;")
            session.run("CREATE CONSTRAINT provider_name IF NOT EXISTS FOR (p:Provider) REQUIRE p.name IS UNIQUE;")
            session.run("CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE;")
            session.run("CREATE CONSTRAINT spatial_coverage_name IF NOT EXISTS FOR (r:SpatialCoverage) REQUIRE r.name IS UNIQUE;")
            session.run("CREATE CONSTRAINT temporal_coverage_name IF NOT EXISTS FOR (t:TemporalCoverage) REQUIRE t.name IS UNIQUE;")

        with self.driver.session() as session:
            for i, (_, row) in enumerate(datasets.iterrows()):
                if i % 500 == 0:
                    logger.info(f"{i}/{len(datasets)}")
                metadata = row.drop(columns="description")
                # add dataset nodes
                session.execute_write(ingest_dataset, metadata, self.kg_config["name"])
                url = row["url"]

                # add keywords, themes, categories, spatial coverage, temporal coverage and provider nodes
                for keyword in row["keywords"]:
                    session.execute_write(ingest_keyword, url, keyword.title(), self.kg_config["name"])
                for theme in row["themes"]:
                    session.execute_write(ingest_theme, url, theme.title(), self.kg_config["name"])
                for category in row["categories"]:
                    session.execute_write(ingest_category, url, category.title(), self.kg_config["name"])
                for spatial_coverage in row["spatial_coverage"]:
                    session.execute_write(ingest_spatial_coverage, url, spatial_coverage.title(), self.kg_config["name"])
                for temporal_coverage in row["temporal_coverage"]:
                    session.execute_write(ingest_temporal_coverage, url, temporal_coverage.title(), self.kg_config["name"])
                session.execute_write(ingest_provider, url, row["provider"].title(), self.kg_config["name"])

        self.add_similarity_edges(datasets, self.kg_config)
        logger.info("Knowledge graph creation completed.")

    def add_similarity_edges(self, datasets: pd.DataFrame, kg_config: dict) -> None:
        """Add similarity edges between datasets based on description embedding and metadata."""
        for i, (_, row) in enumerate(datasets.iterrows()):
            if i % 500 == 0:
                logger.info(f"Adding description similarity edges for dataset {i}")
            # description embedding similarity edges
            similarity_threshold = self.kg_config["similarity_threshold"]
            similar_datasets = self.database.get_similar_datasets_by_embedding(
                row["url"],
                similarity_threshold,
                kg_config["top_k"]
            )
            with self.driver.session() as session:
                session.execute_write(
                    create_description_similarity_edges,
                    row["url"],
                    similar_datasets,
                    kg_config["name"]
                )

    def get_similar_datasets(self, dataset_url: str) -> dict[str, list[tuple[str, float]]]:
        """Get similar datasets based on the knowledge graph.

        Returns:
            A dictionary with keys: 'description', 'keywords', 'themes', 'provider'
            Each value is a list of (url, score) tuples.
        """
        logger.info(f"Retrieving similar datasets based on knowledge graph.")
        similar_datasets = {}

        similarity_types = ["description", "keywords", "themes", "provider"]
        with self.driver.session() as session:
            similar_datasets["description"] = session.execute_read(
                run_similarity_query,
                dataset_url,
                self.kg_config
            )
            logger.info(f"Retrieved {len(similar_datasets["description"])} similar datasets based on description.")

            similar_datasets["keywords"] = get_similar_datasets_based_on_metadata_category(
                session,
                dataset_url,
                "keywords",
                self.kg_config
            )
            logger.info(f"Retrieved {len(similar_datasets["keywords"])} similar datasets based on common keywords.")

            similar_datasets["themes"] = get_similar_datasets_based_on_metadata_category(
                session,
                dataset_url,
                "themes",
                self.kg_config
            )
            logger.info(f"Retrieved {len(similar_datasets["themes"])} similar datasets based on common themes.")

            similar_datasets["provider"] = get_similar_datasets_from_the_same_provider(
                session,
                dataset_url,
                self.kg_config
            )
            logger.info(f"Retrieved {len(similar_datasets["provider"])} similar datasets based on common themes")

        return similar_datasets
