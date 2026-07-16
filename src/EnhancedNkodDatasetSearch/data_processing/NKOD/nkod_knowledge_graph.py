import os

import pandas as pd
from neo4j import Driver, GraphDatabase

from EnhancedNkodDatasetSearch.data_processing.NKOD.kg_queries import *
from EnhancedNkodDatasetSearch.data_processing.database import Database
from EnhancedNkodDatasetSearch.data_processing.knowledge_graph import KnowledgeGraph
from EnhancedNkodDatasetSearch.utils import setup_logger
from pandas import Series

logger = setup_logger(__name__)

class NkodKnowledgeGraph(KnowledgeGraph):
    """Knowledge graph for the NKOD.

    The knowledge graph contains nodes representing datasets and datasets' metadata (e.g. keywords, categories...).
    Each dataset is connected to all metadata nodes that it contains as its metadata."""
    def __init__(self, kg_config: dict, database: Database):
        self.driver: Driver = GraphDatabase.driver(
            os.environ['NEO4J_URI'],
            auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD'])
        )
        self.kg_config = kg_config
        self.database = database

    def _add_dataset_node(self, row: Series, graph_name: str) -> None:
        """Add a dataset node to KG represented by a dataframe row and create all its metadata relationships.

        Create a new dataset node if it doesn't exist, then create its metadata nodes if they don't exist and
        connect the dataset with its metadata by the appropriate relationships."""
        with self.driver.session() as session:
            metadata = row.drop(columns="description")
            # add dataset node
            session.execute_write(create_dataset_node, metadata, graph_name)
            url = row["url"]

            # add keywords, themes, categories, spatial coverage, temporal coverage and provider nodes
            for keyword in row["keywords"]:
                session.execute_write(create_keyword_node, url, keyword.title(), graph_name)
            for theme in row["themes"]:
                session.execute_write(create_theme_node, url, theme.title(), graph_name)
            for category in row["categories"]:
                session.execute_write(create_category_node, url, category.title(), graph_name)
            for spatial_coverage in row["spatial_coverage"]:
                session.execute_write(
                    create_spatial_coverage_node,
                    url,
                    spatial_coverage.title(),
                    graph_name
                )
            for temporal_coverage in row["temporal_coverage"]:
                session.execute_write(
                    create_temporal_coverage_node,
                    url,
                    temporal_coverage.title(),
                    graph_name
                )
            session.execute_write(create_provider_node, url, row["provider"].title(), graph_name)

    def create_or_update_kg(self, datasets: pd.DataFrame, new_datasets: pd.DataFrame, removed_urls: list) -> None:
        """Create a knowledge graph or update it if it exists."""
        if len(new_datasets) == len(datasets):
            self.create_kg(datasets)  # create KG if all datasets are new
        else:
            self.update_kg(datasets, new_datasets, removed_urls)

    def create_kg(self, datasets: pd.DataFrame) -> None:
        """Create knowledge graph based on dataset metadata and description embedding similarity.

        Steps:
            1. Create constraints if they do not exist.
            2. Create a staging KG:
                1. Create dataset nodes and metadata nodes and connect datasets to their metadata.
                2. Create description similarity edges btween datasets.
            3. Delete old live graph if it exists.
            4. Rename staging graph to live.

        This method intended to be used only on the first creation of the KG. Then update_kg() can be used to update the
        existing graph. But this method can also be used to recreate the KG completely if needed.
        A new staging KG is created and only after it is ready, the old live KG is deleted and the staging KG is renamed
        to live. This way, there is no downtime and the old live graph can be used while the new one is being created.
        """
        logger.info(f"Creating knowledge graph from dataset metadata for {len(datasets)} datasets.")

        # create constraints on nodes
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT dataset_url IF NOT EXISTS FOR (d:Dataset) REQUIRE (d.url, d.graph) IS UNIQUE;")
            session.run("CREATE CONSTRAINT keyword_name IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE;")
            session.run("CREATE CONSTRAINT theme_name IF NOT EXISTS FOR (t:Theme) REQUIRE t.name IS UNIQUE;")
            session.run("CREATE CONSTRAINT provider_name IF NOT EXISTS FOR (p:Provider) REQUIRE p.name IS UNIQUE;")
            session.run("CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE;")
            session.run("CREATE CONSTRAINT spatial_coverage_name IF NOT EXISTS FOR (r:SpatialCoverage) REQUIRE r.name IS UNIQUE;")
            session.run("CREATE CONSTRAINT temporal_coverage_name IF NOT EXISTS FOR (t:TemporalCoverage) REQUIRE t.name IS UNIQUE;")

        # use a temporary graph name to create the new KG so that the old one is still avaiable and there is no downtime
        # once the new KG is created, delete the old one and rename the new on
        tmp_graph_name = f"{self.kg_config["name"]}_staging"
        logger.info(f"Creating knowledge graph {tmp_graph_name}.")

        for i, (_, row) in enumerate(datasets.iterrows()):
            if i % 500 == 0:
                logger.info(f"{i}/{len(datasets)}")
            self._add_dataset_node(row, tmp_graph_name)

        self.add_similarity_edges(datasets, tmp_graph_name)

        # delete old graph and rename the new one
        logger.info(f"Deleting old graph {self.kg_config["name"]} and swapping staging graph to live.")
        with self.driver.session() as session:
            session.run(f"MATCH (n) WHERE n.graph = $graph DETACH DELETE n;", graph=self.kg_config["name"])
            session.run(f"MATCH (n) WHERE n.graph = $graph_old SET n.graph = $graph_new;", graph_old=tmp_graph_name, graph_new=self.kg_config["name"])
        logger.info("Knowledge graph creation completed.")

    def update_kg(self, datasets: pd.DataFrame, new_datasets: pd.DataFrame, removed_urls: list):
        """Update an existing knowledge graph by removing old and adding new datasets.

        Steps:
        1. All dataset nodes whose url is in removed_urls are removed and all their relationships as well.
        2. If a dataset in new_datasets already exists in the KG, it is removed frst and then added again.
        3. All datasets in new_datasets are added to KG.
        4. Before deleting dataset nodes in 1. and 2., we keep track of all their neighboring dataset nodes. We
        add similarity edges to those neighboring datasets.
        """
        # remove removed datasets by url
        logger.info(f"Removing {len(removed_urls)} datasets from the knowledge graph.")
        deleted_neighbor_urls = []
        if removed_urls is not None:
            for url in removed_urls:
                with self.driver.session() as session:
                    # first find neighbors of nodes to be deleted
                    deleted_neighbor_urls = get_similar_neighbors_urls(session, url, self.kg_config["name"])
                    # delete nodes
                    session.run("MATCH (n:Dataset) WHERE n.url = $url and n.graph = $graph DETACH DELETE n;", url=url, graph=self.kg_config["name"])
        logger.info(f"Adding {len(new_datasets)} new datasets to the knowledge graph.")
        for _, row in new_datasets.iterrows():
            with self.driver.session() as session:
                # new datasets contain new or updated datasets, so we must first remove dataset nodes that already
                # exist in the KG, they will be added again

                # first find neighbors of nodes to be deleted
                deleted_neighbor_urls.extend(get_similar_neighbors_urls(session, row["url"], self.kg_config["name"]))
                # delete node if it already exists in the KG
                session.run("MATCH (n:Dataset) WHERE n.url = $url and n.graph = $graph DETACH DELETE n;", url=row["url"], graph=self.kg_config["name"])
            self._add_dataset_node(row, self.kg_config["name"])  # add node to the KG

        logger.info(f"Creating similarity edges for {len(new_datasets)} new datasets.")
        self.add_similarity_edges(new_datasets, self.kg_config["name"])

        # add similarity edges to datasets that lost a neighbor
        logger.info(f"Creating similarity edges for {len(deleted_neighbor_urls)} datasets that lost a neighbor.")
        deleted_neighbor_rows = datasets[datasets["url"].isin(deleted_neighbor_urls)]
        self.add_similarity_edges(deleted_neighbor_rows, self.kg_config["name"])
        logger.info("Knowledge graph creation complete.")

    def add_similarity_edges(self, datasets: pd.DataFrame, graph_name: str) -> None:
        """Add similarity edges between datasets based on description embedding similarity."""
        for i, (_, row) in enumerate(datasets.iterrows()):
            if i % 500 == 0:
                logger.info(f"Adding description similarity edges for dataset {i}")
            # description embedding similarity edges
            similar_datasets = self.database.get_similar_datasets_by_embedding(
                row["url"],
                self.kg_config["similarity_threshold"],
                self.kg_config["top_k"]
            )
            with self.driver.session() as session:
                session.execute_write(
                    create_description_similarity_edges,
                    row["url"],
                    similar_datasets,
                    graph_name
                )

    def get_similar_datasets(self, dataset_url: str) -> dict[str, list[tuple[str, float]]]:
        """Get similar datasets based on the knowledge graph.

        Returns:
            A dictionary with keys: 'description', 'keywords', 'themes', 'provider', 'spatial_coverage',
            'temporal_coverage'. Each value is a list of (url, score) tuples.
        """
        logger.info(f"Retrieving similar datasets based on knowledge graph.")

        # set the order of similarity types, so that more important types are closer to the top
        similar_datasets = {
            "description": [],
            "themes": [],
            "keywords": [],
            "provider": [],
            "spatial_coverage": [],
            "temporal_coverage": []
        }

        with self.driver.session() as session:
            # description similarity
            similar_datasets["description"] = session.execute_read(
                run_similarity_query,
                dataset_url,
                self.kg_config
            )
            logger.info(f"Retrieved {len(similar_datasets["description"])} similar datasets based on description.")
            # provider similarity
            similar_datasets["provider"] = get_similar_datasets_from_the_same_provider(
                session,
                dataset_url,
                self.kg_config
            )
            logger.info(f"Retrieved {len(similar_datasets["provider"])} similar datasets based on common themes")
            # other metadata cataegories - keywords, themes, spatial and temporal coverage
            for metadata_category in ["keywords", "themes", "spatial_coverage", "temporal_coverage"]:
                similar_datasets[metadata_category] = get_similar_datasets_based_on_metadata_category(
                    session,
                    dataset_url,
                    metadata_category,
                    self.kg_config
                )
                logger.info(f"Retrieved {len(similar_datasets[metadata_category])} "
                            f"similar datasets based on common {metadata_category}.")

        return similar_datasets
