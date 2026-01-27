import json
import os

import ollama
import yaml

from data_processing.database import Database
from neo4j import GraphDatabase, Driver
from jinja2 import Environment, FileSystemLoader

from data_processing.nkod_datasets import NKOD
from utils import setup_logger

logger = setup_logger(__name__)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

env = Environment(loader=FileSystemLoader('prompts'))
return_json_template = env.get_template("return_json.j2")
return_json_instructions = return_json_template.render()


def preprocess_comma_separated_keywords(comma_separated_keywords: list[str]) -> list[str]:
    """Preprocess keywords containing commas.

    Some keywords like this truly contain commas, but others are actually multiple keywords
    that were formatted incorrectly. Use an LLM to identify and split those."""

    # check if we already processed some keywords
    file = "comma_separated_keywords.json"
    if os.path.exists(file):
        with open(file, "r") as f:
            already_processed_keywords = json.load(f)
    else:
        already_processed_keywords = {}

    new_keywords = set()
    for sequence in comma_separated_keywords:
        if sequence in already_processed_keywords:
            new_keywords.update(already_processed_keywords[sequence])
            continue

        template = env.get_template("keyword_commas.j2")
        prompt = template.render(keyword=sequence, return_json_instructions=return_json_instructions)
        response = ollama.generate(model=config["rag"]["llm"]["model_name"],
                                   prompt=prompt,
                                   format="json",
                                   options={
                                       "temperature": 0,
                                   }).response
        current_new_keywords = json.loads(response)[sequence]  # split sequence into keywords
        print(current_new_keywords)
        new_keywords.update(current_new_keywords)  # update the keyword set

        # save as json, where key is the original sequence, and value is the list of new keywords
        if not os.path.exists(file):
            with open(file, "w") as f:
                json.dump({sequence: current_new_keywords}, f, indent=4, ensure_ascii=False)
        else:
            with open(file, "r") as f:
                content = json.load(f)
                content.update({sequence: current_new_keywords})
            with open(file, "w") as f:
                json.dump(content, f, indent=4, ensure_ascii=False)
    return list(new_keywords)


def embed_keywords(database: Database, keywords: list[str]) -> list[list[float]]:
    """Embed a list of keywords using the database's embedding model."""
    embeddings = database.embedding_model.get_text_embedding_batch(keywords)
    return embeddings


def create_keywords(tx, keywords, embeddings):
    """Create keyword nodes in the knowledge graph."""
    for keyword, embedding in zip(keywords, embeddings):
        if embedding is None or embedding == [] or keyword is None:
            print("null embedding or keywords:", keyword)
            continue
        tx.run(
            """
            MERGE (k:Keyword {text: $text})
            SET k.embedding = $embedding,
            k.graph = 'keyword_graph'
            """,
            text=keyword,
            embedding=embedding
        )

def create_similarity_edges(tx, similarity_threshold):
    """Create similarity edges between keywords based on embedding similarity."""
    tx.run(
        """
        MATCH (k:Keyword)
        WHERE k.embedding IS NOT NULL
        CALL db.index.vector.queryNodes(
            'keyword_embedding_index',
            10,
            k.embedding
        )
        YIELD node AS k2, score
        WHERE k2 <> k AND score >= $threshold
        WITH k, k2, score
        WHERE id(k) < id(k2)
        MERGE (k)-[r:SIMILAR]->(k2)
        SET r.score = score
        """,
        threshold=similarity_threshold
    )


def create_keyword_kg(driver: Driver, keywords: list[str], embeddings: list[list[float]], embed_dim: int) -> None:
    """Create the keyword knowledge graph."""
    with driver.session() as session:
        session.run("MATCH (n) WHERE n.embedding is not null DETACH DELETE n;")
        session.execute_write(create_keywords, keywords, embeddings)
        create_index_query = f"""
            CREATE VECTOR INDEX keyword_embedding_index IF NOT EXISTS
            FOR (k:Keyword)
            ON (k.embedding)
            OPTIONS {{
              indexConfig: {{
                `vector.dimensions`: {embed_dim},
                `vector.similarity_function`: 'cosine'
              }}
            }};"""
        session.run(create_index_query)
        session.execute_write(create_similarity_edges, similarity_threshold=0.9)


def create_clusters(driver: Driver) -> None:
    """Create clusters of similar keywords in the knowledge graph."""
    with driver.session() as session:
        session.run("CALL gds.graph.drop('keywordGraph') YIELD graphName;")
        exists = session.run("CALL gds.graph.exists('keywordGraph') YIELD exists RETURN exists").single()["exists"]
        if not exists:
            session.run("""
                CALL gds.graph.project(
                    'keywordGraph',
                    'Keyword',
                    {
                        SIMILAR: {
                            type: 'SIMILAR',
                            orientation: 'UNDIRECTED',
                            properties: 'score'
                        }
                    }
                )
                """
            )
        # use the Leiden algorithm for clustering
        session.run("""
            CALL gds.leiden.write(
            'keywordGraph',
            {
                writeProperty: 'clusterId',
                relationshipWeightProperty: 'score',
                maxLevels: 2,
                gamma: 7.0
            })
            """
        )

def get_clusters(driver: Driver, save_to_file: bool = True) -> dict[str, list[str]]:
    """Get clusters of keywords from the knowledge graph."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (k:Keyword)
            WITH k.clusterId AS cluster, collect(k.text) AS keywords
            RETURN {cluster: cluster, keywords: keywords} AS clusterMap;
            """
        )
        clusters = {}
        num_clusters = 0
        for record in result:
            cluster_data = record["clusterMap"]
            if cluster_data["keywords"]:  # skip empty clusters
                clusters[str(cluster_data["cluster"])] = cluster_data["keywords"]
                num_clusters += 1
        logger.info(f"Divided keywords into {num_clusters} clusters.")

        if save_to_file:
            with open("clusters.json", "w") as f:
                json.dump(clusters, f, indent=4, ensure_ascii=False)
        return clusters


def preprocess_keywords(database: Database, keywords: list[str]) -> None:
    """Preprocess and embed a list of keywords."""
    logger.info("Preprocessing %d keywords.", len(keywords))
    keywords = [kw.strip() for kw in keywords if kw.strip()]

    comma_separated_keywords = [kw for kw in keywords if "," in kw]
    logger.info("Preprocessing %d keywords containing commas", len(comma_separated_keywords))
    processed_comma_sep_keywords = preprocess_comma_separated_keywords(comma_separated_keywords)

    all_keywords = list(set([kw for kw in keywords if "," not in kw] + processed_comma_sep_keywords))
    all_keywords = [kw for kw in all_keywords if kw and kw != ""]

    logger.info("Creating keyword embeddings.")
    keyword_embeddings = embed_keywords(database, all_keywords)

    driver = GraphDatabase.driver(
        os.environ['NEO4J_URI'],
        auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD'])
    )
    logger.info("Creating keyword knowledge graph.")
    create_keyword_kg(driver, all_keywords, keyword_embeddings, database.vector_store.embed_dim)

    logger.info("Clustering keywords.")

def clustering():
    driver = GraphDatabase.driver(
        os.environ['NEO4J_URI'],
        auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD'])
    )
    create_clusters(driver)
    get_clusters(driver, save_to_file=True)
    logger.info("Keyword preprocessing completed.")


def find_representatives() -> dict[str, list[str]]:
    """Find representative keyword for each cluster."""
    driver = GraphDatabase.driver(
        os.environ['NEO4J_URI'],
        auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD'])
    )

    clusters = get_clusters(driver)
    representatives = {}
    logger.info("Finding representative keyword for each cluster.")
    for cluster_id, cluster_keywords in clusters.items():
        logger.info(f"Processing cluster {cluster_id}")

        template = env.get_template("keywords_clustering.j2")
        prompt = template.render(keywords=", ".join(cluster_keywords), return_json_instructions=return_json_instructions)
        response = ollama.generate(model=config["rag"]["llm"]["model_name"],
                                   prompt=prompt,
                                   format="json",
                                   options={
                                       "temperature": 0,
                                   }).response
        current_representatives = json.loads(response)

        for representative, keywords in current_representatives.items():
            representative = representative.strip()
            if representative in representatives:
                representatives[representative] += keywords
            else:
                representatives[representative] = keywords
            print(representative, keywords)

    logger.info("Cluster representatives found.")
    with open("representatives.json", "w") as f:
        json.dump(representatives, f, indent=4, ensure_ascii=False)

    logger.info("Storing cluster representatives in the knowledge graph.")
    for representative, keywords in representatives.items():
        for keyword in keywords:
            with driver.session() as session:
                session.run(
                    """
                    MATCH (k:Keyword {text: $text})
                    MERGE (c:Cluster {text: $representative})
                    SET c.graph = 'keyword_graph'
                    MERGE (c)-[:REPRESENTS]->(k);
                    """,
                    text=keyword,
                    representative=representative
                )
    return representatives

# dataset_portal = NKOD(config)
# database = Database(config["rag"])
# dataset_portal.load()
#
# preprocess_keywords(database, dataset_portal.get_keywords())