import json
import os
import ollama

from data_processing.database import Database
from neo4j import GraphDatabase, Driver
from jinja2 import Environment, FileSystemLoader

from utils import setup_logger

logger = setup_logger(__name__)
env = Environment(loader=FileSystemLoader('prompts'))
return_json_template = env.get_template("return_json.j2")
return_json_instructions = return_json_template.render()

driver = GraphDatabase.driver(
    os.environ['NEO4J_URI'],
    auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD'])
)


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


def create_keyword_kg(database: Database, keywords: list[str], embed_dim: int) -> None:
    """Embed keywords and create the keyword knowledge graph."""

    logger.info("Creating keyword embeddings.")
    embeddings = embed_keywords(database, keywords)

    logger.info("Creating keyword knowledge graph.")
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

def get_clusters(driver: Driver, clusters_state_file: str, save_to_file: bool = True) -> dict[str, list[str]]:
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
            with open(clusters_state_file, "w") as f:
                json.dump(clusters, f, indent=4, ensure_ascii=False)
        return clusters


def clustering():
    create_clusters(driver)
    get_clusters(driver, save_to_file=True)
    logger.info("Keyword preprocessing completed.")


def find_representatives(clusters_state_file: str, representatives_state_file: str, model_name: str) -> dict[str, list[str]]:
    """Find representative keyword for each cluster."""
    if os.path.exists(clusters_state_file):
        with open(clusters_state_file, "r") as f:
            clusters = json.load(f)
    else:
        clusters = get_clusters(driver, clusters_state_file)
    representatives = {}
    logger.info("Finding representative keyword for each cluster.")
    for cluster_id, cluster_keywords in clusters.items():
        logger.info(f"Processing cluster {cluster_id}")

        template = env.get_template("keywords_clustering.j2")
        prompt = template.render(keywords=", ".join(cluster_keywords), return_json_instructions=return_json_instructions)
        while True:
            try:
                response = ollama.generate(model=model_name,
                                           prompt=prompt,
                                           format="json",
                                           options={
                                               "temperature": 0,
                                           }).response
                current_representatives = json.loads(response)
                break
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {response}. Retrying...")

        for representative, keywords in current_representatives.items():
            representative = representative.strip()
            if representative in representatives:
                representatives[representative] += keywords
            else:
                representatives[representative] = keywords
            print(representative, keywords)

    logger.info("Cluster representatives found.")
    with open(representatives_state_file, "w") as f:
        json.dump(representatives, f, indent=4, ensure_ascii=False)
    with open(representatives_state_file, "r") as f:
        representatives = json.load(f)

    logger.info("Storing cluster representatives in the knowledge graph.")
    for representative, keywords in representatives.items():
        print(representative)
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


def get_representatives(state_dir: str, model_name: str) -> dict[str, list[str]]:
    """Get cluster representatives from file."""
    representatives_state_file = state_dir + "/representatives.json"  # file that stores the cluster representatives found by LLM
    clusters_state_file = state_dir + "/clusters.json"  # file that stores the clusters found by leiden
    if os.path.exists(representatives_state_file):
        with open("representatives.json", "r") as f:
            representatives = json.load(f)
    else:
        representatives = find_representatives(clusters_state_file, representatives_state_file, model_name)
    return representatives

# dataset_portal = NKOD(config)
# database = Database(config["rag"])
# dataset_portal.load()
#
# preprocess_keywords(database, dataset_portal.get_keywords())