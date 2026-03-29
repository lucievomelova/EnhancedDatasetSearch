import os
from collections import defaultdict
from itertools import combinations

import pandas as pd
from neo4j import GraphDatabase, Session
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data_processing.database import Database
from utils import setup_logger


logger = setup_logger(__name__)


def ingest_dataset(tx, doc_id, description, metadata):
    tx.run("""
        MERGE (d:Dataset {id: $id})
        SET d.title = $title,
            d.description = $description,
            d.url = $url,
            d.graph = 'dataset_graph'
        """,
           id=doc_id, title=metadata["title"], description=description, url=metadata["url"])

def ingest_keyword(tx, dataset_id, keyword):
    tx.run("""
        MATCH (d:Dataset {id: $id})
        MERGE (k:Keyword {name: $kw})
        MERGE (d)-[:HAS_KEYWORD]->(k)
        SET k.graph = 'dataset_graph'
        """, id=dataset_id, kw=keyword)

def ingest_theme(tx, dataset_id, theme):
    tx.run("""
        MATCH (d:Dataset {id: $id})
        MERGE (t:Theme {name: $theme})
        MERGE (d)-[:HAS_THEME]->(t)
        SET t.graph = 'dataset_graph'
        """, id=dataset_id, theme=theme)

def ingest_provider(tx, dataset_id, provider):
    tx.run("""
        MATCH (d:Dataset {id: $id})
        MERGE (p:Provider {name: $provider})
        MERGE (d)-[:PROVIDED_BY]->(p)
        SET p.graph = 'dataset_graph'
        """, id=dataset_id, provider=provider)

def ingest_category(tx, dataset_id, category):
    tx.run("""
        MATCH (d:Dataset {id: $id})
        MERGE (c:Category {name: $category})
        MERGE (d)-[:HAS_CATEGORY]->(c)
        SET c.graph = 'dataset_graph'
        """, id=dataset_id, category=category)

def ingest_region(tx, dataset_id, region):
    tx.run("""
        MATCH (d:Dataset {id: $id})
        MERGE (r:Region {name: $region})
        MERGE (d)-[:HAS_REGION]->(r)
        SET r.graph = 'dataset_graph'
        """, id=dataset_id, region=region)

def ingest_time_period(tx, dataset_id, time_period):
    tx.run("""
        MATCH (d:Dataset {id: $id})
        MERGE (t:TimePeriod {name: $time_period})
        MERGE (d)-[:HAS_TIME_PERIOD]->(t)
        SET t.graph = 'dataset_graph'
        """, id=dataset_id, time_period=time_period)


def create_description_similarity_edges(tx, dataset_url: str, similar_datasets: dict[str, float]):
    """Create similarity edges between datasets based on embedding similarity."""
    rows = [{"url": similar_dataset_url, "score": score} for similar_dataset_url, score in similar_datasets.items()]
    tx.run(
        """
        MATCH (d1:Dataset {url: $url})
        UNWIND $rows AS row
        MATCH (d2:Dataset {url: row.url})
        MERGE (d1)-[r:SIMILAR]-(d2)
        ON CREATE SET r.description_similarity = row.score
        ON MATCH SET r.description_similarity = row.score
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
            MATCH (d1:Dataset {{url: e.from}})
            MATCH (d2:Dataset {{url: e.to}})
            MERGE (d1)-[r:SIMILAR]-(d2)
            ON CREATE SET r.{metadata_name}_similarity = e.score
            ON MATCH SET r.{metadata_name}_similarity = e.score
            """,
            edges=batch
        )


def create_overall_similarity_edges(tx):
    """Add overall similarity score to similarity edges"""
    tx.run("""
        MATCH ()-[r:SIMILAR]-()
        SET r.overall_similarity = 
            COALESCE(r.description_similarity, 0) * 0.3 +
            COALESCE(r.keywords_similarity, 0) * 0.2 +
            COALESCE(r.themes_similarity, 0) * 0.2 + 
            COALESCE(r.categories_similarity, 0) * 0.1 +
            COALESCE(r.time_period_similarity, 0) * 0.1 +
            COALESCE(r.region_similarity, 0) * 0.1
       """)


def create_kg(datasets: pd.DataFrame, database: Database) -> None:
    """Create knowledge graph based on dataset metadata and description embedding similarity."""
    driver = GraphDatabase.driver(
        os.environ['NEO4J_URI'],
        auth=(os.environ['NEO4J_USER'], os.environ['NEO4J_PASSWORD'])
    )
    logger.info(f"Creating knowledge graph from dataset metadata for {len(datasets)} datasets.")

    # delete old data
    # with driver.session() as session:
    #     session.run("MATCH (n) WHERE n.embedding is null DETACH DELETE n;")

    # with driver.session() as session:
    #     session.run("CREATE CONSTRAINT dataset_id IF NOT EXISTS FOR (d:Dataset) REQUIRE d.id IS UNIQUE;")
    #     session.run("CREATE CONSTRAINT keyword_name IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE;")
    #     session.run("CREATE CONSTRAINT theme_name IF NOT EXISTS FOR (t:Theme) REQUIRE t.name IS UNIQUE;")
    #     session.run("CREATE CONSTRAINT provider_name IF NOT EXISTS FOR (p:Provider) REQUIRE p.name IS UNIQUE;")
    #     session.run("CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE;")
    #     session.run("CREATE CONSTRAINT region_name IF NOT EXISTS FOR (r:Region) REQUIRE r.name IS UNIQUE;")
    #     session.run("CREATE CONSTRAINT time_period_name IF NOT EXISTS FOR (t:TimePeriod) REQUIRE t.name IS UNIQUE;")
    #
    # with driver.session() as session:
    #     i = 1
    #     for index, row in datasets.iterrows():
    #         logger.info(f"{i}/{len(datasets)}")
    #         i += 1
    #         metadata = row.drop(columns="description")
    #         session.execute_write(ingest_dataset, index, row["description"], metadata)
    #
    #         if row["keywords"] is not []:
    #             for keyword in row["keywords"]:
    #                 session.execute_write(ingest_keyword, index, keyword.title())
    #         if row["themes"] is not []:
    #             for theme in row["themes"]:
    #                 session.execute_write(ingest_theme, index, theme.title())
    #         if row["categories"] is not []:
    #             for category in row["categories"]:
    #                 session.execute_write(ingest_category, index, category.title())
    #         if row["region"] is not []:
    #             for region in row["region"]:
    #                 session.execute_write(ingest_region, index, region.title())
    #         if row["time_periods"] is not []:
    #             for time_period in row["time_periods"]:
    #                 session.execute_write(ingest_time_period, index, time_period.title())
    #         if row["provider"] is not None:
    #             session.execute_write(ingest_provider, index, row["provider"].title())

    add_similarity_edges(datasets, database, driver.session())
    logger.info("Knowledge graph creation completed.")


def add_similarity_edges(datasets: pd.DataFrame, database: Database, session: Session) -> None:
    """Add similarity edges between datasets based on description embedding and metadata."""
    i = 1
    # for _, row in datasets.iterrows():
    #     logger.info(f"Adding description similarity edges for dataset {i}")
    #     i += 1
    #     # description embedding similarity edges
    #     similar_datasets = database.get_similar_datasets_by_embedding(row["url"])
    #     session.execute_write(create_description_similarity_edges, row["url"], similar_datasets)

    logger.info("Adding metadata similarity edges.")
    # TODO too slow
    # metadata similarity - a score that will be calculated based on common metadata
    # keywords and themes - TF-IDF, to take into account how (un)common some words are
    columns = ["themes"]
    for column in columns:
        logger.info(f"Adding metadata similarity edges for {column}.")
        similar_dataset_pairs_with_score = _compute_tfidf_similarity_based_on_column(datasets, column)
        logger.info(f"TF-IDF vectors computed.")
        # similar_dataset_pairs_with_score = _compute_jaccard_similarity_based_on_column(datasets, column)
        # logger.info(f"Jaccard vectors computed.")
        session.execute_write(create_metadata_similarity_edges, similar_dataset_pairs_with_score, column)

    # time period, region - Jaccard
    # we will skip category - too many edges
    columns = ["time_periods", "region"]
    for column in columns:
        logger.info(f"Adding metadata similarity edges for {column}.")
        similar_dataset_pairs_with_score = _compute_jaccard_similarity_based_on_column(datasets, column)
        logger.info(f"Jaccard vectors computed.")
        session.execute_write(create_metadata_similarity_edges, similar_dataset_pairs_with_score, column)

    session.execute_write(create_overall_similarity_edges)


def _compute_jaccard_similarity_based_on_column(datasets: pd.DataFrame, column_name: str, sim_threshold: int = 0.8, top_k: int = 10) -> dict[tuple[str, str], float]:
    """Compute similarity between datasets based on a specific list column using Jaccard similarity."""

    metadata_dict = {}  # key: dataset url, value: set of items in the column
    occurrences = defaultdict(set)  # for tracking which items occur in which datasets
    for url, words in zip(datasets["url"], datasets[column_name]):
        [occurrences[w].add(url) for w in words]  # key is the word, value is list of dataset urls where the word occurs
        metadata_dict[url] = set(words)

    url_pairs = set()
    for word in occurrences:
        for a, b in combinations(occurrences[word], 2):
            url_pairs.add(tuple(sorted((a, b))))

    top_k_dict = defaultdict(list)
    similarity_dict = {}
    for a, b in url_pairs:
        words_a = metadata_dict[a]
        words_b = metadata_dict[b]
        intersection = len(words_a.intersection(words_b))
        union = len(words_a.union(words_b))
        similarity = intersection / union if union > 0 else 0.0
        if similarity > sim_threshold:
            similarity_dict[(a, b)] = similarity
            top_k_dict[a].append((b, float(similarity)))
            if len(top_k_dict[a]) > top_k:
                min_score_item = min(top_k_dict[a], key=lambda x: x[1])
                max_score_item = max(top_k_dict[a], key=lambda x: x[1])
                if min_score_item[1] == max_score_item[1]:
                    continue  # if all scores are the same, dont remove the min
                top_k_dict[a].remove(min_score_item)
                similarity_dict.pop((a, min_score_item[0]), None)

    return similarity_dict


def _compute_tfidf_similarity_based_on_column(datasets: pd.DataFrame, column_name: str, similarity_threshold: float = 0.8, top_k: int = 10) -> dict[tuple[str, str], float]:
    """Compute similarity between datasets based on a specific list column.

    We will use TF-IDF, to take into account how (un)common some words or phrases are."""
    # for each row, take all items (words and phrases) in the metadata column and join them into one long string, so we
    # can vectorize it and use TF-IDF. Store the result in a dict with dataset url as key and the long string as value
    words_lists = {}  # key: url, value: string with all words and phrases from the column merged into one string
    occurrences = defaultdict(set)  # for tracking which words occur in which datasets

    for url, words in zip(datasets["url"], datasets[column_name]):
        # merge phrases into one word separated by _, so that it will represent one word after joining by space
        words_joined = ["_".join(w.split(" ")) for w in words]
        words_lists[url] = " ".join(words_joined) if words_joined is not None else ""  # join the list of words
        [occurrences[w].add(url) for w in words]  # key is the word, value is list of dataset urls where the word occurs
    logger.info("Joined words into strings for TF-IDF vectorization and counted occurrences.")

    # find datasets sharing at least one word, we will only calculate similarity between those to save time
    pairs_by_dataset = defaultdict(list)  # stores pairs of dataset urls that share at least one word
    for word in occurrences:
        for a, b in combinations(occurrences[word], 2):
            a_sorted, b_sorted = tuple(sorted((a, b)))
            pairs_by_dataset[a_sorted].append(b_sorted)
    logger.info(f"Constructed {len(pairs_by_dataset)} dataset pairs that share {column_name}.")

    vectorizer = TfidfVectorizer()
    words_vectors = vectorizer.fit_transform(list(words_lists.values()))
    logger.info(f"TF-IDF vectors computed, shape: {words_vectors.shape}")

    # assign each vector to the corresponding dataset url, so we can use it to calculate similarity
    url_vector_dict = {url: vector for url, vector in zip(words_lists.keys(), words_vectors)}

    similarity_dict = {}
    top_k_dict = defaultdict(list)  # key: dataset url, value: list of (similar dataset url, similarity score) tuples
    for a, candidates in pairs_by_dataset.items():
        vec_a = url_vector_dict[a]
        vecs_b = vstack([url_vector_dict[b] for b in candidates])
        similarities = cosine_similarity(vec_a, vecs_b)[0]  # compute all similarities for one dataset

        for b, sim in zip(candidates, similarities):
            if sim > similarity_threshold:
                similarity_dict[(a, b)] = float(sim)
                top_k_dict[a].append((b, float(sim)))
                if len(top_k_dict[a]) > top_k:
                    min_score_item = min(top_k_dict[a], key=lambda x: x[1])
                    max_score_item = max(top_k_dict[a], key=lambda x: x[1])
                    if min_score_item[1] == max_score_item[1]:
                        continue  # if all scores are the same, dont remove the min
                    top_k_dict[a].remove(min_score_item)
                    similarity_dict.pop((a, min_score_item[0]), None)
        logger.info(f"Similar nodes for dataset {a} added, similarity dict length: {len(similarity_dict)}")

    logger.info(f"Similarity dict constructed, length: {len(similarity_dict)}")
    return similarity_dict


def _run_similarity_query(tx, dataset_url: str, similarity_type: str, session: Session, top_k: int = 5) -> list[tuple[str, float]]:
    """Run a query to get similar datasets based on a specific similarity type."""
    result = tx.run(
        f"""
        MATCH (d:Dataset {{url: $url}})-[r:SIMILAR]-(similar:Dataset)
        WHERE r.{similarity_type}_similarity IS NOT NULL
        RETURN similar.url AS url, r.{similarity_type}_similarity AS sim
        ORDER BY r.{similarity_type}_similarity DESC
        LIMIT $top_k
        """,
        url=dataset_url, top_k=top_k
    )
    return [(record["url"], record["sim"]) for record in result]


def get_similar_datasets(dataset_url: str, top_k: int = 5) -> dict[str, list[tuple[str, float]]]:
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

    # based on dataset description similairty
    with driver.session() as session:
        similar_datasets_results["overall"] = session.execute_read(_run_similarity_query, dataset_url, "overall", session, top_k)
        similar_datasets_results["description"] = session.execute_read(_run_similarity_query, dataset_url, "description", session, top_k)
        similar_datasets_results["keywords"] = session.execute_read(_run_similarity_query, dataset_url, "keywords", session, top_k)
        similar_datasets_results["themes"] = session.execute_read(_run_similarity_query, dataset_url, "themes", session, top_k)

    return similar_datasets_results
