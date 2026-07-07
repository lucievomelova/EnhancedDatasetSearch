"""Evaluation script for evaluating the search pipeline.
The evaluation is based on a golden dataset of example queries and expected search results."""

import asyncio
import logging

import click
import mlflow
import numpy as np
import pandas as pd
import yaml
from llama_index.llms.ollama import Ollama
from sklearn.metrics import ndcg_score

from EnhancedDatasetSearch.database import Database
from EnhancedDatasetSearch.NKOD.knowledge_graph import NkodKnowledgeGraph
from EnhancedDatasetSearch.NKOD.nkod import NkodDataCatalog
from EnhancedDatasetSearch.search.pipeline import SearchPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


mlflow.set_experiment("SearchPipeline Evaluation")


@click.command()
@click.option('--config_path', default='config.yaml', help='Path to the configuration YAML file.')
def main(config_path: str):
    """Setup and run the evaluation."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    llm = Ollama(model=config['pipeline_config']['llm']['model_name'],
                 context_window=config['pipeline_config']['llm']['context_length'],
                 request_timeout=300)

    database = Database(config)
    knowledge_graph = NkodKnowledgeGraph(config["data_processing"]["knowledge_graph"], database)
    data_catalog = NkodDataCatalog(config, knowledge_graph)
    search_pipeline = SearchPipeline(config, llm, data_catalog, database)
    with mlflow.start_run() as parent_run:
        for key in config["pipeline_config"]:
            params = {}
            for sub_key in config["pipeline_config"][key]:
                params[f"{key}_{sub_key}"] = config["pipeline_config"][key][sub_key]
            mlflow.log_params(params)
        ndcg, recall_avg, recall = asyncio.run(evaluate(search_pipeline))
        mlflow.log_metric("ndcg", ndcg)
        mlflow.log_metric("recall_avg", recall_avg)
        mlflow.log_metric("recall", recall)
        mlflow.log_param("Embedding model", config["embedding"]["model_name"])
        mlflow.log_param("LLM", config["pipeline_config"]["llm"]["model_name"])


async def evaluate(search_pipeline: SearchPipeline) -> tuple[float, float, float]:
    golden_dataset = pd.read_csv("data/golden/golden_dataset.csv")
    queries = golden_dataset["query"].unique()  # get all unique queries from golden dataset
    recalls, ndcgs = [], []
    total_relevant, total_retrieved_relevant = 0, 0
    for query in queries:
        with mlflow.start_run(nested=True) as child_run:
            logger.info(f"Evaluating query: {query}")
            results_pred = await search_pipeline.run(query)
            results_true = golden_dataset[golden_dataset["query"] == query][["url", "ranking"]]

            max_rank = results_true["ranking"].max()
            relevance = {row["url"]: max_rank - row["ranking"] + 1 for _, row in results_true.iterrows()}
            y_true, y_pred = [], []
            relevant_count = 0

            for i, pred_row in enumerate(results_pred):
                rel = relevance.get(pred_row["url"], 0)
                y_true.append(rel)
                if rel > 0:
                    relevant_count += 1
                y_pred.append(len(results_pred) - i)

            recall = relevant_count / len(results_true)
            recalls.append(recall)
            ndcg = ndcg_score(np.array([y_true]), np.array([y_pred]))
            ndcgs.append(ndcg)
            total_relevant += len(results_true)
            total_retrieved_relevant += relevant_count

            mlflow.log_param("query", query)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("ndcg", ndcg)
            mlflow.log_metric("number_of_results", len(results_pred))
            mlflow.log_metric("number_of_correct", len(results_true))
            for i in range(len(results_pred)):
                mlflow.log_param(f"{i}. result", f"{results_pred[i]["title"]}, {results_pred[i]["url"]}")

    ndcg = sum(ndcgs) / len(ndcgs)
    recall_avg = sum(recalls)/len(recalls)
    recall = total_retrieved_relevant / total_relevant
    logger.info(f"ndcg: {ndcg}, recall_avg: {recall_avg}, recall: {recall}")
    return ndcg, recall_avg, recall


if __name__ == "__main__":
    main()
