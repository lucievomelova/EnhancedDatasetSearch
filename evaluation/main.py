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

from EnhancedDatasetSearch.data_processing.database import Database
from EnhancedDatasetSearch.data_processing.NKOD.nkod import NkodDataCatalog
from EnhancedDatasetSearch.app.pipeline import SearchPipeline

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

    data_catalog = NkodDataCatalog(config)
    database = Database(config)
    search_pipeline = SearchPipeline(config, llm, data_catalog, database)
    search_result_top_k = config["pipeline_config"]["search"]["top_k"]
    with mlflow.start_run() as parent_run:
        for key in config["pipeline_config"]:
            params = {}
            for sub_key in config["pipeline_config"][key]:
                params[f"{key}_{sub_key}"] = config["pipeline_config"][key][sub_key]
            mlflow.log_params(params)
        ndcg, recalls = asyncio.run(evaluate(data_catalog, search_pipeline, search_result_top_k))

        mlflow.log_metric("ndcg_avg", ndcg)
        mlflow.log_metric("recall_avg", sum(recalls)/len(recalls))
        mlflow.log_param("Embedding model", config["embedding"]["model_name"])
        mlflow.log_param("LLM", config["pipeline_config"]["llm"]["model_name"])
        mlflow.log_param("cutoff", config["pipeline_config"]["postprocessing"]["score_cutoff"])


async def evaluate(data_catalog: NkodDataCatalog, search_pipeline: SearchPipeline, search_result_top_k: int) -> tuple[list, list]:
    golden_dataset = pd.read_csv("data/golden/golden_dataset.csv")
    # find all unique queries in golden dataset
    queries = golden_dataset["query"].unique()
    y_pred_all = []
    y_true_all = []
    recalls = []
    ndcgs = []
    for query in queries:
        with mlflow.start_run(nested=True) as child_run:
            logger.info(f"Evaluating query: {query}")
            results_pred = await search_pipeline.run(query)
            results_true = golden_dataset[golden_dataset["query"] == query][["url", "ranking"]]

            k = len(results_pred)
            y_pred = [(k - i) * 1/k for i in range(k)]  # artificial scores based on ranking position
            y_true = [0] * k  # initialize all true relevance scores to 0
            relevant_count = 0
            for _, true_row in results_true.iterrows():
                for i, pred_row in enumerate(results_pred):
                    if pred_row["url"] == true_row["url"]:
                        # more items can have the same ranking - distribute points among them fairly
                        y_true[i] = 1 / true_row["ranking"]
                        relevant_count += 1

            # because the amount of search results is different each time, pad it with 0
            y_pred.extend([0] * (search_result_top_k - k))
            y_true.extend([0] * (search_result_top_k - k))

            y_pred_all.append(y_pred)
            y_true_all.append(y_true)
            recall = relevant_count / len(results_true)
            recalls.append(recall)
            ndcg = ndcg_score(np.array([y_true]), np.array([y_pred]))
            ndcgs.append(ndcg)

            mlflow.log_param("query", query)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("ndcg", ndcg)
            mlflow.log_metric("number_of_results", len(results_pred))
            mlflow.log_metric("number_of_correct", len(results_true))
            for i in range(len(results_pred)):
                mlflow.log_param(f"{i}. result", f"{results_pred[i]["title"]}, {results_pred[i]["url"]}")

    y_pred_all = np.array(y_pred_all)
    y_true_all = np.array(y_true_all)
    ndcg = ndcg_score(y_true_all, y_pred_all)
    logger.info(f"ndcg_score: {ndcg}")
    ndcg = sum(ndcgs) / len(ndcgs)
    logger.info(f"ndcg: {ndcg}, recall: {recalls}")
    return ndcg, recalls


if __name__ == "__main__":
    main()