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

from app.pipeline import SearchPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


mlflow.set_experiment("SearchPipeline Evaluation")

@click.command()
@click.option('--config', default='config.yaml', help='Path to the configuration YAML file.')
def main(config: str):
    """Setup and run the evaluation."""
    with open(config, "r") as f:
        config = yaml.safe_load(f)

    llm = Ollama(model=config['chatbot']['llm']['model_name'],
                 context_window=config['chatbot']['llm']['context_length'],
                 request_timeout=300)

    search_pipeline = SearchPipeline(config, llm)
    search_result_top_k = config["pipeline_config"]["search"]["top_k"]
    with mlflow.start_run() as parent_run:
        for key in config["pipeline_config"]:
            params = {}
            for sub_key in config["pipeline_config"][key]:
                params[f"{key}_{sub_key}"] = config["pipeline_config"][key][sub_key]
            mlflow.log_params(params)
        ndcg, recalls = asyncio.run(evaluate(search_pipeline, search_result_top_k))

        mlflow.log_metric("ndcg", ndcg)
        mlflow.log_metric("recall_avg", sum(recalls)/len(recalls))
        mlflow.log_param("Embedding model", config["embedding"]["model_name"])
        mlflow.log_param("LLM", config["llm"]["model_name"])


async def evaluate(search_pipeline: SearchPipeline, search_result_top_k: int) -> tuple[list, list]:
    golden_dataset = pd.read_csv("data/golden/golden_dataset.csv")
    # find all unique queries in golden dataset
    queries = golden_dataset["query"].unique()
    y_pred_all = []
    y_true_all = []
    recalls = []
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
                        # e.g. if two items are ranked as 3rd, they should get the average of points for 3rd and 4th place
                        num_of_same_rankings = len(results_true[results_true["ranking"] == true_row["ranking"]])
                        points_to_distribute = sum([true_row["ranking"] - i for i in range(num_of_same_rankings)]) / len(results_true)
                        y_true[i] = points_to_distribute / num_of_same_rankings
                        relevant_count += 1

            # because the amount of search results is different each time, pad it with 0
            y_pred.extend([0] * (search_result_top_k - k))
            y_true.extend([0] * (search_result_top_k - k))

            y_pred_all.append(y_pred)
            y_true_all.append(y_true)
            recall = relevant_count / len(results_true)
            recalls.append(recall)

            mlflow.log_param("query", query)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("number_of_results", len(results_pred))
            mlflow.log_metric("number_of_correct", len(results_true))
            for i in range(len(results_true)):
                mlflow.log_param(f"{i}. result", f"{results_pred[i]["title"]}, {results_pred[i]["url"]}")

    y_pred_all = np.array(y_pred_all)
    y_true_all = np.array(y_true_all)
    ndcg = ndcg_score(y_pred_all, y_true_all)

    logger.info(f"ndcg: {ndcg}, recall: {recalls}")
    return ndcg, recalls


if __name__ == "__main__":
    main()
