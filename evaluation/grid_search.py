"""Evaluation script for evaluating the search pipeline.
The evaluation is based on a golden dataset of example queries and expected search results."""

import asyncio
import copy
import json
import logging

import click
import mlflow
import numpy as np
import pandas as pd
import yaml
import os

from EnhancedDatasetSearch.app.query_prepocessing import QueryPreprocessor
from llama_index.llms.ollama import Ollama
from sklearn.metrics import ndcg_score

from EnhancedDatasetSearch.data_processing.database import Database
from EnhancedDatasetSearch.data_processing.NKOD.nkod import NkodDataCatalog
from EnhancedDatasetSearch.app.pipeline import SearchPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


mlflow.set_experiment("SearchPipeline Evaluation")
golden_dataset = pd.read_csv("data/golden/golden_dataset.csv")


def create_extended_queries(config) -> dict:
    extended_queries = {}
    preprocessor = QueryPreprocessor(config)
    queries = golden_dataset["query"].unique()
    for query in queries:
        intent, extended_query = preprocessor.run(query, None, config["data_processing"]["categories"], config["data_processing"]["other_category"])
        extended_queries[query] = extended_query
    return extended_queries

extended_queries = {}

# override Search pipeline so taht intent and extended query are not regenerated on each run to save time
class EvaluationSearchPipeline(SearchPipeline):
    async def run(self, query: str, applied_filters: dict | None = None) -> list[dict[str, str | list | None]] | None:
        extended_query = extended_queries[query]

        search_results = await self.retriever.run(query, extended_query)
        nodes = self.postprocessor.run(query, extended_query, search_results, {})
        if nodes is not None:
            return nodes
        return None


@click.command()
@click.option('--config_path', default='config_grid_search.yaml', help='Path to the configuration YAML file.')
def main(config_path: str):
    """Setup and run the evaluation."""
    with open(config_path, "r") as f:
        grid_search_config = yaml.safe_load(f)

    global extended_queries
    file = "evaluation/extended_queries_qwen_36_27b.json"
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            extended_queries = json.load(f)
    else:
        extended_queries = create_extended_queries(grid_search_config)
        with open(file, "w", encoding="utf-8") as f:
            json.dump(extended_queries, f, ensure_ascii=False, indent=4)

    data_catalog = NkodDataCatalog(grid_search_config)
    config = copy.deepcopy(grid_search_config)
    i = 1
    for top_k in grid_search_config["pipeline_config"]["search"]["top_k"]:
        config["pipeline_config"]["search"]["top_k"] = top_k
        for vector_top_k in grid_search_config["pipeline_config"]["search"]["vector_top_k"]:
            config["pipeline_config"]["search"]["vector_top_k"] = vector_top_k
            for bm25_top_k in grid_search_config["pipeline_config"]["search"]["bm25_top_k"]:
                config["pipeline_config"]["search"]["bm25_top_k"] = bm25_top_k
                for mode in grid_search_config["pipeline_config"]["search"]["mode"]:
                    config["pipeline_config"]["search"]["mode"] = mode
                    for retriever_weights in grid_search_config["pipeline_config"]["search"]["retriever_weights"]:
                        config["pipeline_config"]["search"]["retriever_weights"] = retriever_weights
                        for reranker in grid_search_config["pipeline_config"]["postprocessing"]["reranker"]:
                            config["pipeline_config"]["postprocessing"]["reranker"] = reranker
                            for score_cutoff in grid_search_config["pipeline_config"]["postprocessing"]["score_cutoff"]:
                                config["pipeline_config"]["postprocessing"]["score_cutoff"] = score_cutoff
                                logger.info(f"{i}. Configuration: {config["pipeline_config"]}")
                                run_experiment(config, data_catalog)


def run_experiment(config: dict, data_catalog: NkodDataCatalog) -> None:
    llm = Ollama(model=config['pipeline_config']['llm']['model_name'],
                 context_window=config['pipeline_config']['llm']['context_length'],
                 request_timeout=300)

    database = Database(config)
    search_pipeline = EvaluationSearchPipeline(config, llm, data_catalog, database)

    search_result_top_k = config["pipeline_config"]["search"]["top_k"]
    with mlflow.start_run():
        for key in config["pipeline_config"]:
            params = {}
            for sub_key in config["pipeline_config"][key]:
                params[f"{key}_{sub_key}"] = config["pipeline_config"][key][sub_key]
            mlflow.log_params(params)
        ndcg, recalls = asyncio.run(evaluate(search_pipeline, search_result_top_k))

        mlflow.log_metric("ndcg", ndcg)
        mlflow.log_metric("recall", sum(recalls)/len(recalls))
        mlflow.log_param("Embedding model", config["embedding"]["model_name"])
        mlflow.log_param("LLM", config["pipeline_config"]["llm"]["model_name"])


async def evaluate(search_pipeline: EvaluationSearchPipeline, search_result_top_k: int) -> tuple[list, list]:
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
                        y_true[i] = 1 / true_row["ranking"]
                        # more items can have the same ranking - distribute points among them fairly
                        # e.g. if two items are ranked as 3rd, they should get the average of points for 3rd and 4th place
                        # num_of_same_rankings = len(results_true[results_true["ranking"] == true_row["ranking"]])
                        # points_to_distribute = sum([true_row["ranking"] - i for i in range(num_of_same_rankings)]) / len(results_true)
                        # y_true[i] = points_to_distribute / num_of_same_rankings
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
            # returned_urls = [r["url"] for r in results_pred]

            # for i, row in results_true.iterrows():
            #     match = data_catalog.datasets.loc[data_catalog.datasets["url"] == row["url"], "title"]
            #     title = match.iloc[0] if not match.empty else "-"
            #     if row["url"] in returned_urls:
            #         mlflow.log_param(f"{i}. expected result - {title}, {row["url"]}", True)
            #     else:
            #         mlflow.log_param(f"{i}. expected result - {title}, {row["url"]}", False)

    y_pred_all = np.array(y_pred_all)
    y_true_all = np.array(y_true_all)
    ndcg = ndcg_score(y_true_all, y_pred_all)
    logger.info(f"ndcg_score: {ndcg}")
    ndcg = sum(ndcgs) / len(ndcgs)
    logger.info(f"ndcg: {ndcg}, recall: {recalls}")
    return ndcg, recalls


if __name__ == "__main__":
    main()
