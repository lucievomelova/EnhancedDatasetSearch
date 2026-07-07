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

from llama_index.llms.ollama import Ollama
from sklearn.metrics import ndcg_score

from EnhancedDatasetSearch.NKOD.knowledge_graph import NkodKnowledgeGraph
from EnhancedDatasetSearch.data_catalog import DataCatalog
from EnhancedDatasetSearch.database import Database
from EnhancedDatasetSearch.NKOD.nkod import NkodDataCatalog
from EnhancedDatasetSearch.search.pipeline import SearchPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

mlflow.set_experiment("SearchPipeline Evaluation - grid search")
golden_dataset = pd.read_csv("data/golden/golden_dataset.csv")


# custom Search pipeline so that intent and extended query are not regenerated on each run to save time
class EvaluationSearchPipeline():
    def __init__(self, config: dict, llm: Ollama, data_catalog: DataCatalog, database: Database, k: int):
        self.search_pipeline = SearchPipeline(config, llm, data_catalog, database)
        self.extended_queries: dict[str, list[str]]
        self.config = config
        self.create_extended_queries(k)

    def create_extended_queries(self, k: int = 3) -> None:
        model_name = self.config["pipeline_config"]["llm"]["model_name"].replace(".", "_").replace(":", "_")
        intent = self.config["pipeline_config"]["preprocessing"]["detect_intent"]
        file = f"evaluation/extended_queries_{model_name}_intent_{str(intent)}.json"
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                self.extended_queries = json.load(f)
                return
        self.extended_queries = {}
        queries = golden_dataset["query"].unique()
        for query in queries:
            for i in range(k):
                intent, extended_query = self.search_pipeline.query_preprocessor.run(
                    query, None,
                    self.config["data_processing"]["categories"],
                    self.config["data_processing"]["other_category"]
                )
                if query not in self.extended_queries:
                    self.extended_queries[query] = [extended_query]
                else:
                    self.extended_queries[query].append(extended_query)
        with open(file, "w", encoding="utf-8") as f:
            json.dump(self.extended_queries, f, ensure_ascii=False, indent=4)

    async def run(self, query: str, index: int) -> list[dict[str, str | list | None]] | None:
        if self.config["pipeline_config"]["preprocessing"]["extend_query"]:
            extended_query = self.extended_queries[query][index]
        else:
            extended_query = ""

        search_results = await self.search_pipeline.retriever.run(query, extended_query)
        nodes = self.search_pipeline.postprocessor.run(query, search_results)
        if nodes is not None:
            return nodes
        return None


@click.command()
@click.option('--config_path', default='config_grid_search.yaml', help='Path to the configuration YAML file.')
def main(config_path: str):
    """Setup and run the evaluation."""
    with open(config_path, "r") as f:
        grid_search_config = yaml.safe_load(f)

    database = Database(grid_search_config)
    knowledge_graph = NkodKnowledgeGraph(grid_search_config["data_processing"]["knowledge_graph"], database)
    data_catalog = NkodDataCatalog(grid_search_config, knowledge_graph)
    config = copy.deepcopy(grid_search_config)
    i = 1
    pipeline_config_gs = grid_search_config["pipeline_config"]
    pipeline_config = config["pipeline_config"]
    for model_name in pipeline_config_gs["llm"]["model_name"]:
        pipeline_config["llm"]["model_name"] = model_name
        for intent in pipeline_config_gs["preprocessing"]["detect_intent"]:
            pipeline_config["preprocessing"]["detect_intent"] = intent
            for extend_query in pipeline_config_gs["preprocessing"]["extend_query"]:
                pipeline_config["preprocessing"]["extend_query"] = extend_query
                for top_k in pipeline_config_gs["search"]["top_k"]:
                    pipeline_config["search"]["top_k"] = top_k
                    for vector_top_k in pipeline_config_gs["search"]["vector_top_k"]:
                        pipeline_config["search"]["vector_top_k"] = vector_top_k
                        for bm25_top_k in pipeline_config_gs["search"]["bm25_top_k"]:
                            pipeline_config["search"]["bm25_top_k"] = bm25_top_k
                            for mode in pipeline_config_gs["search"]["mode"]:
                                pipeline_config["search"]["mode"] = mode
                                for retriever_weights in pipeline_config_gs["search"]["retriever_weights"]:
                                    pipeline_config["search"]["retriever_weights"] = retriever_weights
                                    for reranker in pipeline_config_gs["postprocessing"]["reranker"]:
                                        pipeline_config["postprocessing"]["reranker"] = reranker
                                        for score_cutoff in pipeline_config_gs["postprocessing"]["score_cutoff"]:
                                            pipeline_config["postprocessing"]["score_cutoff"] = score_cutoff
                                            for reranker in pipeline_config_gs["postprocessing"]["reranker"]:
                                                pipeline_config["postprocessing"]["reranker"] = reranker
                                                logger.info(f"{i}. Configuration: {pipeline_config}")
                                                run_experiment(config, data_catalog, database)


def run_experiment(config: dict, data_catalog: NkodDataCatalog, database: Database) -> None:
    llm = Ollama(model=config['pipeline_config']['llm']['model_name'],
                 context_window=config['pipeline_config']['llm']['context_length'],
                 request_timeout=300)

    queries_k = 1
    search_pipeline = EvaluationSearchPipeline(config, llm, data_catalog, database, queries_k)

    with mlflow.start_run():
        for key in config["pipeline_config"]:
            params = {}
            for sub_key in config["pipeline_config"][key]:
                params[f"{key}_{sub_key}"] = config["pipeline_config"][key][sub_key]
            mlflow.log_params(params)
        ndcg, recall_avg, recall = asyncio.run(evaluate(search_pipeline, queries_k))

        mlflow.log_metric("ndcg", ndcg)
        mlflow.log_metric("recall_avg", recall_avg)
        mlflow.log_metric("recall", recall)
        mlflow.log_param("Embedding model", config["embedding"]["model_name"])
        mlflow.log_param("LLM", config["pipeline_config"]["llm"]["model_name"])


async def evaluate(search_pipeline: EvaluationSearchPipeline, queries_k: int) -> tuple[float, float, float]:
    golden_dataset = pd.read_csv("data/golden/golden_dataset.csv")
    queries = golden_dataset["query"].unique()  # get all unique queries from golden dataset
    recalls, ndcgs = [], []
    total_relevant, total_retrieved_relevant = 0, 0

    for query in queries:
        with mlflow.start_run(nested=True) as child_run:
            recalls_iteration = []
            ndcgs_iteration = []
            iterations = queries_k
            for i in range(iterations):
                logger.info(f"Evaluating query: {query}")
                results_pred = await search_pipeline.run(query, i)
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
                recalls_iteration.append(recall)
                ndcg = ndcg_score(np.array([y_true]), np.array([y_pred]))
                ndcgs_iteration.append(ndcg)
                total_relevant += len(results_true)
                total_retrieved_relevant += relevant_count

            recall = sum(recalls_iteration) / iterations
            ndcg = sum(ndcgs_iteration) / iterations
            recalls.append(recall)
            ndcgs.append(ndcg)
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
