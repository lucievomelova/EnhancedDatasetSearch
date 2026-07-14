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

from EnhancedNkodDatasetSearch.data_processing.NKOD.nkod_knowledge_graph import NkodKnowledgeGraph
from EnhancedNkodDatasetSearch.search_platform.data_catalog import DataCatalog
from EnhancedNkodDatasetSearch.data_processing.database import Database
from EnhancedNkodDatasetSearch.search_platform.nkod_data_catalog import NkodDataCatalog
from EnhancedNkodDatasetSearch.search_platform.search_pipeline.pipeline import SearchPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

mlflow.set_experiment("SearchPipeline Evaluation - grid search final new2")
golden_dataset = pd.read_csv("data/golden/golden_dataset.csv")


# custom Search pipeline so that intent and extended query are not regenerated on each run to save time
class EvaluationSearchPipeline():
    def __init__(self, config: dict, llm: Ollama, data_catalog: DataCatalog, database: Database):
        self.search_pipeline = SearchPipeline(config, llm, data_catalog, database)
        self.extended_queries: dict[str, list[str]]
        self.config = config
        self.create_extended_queries()

    def create_extended_queries(self) -> None:
        model_name = self.config["search_platform"]["llm"]["model_name"].replace(".", "_").replace(":", "_")
        intent = self.config["search_platform"]["preprocessing"]["detect_intent"]
        file = f"evaluation/extended_queries_{model_name}_intent_{str(intent)}.json"
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                self.extended_queries = json.load(f)
                return
        self.extended_queries = {}
        queries = golden_dataset["query"].unique()
        for query in queries:
            intent, extended_query = self.search_pipeline.query_preprocessor.run(
                query, None,
                self.config["data_processing"]["categories"],
                self.config["data_processing"]["other_category"]
            )
            self.extended_queries[query] = extended_query
        with open(file, "w", encoding="utf-8") as f:
            json.dump(self.extended_queries, f, ensure_ascii=False, indent=4)

    async def run(self, query: str) -> list[dict[str, str | list | None]] | None:
        if self.config["search_platform"]["preprocessing"]["extend_query"]:
            extended_query = self.extended_queries[query]
        else:
            extended_query = ""

        search_results = await self.search_pipeline.retriever.run(query, extended_query, None)
        nodes = self.search_pipeline.postprocessor.run(query, search_results)
        if nodes is not None:
            return nodes
        return None


@click.command()
@click.option('--config_path', default='config_grid_search.yaml', help='Path to the configuration YAML file.')
def main(config_path: str):
    """Setup and run the evaluation."""
    asyncio.run(run_grid_search(config_path))


async def run_grid_search(config_path: str):
    with open(config_path, "r") as f:
        grid_search_config = yaml.safe_load(f)

    database = Database(grid_search_config)
    knowledge_graph = NkodKnowledgeGraph(grid_search_config["data_processing"]["knowledge_graph"], database)
    data_catalog = NkodDataCatalog(grid_search_config, knowledge_graph)
    config = copy.deepcopy(grid_search_config)
    i = 1
    pipeline_config_gs = grid_search_config["search_platform"]
    pipeline_config = config["search_platform"]
    for model_name in pipeline_config_gs["llm"]["model_name"]:
        pipeline_config["llm"]["model_name"] = model_name
        for intent in pipeline_config_gs["preprocessing"]["detect_intent"]:
            pipeline_config["preprocessing"]["detect_intent"] = intent
            for extend_query in pipeline_config_gs["preprocessing"]["extend_query"]:
                pipeline_config["preprocessing"]["extend_query"] = extend_query
                for top_k in pipeline_config_gs["retriever"]["top_k"]:
                    pipeline_config["retriever"]["top_k"] = top_k
                    for vector_top_k in pipeline_config_gs["retriever"]["vector_top_k"]:
                        pipeline_config["retriever"]["vector_top_k"] = vector_top_k
                        for bm25_top_k in pipeline_config_gs["retriever"]["bm25_top_k"]:
                            pipeline_config["retriever"]["bm25_top_k"] = bm25_top_k
                            for mode in pipeline_config_gs["retriever"]["mode"]:
                                pipeline_config["retriever"]["mode"] = mode
                                for retriever_weights in pipeline_config_gs["retriever"]["retriever_weights"]:
                                    pipeline_config["retriever"]["retriever_weights"] = retriever_weights
                                    for reranker in pipeline_config_gs["postprocessing"]["reranker"]:
                                        pipeline_config["postprocessing"]["reranker"] = reranker
                                        for score_cutoff in pipeline_config_gs["postprocessing"]["score_cutoff"]:
                                            pipeline_config["postprocessing"]["score_cutoff"] = score_cutoff
                                            for reranker in pipeline_config_gs["postprocessing"]["reranker"]:
                                                pipeline_config["postprocessing"]["reranker"] = reranker
                                                logger.info(f"{i}. Configuration: {pipeline_config}")
                                                await run_experiment(config, data_catalog, database)


async def run_experiment(config: dict, data_catalog: NkodDataCatalog, database: Database) -> None:
    llm = Ollama(model=config['search_platform']['llm']['model_name'],
                 context_window=config['search_platform']['llm']['context_length'],
                 request_timeout=300)

    search_pipeline = EvaluationSearchPipeline(config, llm, data_catalog, database)

    with mlflow.start_run():
        for key in config["search_platform"]:
            params = {}
            for sub_key in config["search_platform"][key]:
                params[f"{key}_{sub_key}"] = config["search_platform"][key][sub_key]
            mlflow.log_params(params)
        ndcg, recall, mrr = await evaluate(search_pipeline)

        mlflow.log_metric("NDCG", ndcg)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("MRR", mrr)
        mlflow.log_param("Embedding model", config["embedding"]["model_name"])
        mlflow.log_param("LLM", config["search_platform"]["llm"]["model_name"])


async def evaluate(search_pipeline: EvaluationSearchPipeline) -> tuple[float, float, float]:
    golden_dataset = pd.read_csv("data/golden/golden_dataset.csv")
    queries = golden_dataset["query"].unique()  # get all unique queries from golden dataset
    recalls, ndcgs, reciprocal_ranks = [], [], []
    total_relevant, total_retrieved_relevant = 0, 0
    top_k = 10

    for query in queries:
        with mlflow.start_run(nested=True) as child_run:
            logger.info(f"Evaluating query: {query}")
            results_pred = await search_pipeline.run(query)
            results_true = golden_dataset[golden_dataset["query"] == query][["url", "ranking"]]

            max_rank = results_true["ranking"].max()
            relevance = {row["url"]: max_rank - row["ranking"] + 1 for _, row in results_true.iterrows()}
            y_true, y_pred = [], []

            relevant_count = 0
            reciprocal_rank = 0
            for i, pred_row in enumerate(results_pred):
                rel = relevance.get(pred_row["url"], 0)
                y_true.append(rel)
                if rel > 0 and i < top_k:  # recall@k and MRR@k, so we only look at first k results
                    relevant_count += 1
                    if reciprocal_rank == 0:
                        reciprocal_rank = 1.0 / (i + 1)
                y_pred.append(len(results_pred) - i)

            reciprocal_ranks.append(reciprocal_rank)
            recall = relevant_count / len(results_true)
            recalls.append(recall)
            ndcg = ndcg_score(np.array([y_true]), np.array([y_pred]), k=top_k)
            ndcgs.append(ndcg)
            total_relevant += len(results_true)
            total_retrieved_relevant += relevant_count

            mlflow.log_param("query", query)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("NDCG", ndcg)
            mlflow.log_metric("reciprocal_rank", reciprocal_rank)
            mlflow.log_metric("number_of_results", len(results_pred))
            for i in range(len(results_pred)):
                mlflow.log_param(f"{i}. result", f"{results_pred[i]["title"]}, {results_pred[i]["url"]}")

    ndcg = sum(ndcgs) / len(ndcgs)
    recall = sum(recalls) / len(recalls)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    logger.info(f"ndcg: {ndcg}, recall: {recall}, mrr: {mrr}")
    return ndcg, recall, mrr


if __name__ == "__main__":
    main()
