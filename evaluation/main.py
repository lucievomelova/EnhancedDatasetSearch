import asyncio
import logging

import click
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import ndcg_score
from app.pipeline import SearchPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option('--config', default='config.yaml', help='Path to the configuration YAML file.')
def main(config: str):
    with open(config, "r") as f:
        config = yaml.safe_load(f)
    search_pipeline = SearchPipeline(config)
    asyncio.run(evaluate(search_pipeline))

async def evaluate(search_pipeline: SearchPipeline):
    golden_dataset = pd.read_csv("data/golden/golden_dataset.csv")
    # find all unique queries in golden dataset
    queries = golden_dataset["query"].unique()
    y_pred_all = []
    y_true_all = []
    recalls = []
    for query in queries:
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

        y_pred_all.append(y_pred)
        y_true_all.append(y_true)
        recall = relevant_count / len(results_true)
        recalls.append(recall)

    y_pred_all = np.array(y_pred_all)
    y_true_all = np.array(y_true_all)
    ndcg = ndcg_score(y_pred_all, y_true_all)

    logger.info(f"ndcg: {ndcg}, recall: {recalls}")


if __name__ == "__main__":
    main()