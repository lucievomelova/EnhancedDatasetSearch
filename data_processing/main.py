import asyncio
import os

import yaml
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from data_processing.keywords import create_keyword_kg, clustering, get_representatives
from data_processing.knowledge_graph import create_kg
from data_processing.nkod_datasets import NKOD
from data_processing.database import Database
import click



class DataPreprocessingPipeline:
    """Pipeline for preprocessing data and loading it into the RAG database."""
    def __init__(self, config: dict):
        self.rag_config = config["rag"]
        self.state_dir = config["state_dir"]
        self.dataset_portal = NKOD(config)
        self.super_df = self.dataset_portal.super_df
        self.llm = Ollama(model=self.rag_config['llm']['model_name'],
                          context_window=self.rag_config['llm']['context_length'])
        Settings.llm = self.llm
        Settings.embed_model = OllamaEmbedding(
            model_name=self.rag_config['embedding']['model_name'],
            base_url=self.rag_config['embedding']['base_url'],
            embed_batch_size=self.rag_config['embedding']['embed_batch_size'],
        )
        self.database = Database(self.rag_config, config["state_dir"])

    async def run(self) -> list[dict] | None:
        """Run the preprocessing pipeline."""

        if not os.path.exists(self.state_dir):
            os.makedirs(self.state_dir)

        # datasets_documents = await self.dataset_portal.get_new_datasets()
        # self.database.load_documents(datasets_documents)
        self.dataset_portal.init()
        create_kg(self.dataset_portal.extended_df, self.database)
        # create_keyword_kg(self.database, self.dataset_portal._all_keywords, self.rag_config["db"]["embed_dim"])
        # get_representatives()

@click.command()
@click.option('--config', default='config.yaml', help='Path to the configuration YAML file.')
def main(config: str):
    with open(config, "r") as f:
        config = yaml.safe_load(f)

    pipeline = DataPreprocessingPipeline(config)
    asyncio.run(pipeline.run())


if __name__ == "__main__":
    main()


# # TODO just for debugging
# with open("evaluation/config_eval.yaml", "r") as f:
#     config = yaml.safe_load(f)
#
# pipeline = DataPreprocessingPipeline(config)
# asyncio.run(pipeline.run())