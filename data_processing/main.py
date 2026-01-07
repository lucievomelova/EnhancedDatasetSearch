import asyncio

import yaml
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

from custom_ollama_embedding import CustomOllamaEmbedding
from data_processing.knowledge_graph import create_kg
from data_processing.nkod_datasets import NKOD
from data_processing.database import Database
import click
class DataPreprocessingPipeline:
    """Pipeline for preprocessing data and loading it into the RAG database."""
    def __init__(self, config: dict):
        self.rag_config = config["rag"]
        self.dataset_portal = NKOD(config)
        self.super_df = self.dataset_portal.super_df
        self.llm = Ollama(model=self.rag_config['llm']['model_name'],
                          context_window=self.rag_config['llm']['context_length'])
        Settings.llm = self.llm
        Settings.embed_model = CustomOllamaEmbedding(
            model_name=self.rag_config['embedding']['model_name'],
            base_url=self.rag_config['embedding']['base_url'],
            embed_batch_size=self.rag_config['embedding']['embed_batch_size'],
        )
        self.database = Database(self.rag_config)

    async def run(self) -> list[dict] | None:
        """Run the preprocessing pipeline."""

        datasets_documents = await self.dataset_portal.get_new_datasets()
        self.database.load_documents(datasets_documents)
        create_kg(self.dataset_portal.extended_df)


@click.command()
@click.option('--config', default='config.yaml', help='Path to the configuration YAML file.')
def main(config: str):
    with open(config, "r") as f:
        config = yaml.safe_load(f)

    pipeline = DataPreprocessingPipeline(config)
    asyncio.run(pipeline.run())


if __name__ == "__main__":
    main()