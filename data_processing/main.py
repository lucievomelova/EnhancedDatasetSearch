import asyncio
import os

import yaml
from data_processing.data_catalog import DataCatalog
from data_processing.knowledge_graph import KnowledgeGraph
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from data_processing.NKOD.knowledge_graph import NkodKnowledgeGraph
from data_processing.NKOD.nkod import NkodDataCatalog
from data_processing.database import Database
import click


class DataPreprocessingPipeline:
    """Pipeline for preprocessing data and loading it into the database."""
    def __init__(self, config: dict):
        self.config = config
        self.state_dir = config["state_dir"]
        self.database = Database(self.config)
        self.data_catalog: DataCatalog = NkodDataCatalog(config, True)
        self.knowledge_graph: KnowledgeGraph = NkodKnowledgeGraph(
            self.config["data_processing"]["knowledge_graph"],
            self.database
        )
        self.llm = Ollama(model=self.config['llm']['model_name'],
                          context_window=self.config['llm']['context_length'])
        Settings.llm = self.llm
        Settings.embed_model = OllamaEmbedding(
            model_name=self.config['embedding']['model_name'],
            base_url=self.config['embedding']['base_url'],
            embed_batch_size=self.config['embedding']['embed_batch_size'],
        )

    def run(self) -> list[dict] | None:
        """Run the preprocessing pipeline."""

        if not os.path.exists(self.state_dir):
            os.makedirs(self.state_dir)

        asyncio.run(self.data_catalog.update_datasets())
        datasets_documents = self.data_catalog.prepare_documents_for_upload(self.data_catalog.datasets)
        self.database.load_documents(datasets_documents)
        self.knowledge_graph.create_kg(self.data_catalog.datasets)


@click.command()
@click.option('--config', default='config.yaml', help='Path to the configuration YAML file.')
def main(config: str):
    with open(config, "r") as f:
        config = yaml.safe_load(f)

    pipeline = DataPreprocessingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()


# # TODO just for debugging
# with open("evaluation/config_eval.yaml", "r") as f:
#     config = yaml.safe_load(f)
#
# pipeline = DataPreprocessingPipeline(config)
# asyncio.run(pipeline.run())