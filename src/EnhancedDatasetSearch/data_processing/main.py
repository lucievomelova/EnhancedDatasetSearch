import asyncio
import os

import click
import yaml
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from EnhancedDatasetSearch.data_processing.data_processing_pipeline import DataProcessingPipeline
from EnhancedDatasetSearch.database import Database
from EnhancedDatasetSearch.knowledge_graph import KnowledgeGraph
from EnhancedDatasetSearch.NKOD.knowledge_graph import NkodKnowledgeGraph
from EnhancedDatasetSearch.data_processing.NKOD.nkod_data_processing_pipeline import NkodDataProcessingPipeline


@click.command()
@click.option('--config_path', default='config.yaml', help='Path to the configuration YAML file.')
def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    state_dir = config["state_dir"]
    database = Database(config)
    data_processing_pipeline: DataProcessingPipeline = NkodDataProcessingPipeline(config)
    knowledge_graph: KnowledgeGraph = NkodKnowledgeGraph(
        config["data_processing"]["knowledge_graph"],
        database
    )
    llm = Ollama(model=config["data_processing"]["llm"]["model_name"],
                      context_window=config["data_processing"]["llm"]["context_length"])
    Settings.llm = llm
    Settings.embed_model = OllamaEmbedding(
        model_name=config['embedding']['model_name'],
        base_url=config['embedding']['base_url'],
        embed_batch_size=config['embedding']['embed_batch_size'],
    )

    # run data processing
    if not os.path.exists(state_dir):
        os.makedirs(state_dir)

    new_datasets, removed_urls = asyncio.run(data_processing_pipeline.update_datasets())
    datasets_documents = data_processing_pipeline.prepare_documents_for_upload(data_processing_pipeline.datasets)
    database.load_documents(datasets_documents)
    # knowledge_graph.create_or_update_kg(data_processing_pipeline.datasets, new_datasets, removed_urls)
    knowledge_graph.create_or_update_kg(data_processing_pipeline.datasets, data_processing_pipeline.datasets, [])


if __name__ == "__main__":
    main()


# # TODO just for debugging
# with open("evaluation/config_eval.yaml", "r") as f:
#     config = yaml.safe_load(f)
#
# pipeline = DataPreprocessingPipeline(config)
# asyncio.run(pipeline.run())