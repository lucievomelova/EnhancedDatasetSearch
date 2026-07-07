"""Run data processing pipeline. The pipeline has the following steps:
    1. Dataset processing pipeline: update metadata dataset and get updated and removed datasets info
    2. Documents creation: create llama_index documents from the metadata dataset
    3. Load to DB: load all the documents to the knowledge base
    4. Update knowledge graph: create or update the knowledge graph based on the updated metadata dataset
"""

import asyncio
import os

import click
import yaml
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from EnhancedDatasetSearch.data_processing.dataset_processing_pipeline import DatasetProcessingPipeline
from EnhancedDatasetSearch.data_processing.documents import DocumentConverter
from EnhancedDatasetSearch.data_processing.NKOD.nkod_documents import NkodDocumentConverter
from EnhancedDatasetSearch.database import Database
from EnhancedDatasetSearch.knowledge_graph import KnowledgeGraph
from EnhancedDatasetSearch.NKOD.knowledge_graph import NkodKnowledgeGraph
from EnhancedDatasetSearch.data_processing.NKOD.nkod_dataset_processing_pipeline import NkodDatasetProcessingPipeline


@click.command()
@click.option('--config_path', default='config.yaml', help='Path to the configuration YAML file.')
def main(config_path: str):
    """Run data processing pipeline.

    Run dataset processing pipeline, create llama_index documents from the processed metadata dataset, load the documents
    to the knowledge base and create or update the knowldge graph.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    state_dir = config["state_dir"]
    database = Database(config)
    dataset_processing_pipeline: DatasetProcessingPipeline = NkodDatasetProcessingPipeline(config)
    document_converter: DocumentConverter = NkodDocumentConverter()

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

    if not os.path.exists(state_dir):  # create state dir
        os.makedirs(state_dir)

    # # Run data processing:
    # # 1. update metadata dataset and get updated and removed datasets info
    new_datasets, removed_urls = asyncio.run(dataset_processing_pipeline.update_datasets())
    #
    # # 2. create llama_index documents from the metadata dataset
    datasets_documents = document_converter.create_documents(dataset_processing_pipeline.datasets)
    #
    # # 3. load the documents to the knowledge base
    database.load_documents(datasets_documents)
    #
    # # 4. create or update the knowledge graph
    # knowledge_graph.create_or_update_kg(dataset_processing_pipeline.datasets, new_datasets, removed_urls)
    knowledge_graph.create_or_update_kg(dataset_processing_pipeline.datasets, dataset_processing_pipeline.datasets, [])


if __name__ == "__main__":
    main()
