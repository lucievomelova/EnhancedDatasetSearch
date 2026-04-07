# Enhancing Dataset Discovery in Data Catalogs through Knowledge Graphs and Large Language Models
This is repository contains application code for the master thesis 
"Enhancing Dataset Discovery in Data Catalogs through Knowledge Graphs and Large Language Models". 

The application is built using Python and is structured into two main parts:
1. **Data preprocessing pipeline** - responsible for loading, cleaning, and enhancing the dataset metadata, as well as creating a knowledge base and a knowledge graph.
2. **Dataset retrieval pipeline** - responsible for retrieving relevant datasets based on user queries, using the knowledge base and the knowledge graph created in the preprocessing step.

The application is written in Python. It uses flask for the web interface, llamaindex for handling LLM-based parts 
and neo4j for knowledge graph handling. 

For more details on the implementation of each part, 
please refer to the [data preprocessing pipeline documentation](data_processing/README.md) 
and the [dataset retrieval pipeline documentation](dataset_retrieval/README.md).