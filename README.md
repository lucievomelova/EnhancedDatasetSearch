# Enhancing Dataset Discovery in Data Catalogs through Knowledge Graphs and Large Language Models
This repository contains application code for the master thesis 
"Enhancing Dataset Discovery in Data Catalogs through Knowledge Graphs and Large Language Models". 

It works with datasets from the Czech National Open Data Catalog (NKOD - Národní katalog otevřených dat).

## Structure
The system is structured into two main parts:
1. **Data processing** - responsible for loading, cleaning, transforming and enhancing the datasets' metadata,
as well as creating the knowledge base.
2. **Search platform** - a web application that provides a search engine over the datasets and a chatbot. It works with
data obtained by the Data processing step.

The application is written in Python. It uses `Flask` for the web interface, `llama_index` for handling LLM-based parts 
and `Neo4j` for knowledge graph handling. The knowledge base is stored in a `PostgreSQL` database.

# Implementation
## Data Processing
The data processing pipeline manages the preprocessing of data. First, the datasets' metadata are downloaded from
the [Czech Data Portal](data.gov.cz). After that the data is cleaned and transformed, so that we obtain a *dataset
of datasets*, where one row represents one dataset from the NKOD. 

Each dataset is represented by its metadata:
* **title** - dataset title
* **description** - text describing the dataset
* **url** - dataset url
* **keywords** - important keywords associated with the dataset
* **themes** - important themes that represent the dataset
* **categories** - categorization of the dataset into predefined categories
* **provider** - the institution that provides the dataset
* **spatial coverage** - associated geographical region
* **temporal coverage** - associated time period

### Data Processing steps
#### 1. Raw dataset loading and preprocessing
Loading raw datasets file from NKOD and performing initial preprocessing steps. The raw file contains one line per each 
metadata configuration. When a dataset has multiple keywords or themes assigned, each is handled on a separate line. 
This is quite impractical, so the initial preprocessing step includes merging keywords and themes for each dataset into 
a single list. After that, additional metadata (spatial and temporal coverage) are obtained through the NKOD SPARQL 
endpoint.

#### 2. LLM-based Metadata Enhancement
Using a Large Language Model (LLM) to enhance the metadata by generating additional metadata based on the 
already existing metadata, which can help improve the discoverability of datasets.

#### 3. Data Cleaning
Removing duplicates, handling missing values, ensuring that the data has a consistent format. This step also includes 
heuristical keyword and theme processing, where we try to find and fix typos by finding words that are used sporadically and 
are very similar to more frequently used words. 
This is done by calculating the Levenshtein distance between words and merging those that are similar.

#### 4. Knowledge base creation
Creating a knowledge base from the preprocessed data. The knowledge base includes the metadata dataset obtained 
in the previous steps, a Postgres database and a knowledge graph.

##### Postgres database
The Postgres database contains a document store and a vector store. We use
`PGVectorStore` and `PostgresDocumentStore` from llamaindex. 

Each document represents one dataset and it contains the dataset's metadata. 
The vector store contains embeddings of the datasets' descriptions, which are used for semantic search. 
The embeddings are generated using an embedding model.

##### Knowledge graph creation
Creating a knowledge graph based on dataset metadata to capture relationships between datasets. This knowledge 
graph is then used for retrieval of similar datasets.


### Running the Data Processing Pipeline
To run the data preprocessing pipeline, run the following command from the project directory (`EnhancedDatasetSearch/`)
```bash
python -m EnhancedDatasetSearch.data_preprocessing.main --config <path-to-config>
```

## Web Application
The web application provides a simple user interface, where users can search for datasets, apply metadata filters, 
check dataset detail page or chat with a chatbot. On the backend the main part is an LLM-based search pipeline.
This pipeline handles user query preprocessing, retrieval of relevant datasets and result postprocessing
and it is used both by the search engine and by the chatbot.

### User Interface
#### Search
The application offers a search interface that allows users to find relevant datasets based on their queries. 
Users can input a query in natural language, which will be processed by an LLM and relevant datasets will be returned.

#### Dataset detail
Users can also go to a dataset_detail page, where they can see detailed information about the selected dataset. 
This page shows datasets that are similar to the selected dataset. The similarity is based on semantic similarity of 
the datasets' descriptions or on common metadata.

#### Chatbot
There is also a chatbot, which offers a conversational approach to searching in the NKOD. 
The chatbot uses the same search pipeline for finding relevant datasets, but it has other tools as well.

### Search Pipeline
The search pipeline consists of the following steps:
1. **Query preprocessing:** the user query is processed by an LLM to extract user intent and it is extended for better retrieval quality
2. **Retrieval:** the extended query is sued for the hybrid search.
3. **Result postprocessing:** the retrieved datasets are ranked based on their relevance to the user query, some
irrelevant datasets can be dropped from the search results and the remaining are sorted by their relevance.

#### Retrieval
The retrieval is based on hybrid retrieval strategy, which combines keyword-based search with semantic search. 
The keyword-based retrieval uses BM25 algorithm to find datasets that match the keywords in the user query. 
The semantic retrieval is based on embedding similarity - the user query is embedded and then the most similar datasets
are retrieved based on cosine similarity of their description embedding vectors.
