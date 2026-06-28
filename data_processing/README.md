# Data Preprocessing Pipeline
This pipeline manages the preprocessing of data - cleaning, transforming, etc. It also uses a LLM to enhance data quality.

## Running the Pipeline
To run the data preprocessing pipeline, run the following command from `app` directory:
```bash
python -m data_preprocessing_pipeline --config <path-to-config>
```

## Documentation
The preprocessing pipeline handles data preprocessing, which includes:
#### 1. Raw dataset loading and preprocessing
Loading raw datasets file frm NKOD and performing initial preprocessing steps. The raw file contains one line per each 
metadata configuration. When a dataset has multiple keywords or themes assigned, each is handled on a separate line. 
This is quite impractical, so the initial preprocessing step includes merging kewyrods and themes for each dataset into 
a single list.

#### Data Cleaning
Removing duplicates, handling missing values, ensuring that the data has a consistent format. This step also includes 
heuristical keyword and theme processing, where we try to find and fix typos by finding words that are used sporadically and 
are very similar to more frequently used words. 
This is done by calculating the Levenshtein distance between words and merging those that are similar.

#### LLM-based Metadata Enhancement
Using a Large Language Model (LLM) to enhance the metadata by generating additional keywords and themes based on the 
already existing metadata, which can help improve the discoverability of datasets.

#### Knowledge base creation
Creating a knowledge base from the preprocessed data - a document store and a vector store. We use postgres, 
specifically PGVectorStore and PostgresDocumentStore from llamaindex. The knwoldge base stores information about each
dataset, including its metadata and a description of its content.
This allows for efficient retrieval of relevant datasets based on user queries.

#### Knowledge graph creation
Creating a knowledge graph based on dataset metadata, to capture relationships between datasets. This knowledge 
graph is then used for similar dataset retrieval.
