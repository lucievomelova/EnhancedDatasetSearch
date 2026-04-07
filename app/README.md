# NKOD dataset search
This application allows users to search through the datasets offered at the Czech national open data catalog
(Národní katalog otevřených dat - NKOD), which contains datasets from various public institutions in the Czech Republic.

## Application
### Search Interface
The application offers a search interface that allows users to find relevant datasets based on their queries. 
Users can input a query in natural language, which will be processed by an LLM and relevant datasets will be returned.

### Dataset detail
Users can also go to a dataset_detail page, where they can see detailed information about the selected dataset. 
This page offers datasets that are similar to the selected dataset. The similarity is based on semantic similairity of 
the dataset description or it is based on common metadata (keywords or themes).


## Search Functionality
The search is based on hybrid retrieval strategy, which combines keyword-based search with semantic search. 
The keyword-based search uses BM25 algorithm to find datasets that match the keywords in the user query. 
The semantic search is based on embedding similarity - the user query is embedded and then the most similar datasets
are retrieved based on cosine similarity of their description embeddings.

### Search pipeline
The search pipeline consists of the following steps:
1. **Query preprocessing:** the user query is processed by an LLM to extract user intent and it is extended for better retrieval quality
2. **Search:** the extended query is sued for the hybrid search.
3. **Result processing:** the retrieved datasets are ranked based on their relevance to the user query.

## Chatbot
The application also includes a chatbot interface, where users can ask for datasets in a conversational manner. 
The chatbot uses the same search pipeline as the search interface.