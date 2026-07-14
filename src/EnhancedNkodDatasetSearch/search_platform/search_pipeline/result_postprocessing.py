from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.schema import NodeWithScore

from EnhancedNkodDatasetSearch.search_platform.data_catalog import DataCatalog
from EnhancedNkodDatasetSearch.utils import setup_logger

logger = setup_logger(__name__)

class PostProcessor:
    """Search result postprocessor."""

    def __init__(self, postprocessing_config: dict, data_catalog: DataCatalog) -> None:
        self.postprocessing_config: dict = postprocessing_config
        self.data_catalog: DataCatalog = data_catalog
        self.top_k: int = self.postprocessing_config["top_k"]

        # initialize sentence transformer if it is set as reranker
        if self.postprocessing_config["reranker"] == "cross_encoder":
            self.cross_encoder = SentenceTransformerRerank(
                model=self.postprocessing_config["cross_encoder_model"], top_n=self.top_k
            )

    def run(self, user_query: str, results: list[NodeWithScore]) -> list[dict]:
        """Post-process search results.

        Search results are reranked and then top k results are returned.
        """
        if not results:
            logger.info("No results found.")
            return []

        logger.info(f"Post-processing {len(results)} results.")
        results = self.rerank(user_query, results)
        if len(results) > self.top_k:
            results = results[:self.top_k]  # keep only top_k results if there are more
        results_with_info = [self.data_catalog.get_dataset_by_url(str(res.metadata["url"])) for res in results]

        # filter out None results
        results_with_info = [result for result in results_with_info if result is not None]
        return results_with_info

    def rerank(self, user_query: str, results: list[NodeWithScore]) -> list[NodeWithScore]:
        """Rerank search results.

        Search results with low retrieval score are deleted, the remaining results are reranked using the reranker
        specified in the config and this reranked list is returned.
        The available rerankers are:
        * simple - rerank results using their retrieval score.
        * sentence transformer - use a sentence transformer for reranking. The model can be specified in the config.
        """
        # cut off nodes with low score
        cutoff = self.postprocessing_config["score_cutoff"]
        results = [result for result in results if result.score >= cutoff]
        logger.info(f"{len(results)} results after cutting off low similarity scores (cutoff: {cutoff}).")

        logger.info(f"Reranking remaining results.")
        reranker = self.postprocessing_config["reranker"]

        # sort results by retrieval score
        if reranker == "simple":
            results = sorted(results, key=lambda r: r.score, reverse=True)

        # sort results using sentence transformer
        elif reranker == "cross_encoder":
            results = self.cross_encoder.postprocess_nodes(nodes=results, query_str=user_query)
        logger.info(f"Reranking complete.")
        return results
