from data_processing.NKOD.data_catalog import DataCatalog
from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor import SentenceTransformerRerank

from utils import setup_logger

logger = setup_logger(__name__)

class PostProcessor:
    """Search result postprocessor."""

    def __init__(self, postprocessing_config: dict, data_catalog: DataCatalog) -> None:
        self.postprocessing_config = postprocessing_config
        self.data_catalog = data_catalog

    def run(self, user_query: str,
                              extended_query: str,
                              results: list[NodeWithScore],
                              intent: dict[str, str]) -> list[dict | None]:
        """Post-process search results."""
        if not results:
            logger.info("No results found.")
            return None

        logger.info(f"Post-processing {len(results)} results.")
        results = self.rerank(user_query, results, intent)
        results_with_info = [self.data_catalog.get_dataset_by_url(str(res.metadata["url"])) for res in results]
        return results_with_info


    def rerank(self, user_query: str, results: list[NodeWithScore], intent: dict[str, str]) -> list[NodeWithScore]:
        """ Rerank search results."""
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
        elif reranker == "sentence_transformer":
            transformer = SentenceTransformerRerank(
                model=self.postprocessing_config["sentence_transformer_model"], top_n=self.postprocessing_config["top_k"]
            )

            results = transformer.postprocess_nodes(nodes=results, query_str=user_query)
        logger.info(f"Reranking complete.")
        return results
