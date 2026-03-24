from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor import SimilarityPostprocessor, SentenceTransformerRerank

from utils import setup_logger

logger = setup_logger(__name__)

class PostProcessor():
    """Search result postprocessor."""

    def __init__(self, postprocessing_config: dict):
        self.postprocessing_config = postprocessing_config


    def _remove_title_and_metadata_from_text(self, text: str) -> str:
        """Remove dataset title from the text chunk."""
        without_title = text.split("\n")[1:]  # title is on the first line
        joined_string = '\n'.join(without_title)  # join the split string back into one
        without_metadata = joined_string.split("Poskytovatel: ")[0]  # metadata is starting from "Poskytovatel: "
        return without_metadata


    def _format_nodes(self, nodes: list[NodeWithScore]) -> list[dict[str, str | list | None]]:
        formatted_nodes = []
        for node in nodes:
            text = self._remove_title_and_metadata_from_text(node.text)
            formatted_node = {
                "title": node.metadata["title"],
                "url": node.metadata["url"],
                "text": text,
                "explanation": "",
                "metadata": {
                    "themes": node.metadata.get("themes", []),
                    "keywords": node.metadata.get("keywords", []),
                    "categories": node.metadata.get("categories", []),
                    "regions": node.metadata.get("regions", []),
                    "time periods": node.metadata.get("time_periods", []),
                    "provider": node.metadata.get("provider", ""),
                }
            }
            formatted_nodes.append(formatted_node)

        return formatted_nodes

    def run(self, user_query: str,
                              extended_query: str,
                              results: list[NodeWithScore],
                              intent: dict[str, str]) -> list[dict[str, str | list | None]] | None:
        """Post-process search results."""
        if not results:
            logger.info("No results found.")
            return None

        logger.info(f"Post-processing {len(results)} results.")
        results = self.rerank(user_query, results, intent)
        results = self._format_nodes(results)
        return results


    def rerank(self, user_query: str, results: list[NodeWithScore], intent: dict[str, str]) -> list[NodeWithScore] | None:
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
