from abc import ABC, abstractmethod

import pandas as pd


class KnowledgeGraph(ABC):
    """Class representing a generic metadata knowledge graph."""

    @abstractmethod
    def create_or_update_kg(self, datasets: pd.DataFrame, new_datasets: pd.DataFrame, removed_urls: list) -> None:
        """Create or update the knowledge graph."""

    @abstractmethod
    def get_similar_datasets(self, dataset_url: str) -> dict[str, list[tuple[str, float]]]:
        """Get similar datasets based on the knowledge graph.

        Returns:
            A dictionary with keys: key is similarity category, value is a list of (url, score) tuples of similar
            datasets' urls and similarity scores.
        """
