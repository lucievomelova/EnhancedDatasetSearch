import pandas as pd


class KnowledgeGraph:
    """Class representing a metadata knowledge graph."""
    def create_kg(self, datasets: pd.DataFrame) -> None:
        """Create knowledge graph."""
        pass


    def get_similar_datasets(self, dataset_url: str) -> dict[str, list[tuple[str, float]]]:
        """Get similar datasets based on the knowledge graph.

        Returns:
            A dictionary with keys: key is similarity category, value is a list of (url, score) tuples of similar
            datasets' urls and similarity scores.
        """
        pass
