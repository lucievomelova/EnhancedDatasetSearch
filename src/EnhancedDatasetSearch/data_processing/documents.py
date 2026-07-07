from abc import ABC, abstractmethod

import pandas as pd
from llama_index.core import Document


class DocumentConverter(ABC):
    """Generic class for creating llama_index documents from datasets metadata."""

    @abstractmethod
    def create_documents(self, datasets: pd.DataFrame) -> list[Document]:
        """Create a list of llama_index documents from the metadata datset.

        The result is the list of all documents that should be uploaded to the knowledge base."""
