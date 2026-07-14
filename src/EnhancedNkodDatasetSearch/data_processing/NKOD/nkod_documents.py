import pandas as pd

from llama_index.core import Document

from EnhancedNkodDatasetSearch.data_processing.documents import DocumentConverter
from EnhancedNkodDatasetSearch.utils import setup_logger

logger = setup_logger(__name__)


class NkodDocumentConverter(DocumentConverter):
    """Class for creating llama_index documents from NKOD datasets metadata."""

    def create_documents(self, datasets: pd.DataFrame) -> list[Document]:
        """Create a list of llama_index documents from the provided metadata dataset.

        The result is the list of all documents that should be uploaded to the knowledge base.
        """
        if datasets.empty:
            return []
        documents = self._create_documents(datasets)
        return documents

    @staticmethod
    def _create_documents(datasets: pd.DataFrame) -> list[Document]:
        """Create llama index Documents from the metadata dataset. Each row will be used to create one Document.

        The Document text will be: title + description + keywords + themes + categories + provider + spatial_coverage
        + temporal_coverage. All columns will also be stored in the metadata of the Document.
        """
        documents = []
        descriptions = datasets["description"]
        datasets = datasets.where(datasets.notna(), None)
        metadata_df = datasets.drop(columns="description").to_dict(orient="records")
        logger.info(f"Creating {len(datasets)} llama_index Documents.")

        for description, metadata in zip(descriptions, metadata_df):
            text = f"""
                {metadata['title']}
                {description}\n
                Poskytovatel: {metadata['provider']}
                Klíčová slova: {metadata['keywords']}
                Témata: {metadata['themes']}
                Kategorie: {metadata['categories']}
                Prostorové pokrytí": {metadata['spatial_coverage']}
                Časové pokrytí: {metadata['temporal_coverage']}
            """
            document = Document(text=text, metadata=metadata, id_=metadata["url"])
            documents.append(document)

        logger.info("Documents created.")
        return documents
