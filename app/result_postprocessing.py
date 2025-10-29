import pandas as pd
from utils import setup_logger

logger = setup_logger(__name__)


def result_postprocessing(results: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    """Postprocess the search results and return the most relevant datasets."""

    results = results[:limit]
    logger.info("Results: %s", results)
    return results