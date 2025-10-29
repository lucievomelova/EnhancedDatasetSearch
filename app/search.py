import pandas as pd
from utils import setup_logger

logger = setup_logger(__name__)


def search(expanded_query: list, data_df: pd.DataFrame, limit: int = 100) -> pd.DataFrame | None:
    """Search the data for the given query."""

    logger.info("Searching.")
    if expanded_query:
        def _match_score(row):
            text = f"{row['název']} {row['popis']}".lower()
            return sum(1 for w in expanded_query if w in text)
        data_df['score'] = data_df.apply(_match_score, axis=1)
        results = data_df[data_df['score'] > 0].sort_values(by='score', ascending=False)[:limit]
        logger.info("Searching complete, found %s relevant results.", results.shape[0])
        return results
    return None
