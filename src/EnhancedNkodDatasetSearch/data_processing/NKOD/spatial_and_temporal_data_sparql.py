"""File containing functions for extracting spatial and temporal coverage from NKOD using SPARQL and processing
of the retrieved data."""

from datetime import datetime

import pandas as pd
from rdflib import Graph
from rdflib.plugins.stores.sparqlstore import SPARQLStore

from EnhancedNkodDatasetSearch.utils import setup_logger

logger = setup_logger(__name__)


def get_spatial_and_temporal_coverage(graph: Graph) -> tuple[dict, dict]:
    """Run a SPARQL query to get spatial and temporal coverage from NKOD for all datasets."""
    query = """
    PREFIX dct: <http://purl.org/dc/terms/>
    PREFIX dcat: <http://www.w3.org/ns/dcat#>
    PREFIX schema: <http://schema.org/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?dataset (STR(?start) AS ?startStr) (STR(?end) AS ?endStr) ?spatialName
    WHERE {
        ?dataset dct:temporal ?period ;
                 dct:spatial ?spatial .

        OPTIONAL { ?period dcat:startDate ?start }
        OPTIONAL { ?period dcat:endDate ?end }

        OPTIONAL { ?spatial schema:name ?schemaName }
        OPTIONAL { ?spatial rdfs:label ?rdfsLabel }
        BIND(COALESCE(?schemaName, ?rdfsLabel) AS ?spatialName)

        FILTER(!BOUND(?spatialName) || lang(?spatialName) = "cs")
    }
    """
    spatial_data = {}
    temporal_data = {}
    for dataset, start, end, spatial in graph.query(query):
        url = str(dataset)
        if url not in spatial_data:
            spatial_data[url] = []
        if url not in temporal_data:
            temporal_data[url] = []
        temporal_coverage = get_range_from_start_and_end(str(start), str(end))
        if temporal_coverage not in temporal_data[url]:  # check so we don't add something multiple times
            temporal_data[url].append(temporal_coverage)
        if spatial and str(spatial) not in spatial_data[url]:  # check so we don't add something multiple times
            # skip Czech Republic as spatial coverage because it is not a useful information
            if str(spatial) not in ["Česká Republika", "Česká republika"]:
                spatial_data[url].append(str(spatial))
    return temporal_data, spatial_data


def get_year_from_date(date: str) -> str:
    """Extract year from date string."""
    # there multiple formats used on NKOD
    for date_format in ("%Y-%m-%d", "%Y-%m", "%Y-%m-%d %H:%M:%S"):
        try:
            return str(datetime.strptime(date, date_format).year)
        except ValueError:
            continue
    return date  # cannot easily extract year


def get_range_from_start_and_end(temporal_start: str, temporal_end: str) -> str:
    """Process extracted temporal coverage into a suitable format - a year or range of years.

    Get either a range of years (e.g. 2010 – 2020) or a single year if start and end are the same."""
    year_start = get_year_from_date(temporal_start)
    year_end = get_year_from_date(temporal_end)
    if year_start == year_end:
        return year_start  # return single year, because start and end are the same
    elif year_start  == "None":
        return year_end  # return single year, because start is not set
    elif year_end == "None":
        return year_start  # return single year, because end is not set
    return f"{year_start} – {year_end}"  # range


def add_metadata_to_datasets_from_sparql(config: dict, datasets: pd.DataFrame | None) -> None:
    """Add spatial and temporal coverage to datasets.
    
    These metadata will be extracted using a sparql query, processed into a suitable format and then put in the
    corresponding columns in the given datframe."""
    logger.info("Adding spatial and temporal coverage data.")
    if datasets is None:
        logger.error("No datasets provided.")
        return
    endpoint = config["data"]["sparql_endpoint"]
    graph = Graph(SPARQLStore(endpoint))

    try:
        temporal_data, spatial_data = get_spatial_and_temporal_coverage(graph)
        datasets["temporal_coverage"] = datasets["url"].map(lambda x: temporal_data.get(x, []))
        datasets["spatial_coverage"] = datasets["url"].map(lambda x: spatial_data.get(x, []))
        logger.info("Spatial and temporal info added.")
    except Exception as e:
        logger.error(f"NKOD SPARQL endpoint inaccessible. Cannot retrieve spatial and temporal coverage: {str(e)}")
        # set all spatial and temporal coverage as empty if there's an error
        datasets["temporal_coverage"] = [[] for _ in range(len(datasets))]
        datasets["spatial_coverage"] = [[] for _ in range(len(datasets))]
