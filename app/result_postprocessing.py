import json
from typing import Dict, List
from llama_index.core.evaluation import ContextRelevancyEvaluator
from jinja2 import Environment, FileSystemLoader
import ollama
from llama_index.core.schema import NodeWithScore

from utils import setup_logger

logger = setup_logger(__name__)
env = Environment(loader=FileSystemLoader('prompts'))
intro_template = env.get_template("intro.j2")
intro_prompt = intro_template.render()
return_json_template = env.get_template("return_json.j2")
return_json_instructions = return_json_template.render()


def _remove_title_and_metadata_from_text(text: str) -> str:
    """Remove dataset title from the text chunk."""
    without_title = text.split("\n")[1:]  # title is on the first line
    joined_string = '\n'.join(without_title)  # join the split string back into one
    without_metadata = text.split("Poskytovatel: ")[0]  # metadata is starting from "Poskytovatel: "
    return without_metadata


def _format_nodes(nodes: list[NodeWithScore]) -> list[dict[str, str]]:
    formatted_nodes = []
    for node in nodes:
        text = _remove_title_and_metadata_from_text(node.text)
        formatted_node = {
            "title": node.metadata["title"],
            "url": node.metadata["url"],
            "text": text,
            "metadata": {
                "themes": node.metadata.get("themes", []),
                "keywords": node.metadata.get("keywords", []),
                "provider": node.metadata.get("provider", ""),
                "categories": node.metadata.get("categories", []),
                "regions": node.metadata.get("regions", []),
                "time_periods": node.metadata.get("time_periods", []),
            }
        }
        formatted_nodes.append(formatted_node)

    return formatted_nodes

def format_search_results(results: list[NodeWithScore]) -> list[dict[str, str]] | None:
    """Format search results, don't add any additional postprocessing."""
    if not results:
        logger.info("No results found.")
        return None
    logger.info(f"Formatting {len(results)} results.")
    results = _format_nodes(results)
    return results


def result_postprocessing(user_query: str,
                          extended_query: str,
                          results: list[NodeWithScore],
                          intent: Dict[str, str], k: int = 10) -> list[dict[str, str]] | None:
    """Post-process search results."""
    if not results:
        logger.info("No results found.")
        return None

    logger.info(f"Post-processing {len(results)} results.")
    results = _format_nodes(results)
    results_str = json.dumps(results, indent=2, ensure_ascii=False)

    template = env.get_template("rerank_results.j2")
    prompt = template.render(intro=intro_prompt,
                             user_query=user_query,
                             extended_query=extended_query,
                             categories_intent=", ".join(intent["categories"]),
                             regions_intent=", ".join(intent["regions"]),
                             time_periods_intent=", ".join(intent["time_periods"]),
                             results=results_str,
                             k=k,
                             return_json_instructions=return_json_instructions)

    response = ollama.generate(model='mistral-small3.2',
                               format="json",
                               prompt=prompt,
                               options={
                                       "temperature": 0,
                               }).response

    result = json.loads(response)
    results_as_list = [r for r in result.values()]
    logger.info(f"Reranked results:\n{[r["title"] + ": " + str(r["score"]) + "\n" for r in results_as_list]}.")
    return results_as_list


def rerank_with_llm(llm, user_query: str, results: Dict[str, List[NodeWithScore]], intent: Dict[str, str], k: int = 10):
    """ Rerank search results using LLM."""
    evaluator = ContextRelevancyEvaluator(llm=llm)

    for query, nodes in results.items():
        for node in nodes:
            score = evaluator.evaluate(
                query=query,
                contexts=[node.text]
            ).score

            print(score, node.text[:200])
