import json

import httpx
from ollama import Client

from EnhancedNkodDatasetSearch.utils import setup_logger

logger = setup_logger(__name__)


class OllamaClient:
    """Custom Ollama client.

    This custom client serves as an intermediate step for calling LLMs. It provides a simple interface:
    get_llm_response (returns text) and get_llm_json_response (returns structured response - json).
    The LLM parameters can be set up using the llm_config argument, timeout in seconds can be specified using
    the timeout argument."""
    def __init__(self, llm_config: dict, timeout: int | None = None):
        """Initialize the client.

        Args:
            llm_config (dict): LLM configuration dict, must contain base_url, model_name and context_length.
            timeout (int, optional): LLM timeout (in seconds)."""
        self.llm_config = llm_config
        self.client = Client(
            host=llm_config["base_url"],
            timeout=timeout
        )

    def get_llm_response(self, prompt: str) -> str:
        """Call the LLM with the specified prompt."""
        response = self.client.generate(model=self.llm_config["model_name"],
                                        prompt=prompt,
                                        options={
                                            "num_ctx": self.llm_config["context_length"],
                                            "temperature": 0}
                                        ).response
        return response

    def get_llm_json_response(self, prompt: str, num_retry_attempts: int = 3) -> tuple[dict, int]:
        """Call the LLM with the specified prompt and parse the returned json response to dict."""
        retry = 0
        while True:
            try:
                response = self.client.generate(model=self.llm_config["model_name"],
                                                prompt=prompt,
                                                format='json',
                                                options={
                                                    "num_ctx": self.llm_config["context_length"],
                                                    "temperature": 0}
                                                ).response
                response_dict = json.loads(response)
                return response_dict, retry
            except json.decoder.JSONDecodeError as e:
                logger.error(f"Error while deserializing LLM response from json: {e}. Retrying...")
                retry += 1
                if retry >= num_retry_attempts:
                    return {}, retry
            except httpx.ReadTimeout as e:
                logger.error(f"Timeout error: {e}. Retrying...")
                retry += 1
                if retry >= num_retry_attempts:
                    return {}, retry
