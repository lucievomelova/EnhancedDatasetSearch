import json

import ollama
from ollama import Client
from utils import setup_logger

logger = setup_logger(__name__)


class OllamaClient:
    def __init__(self, llm_config: dict, timeout: int | None = None):
        self.llm_config = llm_config
        self.client = Client(
            host=llm_config["base_url"],
            timeout=timeout
        )

    def get_llm_response(self, prompt: str) -> str:
        """Call LLM with the specified prompt."""
        response = self.client.generate(model=self.llm_config["model_name"],
                                        prompt=prompt,
                                        options={
                                            "num_ctx": self.llm_config["context_length"],
                                            "temperature": 0}
                                        ).response
        return response

    def get_llm_json_response(self, prompt: str, num_retry_attempts: int = 3) -> (dict | None, int):
        """Call LLM with the specified prompt and parse the returned json response to dict."""

        retry = 0
        while True:
            response = self.client.generate(model=self.llm_config["model_name"],
                                            prompt=prompt,
                                            format='json',
                                            options={
                                                "num_ctx": self.llm_config["context_length"],
                                                "temperature": 0}
                                            ).response
            try:
                response_dict = json.loads(response)
                return response_dict, retry
            except json.decoder.JSONDecodeError as e:
                logger.error(f"Error while deserializing LLM response from json: {e}. Retrying...")
                if retry >= num_retry_attempts:
                    return None, retry
                retry += 1
