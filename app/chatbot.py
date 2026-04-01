import json

import ollama
from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.schema import NodeWithScore
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow import AgentWorkflow

from app.pipeline import SearchPipeline
from utils import setup_logger, render_template

logger = setup_logger(__name__)

class Chatbot:
    def __init__(self, config: dict, search_pipeline: SearchPipeline):
        self.chatbot_config = config["chatbot"]
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=65000)
        self.condensed_chat_history: str | None = None
        self.search_pipeline = search_pipeline
        self.dataset_detail_url = config["url"] + "dataset"

    def chat(self, user_message: str) -> dict | None:
        """Chats with the user.

        After each user message, determines if we have enough info to perform a search or if more info is needed."""
        prompt = render_template("chatbot/chat.j2",
                                 {"chat_history": self.memory.get(), "user_message": user_message})

        retry = 0
        while retry < 3:
            try:
                response = ollama.generate(model=self.chatbot_config["llm"]["model_name"],
                                           prompt=prompt, format="json").response
                response_dict = json.loads(response)
                current_record = [
                    ChatMessage(role="user", content=user_message),
                    ChatMessage(role="assistant", content=response),
                ]
                self.memory.put_messages(current_record)
                logger.info(f"Chat response: {json.dumps(response_dict, indent=2)}")
                return response_dict
            except json.decoder.JSONDecodeError as e:
                logger.error("Error: {e}. Retrying...")
                retry += 1
        return None


    def generate_answer_from_results(self, search_results_json: str) -> str:
        """Generate an answer to the user query based on retrieved chunks."""

        prompt = render_template("chatbot/construct_final_answer.j2",
                                 {"chat_history": self.memory.get(), "search_results": search_results_json})
        response = ollama.generate(model=self.chatbot_config["llm"]["model_name"],
                                   prompt=prompt).response
        return response


    async def react_to_message(self, user_message: str) -> str:
        """Run the chatbot in a loop."""
        error_message = "There was an error during answer generation. Try rewriting your message or start a new chat."

        response = self.chat(user_message)
        if not response:  # there was an error during response generation
            return error_message
        if response.get("action") == "search":
            search_query = response.get("search_query", None)
            if search_query is not None:
                search_results = await self.search_pipeline.run(search_query)
                processed_results = self._keep_only_title_and_url(search_results)
                return self.generate_answer_from_results(processed_results)
        else:
            follow_up_question = response.get("follow_up_question", None)
            if follow_up_question is not None:
                return follow_up_question

        return error_message

    def _keep_only_title_and_url(self, search_results: list[dict]) -> str:
        """Keep only title of a dataset and its url to the dataset_detail page.

        Returns:
            Processed search results, already converted to json string."""
        processed_results = []
        for result in search_results:
            processed_results.append({
                "title": result["title"],
                "url": self.dataset_detail_url + "/" + result["url"]
            })
        return json.dumps(processed_results)
