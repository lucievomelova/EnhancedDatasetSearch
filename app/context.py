import logging

from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow import AgentWorkflow


logging.getLogger("httpx").setLevel(logging.DEBUG)

class Agent:
    def __init__(self, index: VectorStoreIndex, retriever: BaseRetriever, llm: FunctionCallingLLM):
        self.index = index
        self.retriever = retriever
        self.query_engine = self.index.as_query_engine()
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=8196)
        self.agent = FunctionAgent(
            llm=llm,
            tools=self._create_tools(),
            verbose=True,
            memory=self.memory,
            system_prompt=(
                "You are an AI chatbot at a dataset portal. The user is searching for datasets."
                "There is a RAG database containing information about provided datasets."
                "Chat with the user naturally. When the user asks about data or datasets, "
                "use the 'rag_db_search' tool to retrieve relevant datasets."
                "If you decide to use the tool and you think that the retrieved datasets are relevant "
                "to the query, make sure to provide a commentary for each dataset "
                "you provide to the user, explaining why it is relevant. Offer the most relevant datasets "
                "with short explanation. Provide also the title and URL of each dataset to the user."
                "You can ask for additional information if you don't have enough information "
                "or if the retrieved datasets are not relevant enough."
                "Use the provided tool at most once and then return an answer to the user."
            ),
        )
        self.workflow = AgentWorkflow(
            agents=[self.agent],
        )

    def _create_tools(self) -> list[QueryEngineTool]:
        rag_tool = QueryEngineTool(
            query_engine=self.query_engine,
            metadata=ToolMetadata(
                name="rag_db_search",
                description=(
                    "RAG DB contains information about all provided datasets on the portal. "
                    "Use this tool when you want to find relevant datasets based on user queries. "
                    "Return the most relevant chunks along with url from the metadata "
                    "as a python dictionary with 'text' and 'url' fields. So return a list of dictionaries."
                    "Don't include any additional commentary and use only info from the RAG DB, "
                ),
            ),
        )
        return [rag_tool]

    async def run_chatbot(self) -> None:
        print("Chatbot is ready. Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            response = await self.agent.run(user_msg=user_input, memory=self.memory)
            print(f"Bot: {response}")
            print(self.memory.get())
