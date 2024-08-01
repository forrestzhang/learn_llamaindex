from llama_index.core.agent import ReActAgent
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

"""
Use FunctionTool to create tools for adding and multiplying numbers. 
Those tools will be used by the ReActAgent to solve the math problem related with 2023 canadian budget.
"""

Settings.llm = Ollama(model="llama3") # for this project glm4 is better than llama3

# Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-base-en-v1.5")
Settings.embed_model = OllamaEmbedding("nomic-embed-text:latest")

def multiply(a: float, b: float) -> float:
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: float, b: float) -> float:
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)


# Rag
documents = SimpleDirectoryReader("./data/2023budget/").load_data()

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

# query budget for test
print("Querying the budget for test")
response = query_engine.query("What was the total amount of the 2023 Canadian federal budget?")
print(response)
print("\n------------------------\n")

# rag as tool
budget_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="canadian_budget_2023_tool",
    description="Use this tool to lookup the detail information of the 2023 Canadian federal budget.",
)

agent = ReActAgent.from_tools([multiply_tool, add_tool, budget_tool], llm=Settings.llm, verbose=True, max_iterations=30)

response = agent.chat("What is the total amount of the 2023 Canadian federal budget multiplied by 3? Go step by step, using a tool to do any math.")

print(response)

# response = agent.chat("How much exactly was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget?")

# print(response)

# response = agent.chat("How much was allocated to a implement a means-tested dental care program in the 2023 Canadian federal budget?")

# print(response)

# response = agent.chat("How much was the total of those two allocations added together? Use a tool to answer any questions.")

# print(response)