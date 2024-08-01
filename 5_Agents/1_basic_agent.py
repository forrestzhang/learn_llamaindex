from llama_index.core.agent import ReActAgent
from llama_index.core import Settings
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="llama3")

def multiply(a: float, b: float) -> float:
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: float, b: float) -> float:
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

agent = ReActAgent.from_tools([multiply_tool, add_tool], llm=Settings.llm, verbose=True, max_iterations=30)

response = agent.chat("What is 20+(2*4)? Use a tool to calculate every step.")


print(response)