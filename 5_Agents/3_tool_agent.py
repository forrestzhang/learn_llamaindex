from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool
from llama_index.core import Settings
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from llama_index.tools.wikipedia import WikipediaToolSpec

# 可以从 https://llamahub.ai/ 找到合适的tool
"""
pip install llama-index-tools-yahoo-finance
pip install llama-index-tools-wikipedia 
"""
Settings.llm = Ollama(model="llama3")

# function tools
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

# Yahoo Finance tool
print("----------------------Yahoo Finance Tool----------------------")

finance_tools = YahooFinanceToolSpec().to_tool_list()
finance_tools.extend([multiply_tool, add_tool])

yahoo_fiance_agent = ReActAgent.from_tools(finance_tools, verbose=True, max_iterations=30)

response = yahoo_fiance_agent.chat("What is the current price of NVDA?")

print(response)

# Wikipedia tool
print("----------------------Wikipedia Tool----------------------")
wiki_tools = WikipediaToolSpec().to_tool_list()
wiki_agent = ReActAgent.from_tools(wiki_tools, verbose=True, max_iterations=30)

response = wiki_agent.chat("What is the capital of France?")
print(response)

