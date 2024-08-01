from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.core.tools import FunctionTool
from llama_index.core import Settings
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from llama_index.tools.wikipedia import WikipediaToolSpec
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec

# 可以从 https://llamahub.ai/ 找到合适的tool
"""
pip install llama-index-tools-yahoo-finance
pip install llama-index-tools-wikipedia 
pip install llama-index-tools-duckduckgo    
"""
Settings.llm = Ollama(model="llama3.1")

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
print("\n\n----------------------Yahoo Finance Tool----------------------")

finance_tools = YahooFinanceToolSpec().to_tool_list()
finance_tools.extend([multiply_tool, add_tool])

yahoo_fiance_agent = ReActAgent.from_tools(finance_tools, verbose=False, max_iterations=100)

response = yahoo_fiance_agent.chat("What is the current price of NVDA?")
print("Q: What is the current price of NVDA?")

print(response)

# Wikipedia tool
print("\n\n----------------------Wikipedia Tool----------------------")
wiki_tools = WikipediaToolSpec().to_tool_list()
wiki_agent = ReActAgent.from_tools(wiki_tools, verbose=False, max_iterations=100)

response = wiki_agent.chat("What is the capital of France?")
print("Q: What is the capital of France?")
print(response)


# DuckDuckGo Search tool
print("\n\n----------------------DuckDuckGo Search Tool----------------------")
search_tools = DuckDuckGoSearchToolSpec().to_tool_list()
search_agent = ReActAgent.from_tools(search_tools, verbose=False, max_iterations=100)

response = search_agent.chat("which country host olympic 2024?")
print("Q: which country host olympic 2024?")
print(response)