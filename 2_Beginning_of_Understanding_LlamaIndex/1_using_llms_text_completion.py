from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

Settings.llm = Ollama(model="llama3")

completions = Settings.llm.stream_complete("Paul Graham is ")
for completion in completions:
    print(completion.delta, end="")