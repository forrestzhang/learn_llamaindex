from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings


Settings.llm = Ollama(model="llama3")

messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]

response = Settings.llm.chat(messages)
print(response)