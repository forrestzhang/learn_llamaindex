from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

documents = SimpleDirectoryReader("data").load_data()

Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-base-en-v1.5")

Settings.llm = Ollama(model="llama3")

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)

"""
pip install llama-index-llms-ollama
pip install llama-index-embeddings-huggingface
"""