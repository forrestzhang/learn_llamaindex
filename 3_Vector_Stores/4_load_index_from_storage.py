import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

"""
Use VectorStoreIndex to load builed index from qdrant storage
"""

Settings.llm = Ollama(model="llama3")
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-base-en-v1.5")

client = qdrant_client.QdrantClient(
    host="localhost",
    port=6333
)

vector_store = QdrantVectorStore(client=client, collection_name="pubmed_demo")


index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

query_engine = index.as_query_engine()

print("-------Demo1-------")
response = query_engine.query(
    "Is rice sensitvie to low temperature stress?"
)

print(str(response))


print("-------Demo2-------")
response = query_engine.query(
    "what is Cis-Element?"
)

print(str(response))
