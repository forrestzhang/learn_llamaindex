import qdrant_client

from time import time
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import MarkdownReader
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.litellm import LiteLLM
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)

from dotenv import load_dotenv
import os


# load env
load_dotenv()

# Settings.llm = Ollama(model="llama3")
Settings.llm = LiteLLM(model="deepseek/deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"), api_base="https://api.deepseek.com/v1")
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-large-en-v1.5")

qdrant_client = qdrant_client.QdrantClient(
    host="localhost",
    port=6333,
)

vector_store = QdrantVectorStore(client=qdrant_client, collection_name="uORFs")


index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

query_engine = index.as_query_engine(similarity_top_k=10, similarity_cutoff=0.3)

start_time = time()
response = query_engine.query(
    "What is the role of uORFs in plant stress response?",
)
end_time = time()
print("-------Normal RAG-------")
print(str(response))
print(f"Time taken: {end_time - start_time} seconds")


rerank = FlagEmbeddingReranker(model="BAAI/bge-reranker-large", top_n=5)
query_engine_with_rerank = index.as_query_engine(
    similarity_top_k=10, node_postprocessors=[rerank], similarity_cutoff=0.3
)

start_time = time()
response_with_rerank = query_engine_with_rerank.query(
    "What is the role of uORFs in plant stress response?",
)
end_time = time()
print("-------Reranked RAG-------")
print(str(response_with_rerank))
print(f"Time taken: {end_time - start_time} seconds")


"""
pip install llama-index-postprocessor-flag-embedding-reranker
pip install git+https://github.com/FlagOpen/FlagEmbedding.git
pip install air_benchmark
pip install faiss-cpu 
pip install beir
pip install mteb 
"""