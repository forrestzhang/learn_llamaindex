import os
import qdrant_client

from crewai import Agent, Task, Crew, Process
from crewai_tools import LlamaIndexTool

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


Settings.llm = Ollama(model="qwen2.5")
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-base-en-v1.5")


documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

client = qdrant_client.QdrantClient(
    # you can use :memory: mode for fast and light-weight experiments,
    # it does not require to have Qdrant deployed anywhere
    # but requires qdrant-client >= 1.1.1
    # location=":memory:"
    # otherwise set Qdrant instance address with:
    # url="http://<host>:<port>"
    # otherwise set Qdrant instance with host and port:
    host="localhost",
    port=6333
    # set API KEY for Qdrant Cloud
    # api_key="<qdrant-api-key>",
)

vector_store = QdrantVectorStore(client=client, collection_name="paul_graham")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()


query_tool = LlamaIndexTool.from_query_engine(
    query_engine,
    name="Query Tool",
    description="Use this tool to lookup the detail information of the query.",
)

query_tool.args_schema.schema()



"""
pip install 'crewai[tools]'
"""