import json
import gzip
import qdrant_client
from glob import glob

from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core import StorageContext
from llama_index.core import Document


Settings.llm = Ollama(model="llama3")
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-base-en-v1.5")

splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=Settings.embed_model
)

# also baseline splitter
base_splitter = SentenceSplitter(chunk_size=512)

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

vector_store = QdrantVectorStore(client=client, collection_name="pubmed_demo")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

documents = []

jsonfiles = glob("../data/pubmed_cis_json/*.json.gz")

# print(jsonfiles)

for jsonfile in jsonfiles:
    with gzip.open(jsonfile) as f:
        data = json.load(f)

    for pmid in data:
        #print(pmid)
        abstract = data[pmid]["abstract"]
        journal = data[pmid]["journal"]
        pubdate = data[pmid]["pubdate"]
        document = Document(text=abstract, 
                            metadata = {"pmid": pmid, "journal": journal, "pubdate": pubdate})
        documents.append(document)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

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