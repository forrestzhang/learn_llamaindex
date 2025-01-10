from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import MarkdownReader
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv
import os
import qdrant_client
from llama_index.llms.litellm import LiteLLM
# load env
load_dotenv()

# Settings.llm = Ollama(model="llama3")
Settings.llm = LiteLLM(model="deepseek/deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"), api_base="https://api.deepseek.com/v1")
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-large-en-v1.5")

qdrant_client = qdrant_client.QdrantClient(
    host="localhost",
    port=6333,

)
collection_name = "uORFs"
data_dir = "data/markdown"

# Create vector store
vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)

# Get all markdown files in data directory
markdown_files = [f for f in os.listdir(data_dir) if f.endswith('.md')]

# check if collection exists
points = []
try:
    collection_info = qdrant_client.get_collection(collection_name=collection_name)
    
    print(f"Collection {collection_name} already exists")

    # Get existing documents from Qdrant
    points = qdrant_client.scroll(
        collection_name=collection_name,
        scroll_filter=None,
        limit=999999 # Adjust based on expected number of documents
    )[0]

except Exception as e:

    print(f"Collection {collection_name} does not exist")

# Extract filenames from metadata of existing documents
existing_files = []
if points:
    existing_files = [point.payload.get('file_name', '') for point in points if point.payload]


# Check which files need to be loaded
files_to_load = [f for f in markdown_files if f not in existing_files]

if not files_to_load:
    print("All markdown files already loaded, skipping document loading...")
    index = VectorStoreIndex.from_vector_store(vector_store)
else:
    print(f"Loading {len(files_to_load)} new markdown files...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Load only new documents
    # documents = SimpleDirectoryReader(data_dir).load_data()
    
    reader = MarkdownReader(remove_images=True, remove_hyperlinks=True)
    documents = []
    for file in files_to_load:
        file_path = os.path.join(data_dir, file)
        docs = reader.load_data(file_path)
        # Add metadata to each document
        for doc in docs:
            doc.metadata.update({
                "file_name": file,
                "file_path": file_path,
                "file_type": "markdown"
            })
        documents.extend(docs)

    # Create index and store in Qdrant
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

# Create query engine to test
query_engine = index.as_query_engine()

# Test query
response = query_engine.query(
    "what is the definition of uORF?"
)
print("\n-------Test Query Response-------")
print(response)
