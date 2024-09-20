from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, load_index_from_storage
from llama_index.agent.lats import LATSAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.storage import StorageContext
import os

import nest_asyncio 
nest_asyncio.apply()

llm = Ollama(model="qwen2.5:14b") # qwen2.5 is good for this task
# llm = Ollama(model="mistral-small", request_timeout=300)

embed_model = OllamaEmbedding("bge-large:latest")
# embed_model = HuggingFaceEmbedding("BAAI/bge-base-en-v1.5")

Settings.llm = llm
Settings.embed_model = embed_model

if not os.path.exists("/home/forrest/Github/learn_llamaindex/data/storage/lyft"):
    # load data
    lyft_docs = SimpleDirectoryReader(
        input_files=["/home/forrest/Github/learn_llamaindex/data/10k/lyft_2021.pdf"]
    ).load_data()
    uber_docs = SimpleDirectoryReader(
        input_files=["/home/forrest/Github/learn_llamaindex/data/10k/uber_2021.pdf"]
    ).load_data()

    # build index
    lyft_index = VectorStoreIndex.from_documents(lyft_docs)
    uber_index = VectorStoreIndex.from_documents(uber_docs)

    # persist index
    lyft_index.storage_context.persist(persist_dir="/home/forrest/Github/learn_llamaindex/data/storage/lyft")
    uber_index.storage_context.persist(persist_dir="/home/forrest/Github/learn_llamaindex/data/storage/uber")
else:
    storage_context = StorageContext.from_defaults(
        persist_dir="/home/forrest/Github/learn_llamaindex/data/storage/lyft"
    )
    lyft_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir="/home/forrest/Github/learn_llamaindex/data/storage/uber"
    )
    uber_index = load_index_from_storage(storage_context)

# uber_docs=SimpleDirectoryReader(input_files=['/home/forrest/Github/learn_llamaindex/data/UBER/uber_2021.pdf']).load_data()

# uber_index = VectorStoreIndex.from_documents(uber_docs)

# uber_index.storage_context.persist(persist_dir="/home/forrest/Github/learn_llamaindex/data/storage/uber")
lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021. "
                "Use a detailed plain text question as input to the tool. "
                "The input is used to power a semantic search engine."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021. "
                "Use a detailed plain text question as input to the tool. "
                "The input is used to power a semantic search engine."
            ),
        ),
    ),
]

agent_worker = LATSAgentWorker.from_tools(
    query_engine_tools,
    llm=llm,
    num_expansions=2,
    max_rollouts=-1,  # using -1 for unlimited rollouts
    verbose=True,
)
agent = AgentRunner(agent_worker)

task = agent.create_task(
    "Given the risk factors Uber described in their 10K files, "
    "what are the most important factors to consider for investing in Uber?"
)

step_output = agent.run_step(task.task_id)
# step_output_dict = step_output.dict() if hasattr(step_output, 'dict') else step_output
# agent.chat( "Given the risk factors Uber described in their 10K files, what are the most important factors to consider for investing in Uber?")

for step in (
    step_output.task_step.step_state["root_node"].children[0].current_reasoning
):
    print(step)
    print("---------")

for step in (
    step_output.task_step.step_state["root_node"]
    .children[0]
    .children[0]
    .current_reasoning
):
    print(step)
    print("---------")

while not step_output.is_last:
    step_output = agent.run_step(task.task_id)

response = agent.finalize_response(task.task_id)

print(str(response))