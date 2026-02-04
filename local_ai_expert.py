from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import AsyncOpenAI
from typing import List
import httpx
from typing import List
from pymilvus import MilvusClient 

load_dotenv()

base_url = "http://192.168.0.19:11434/v1"
OLLAMA_HTTP = base_url.replace("/v1", "")
MILVUS_URI = os.getenv("MILVUS_URI", "milvus_demo.db")  # Lite path or full URI
COLLECTION = os.getenv("MILVUS_COLLECTION", "webpages")
EMBED_DIM = 1024

model = OpenAIModel(
    model_name="llama3.2", provider=OpenAIProvider(base_url=base_url, api_key="ollama")
)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    milvus_client: MilvusClient
   

system_prompt = """
You are an expert assistant for the University of Oregon's Computer Science program. 
You have access to the department's official documentation, including course listings, program requirements, research opportunities, admissions information, and other academic resources.

Your sole purpose is to assist with questions related to the University of Oregon Computer Science program. 
You do not answer unrelated questions, and you always stay focused on academic and departmental content.

Do not ask the user for permission to act—take direct action. Always consult the documentation using the available tools before answering, unless you already have the relevant information.

When first accessing documentation, always begin with a RAG (retrieval-augmented generation) search. 
Also check the list of available documentation pages and retrieve the content of any page that may help answer the question.

Always be transparent: if you cannot find the answer in the documentation or the appropriate page URL, let the user know clearly and honestly.
"""

pydantic_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

async def get_embedding(text: str) -> List[float]:
    """Use local Ollama server to generate embeddings."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_HTTP}/api/embeddings",
                json={
                    "model": "mxbai-embed-large",
                    "prompt": text
                },
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()["embedding"]
    except Exception as e:
        print(f"Error getting embedding from Ollama: {e}")
        return [0.0] * 1024  # mxbai-embed-large returns 1024-dim vectors@pydantic_ai_expert.tool

@pydantic_ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        milvus = ctx.deps.milvus_client
        query_embedding = await get_embedding(user_query)
        
        # Query Milvus Database for relevant documents
        res = milvus.search(
            collection_name=COLLECTION,
            data=[query_embedding],
            anns_field="vector",
            limit=5,
            output_fields=["title", "url", "content", "chunk_number"],
            params={"metric_type": "COSINE"},  # ensure cosine matches your index
        )        
        
        formatted = []
        for hit in res[0]:
            # Each hit has .entity with dynamic fields (or dict-like in MilvusClient)
            ent = hit["entity"] if isinstance(hit, dict) else hit.entity
            title = ent.get("title", "Untitled")
            content = ent.get("content", "")
            url = ent.get("url", "")
            chunk_no = ent.get("chunk_number", None)
            header = f"# {title}" + (f" (chunk {chunk_no})" if chunk_no is not None else "")
            suffix = f"\n\nSource: {url}" if url else ""
            formatted.append(f"{header}\n\n{content}{suffix}")
            print(f"\n{header}---------------{content}\n")
        return "\n\n---\n\n".join(formatted)
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"



