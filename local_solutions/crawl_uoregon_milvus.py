import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# ---- MODEL/EMBEDDINGS CLIENT (you’re already pointing at Ollama’s OpenAI-compatible API) ----
from openai import AsyncOpenAI  # keep your AsyncOpenAI us4000
# Supabase (unused now for storage; keeping import if you still need it elsewhere)
from supabase import create_client, Client

# ------------------ MILVUS ------------------
from pymilvus import MilvusClient, DataType, Function, FunctionType

load_dotenv()

# ====== MODEL + SUPABASE (unchanged) ======
base_url = os.getenv("OLLAMA_OPENAI_BASE", "http://192.168.0.19:11434/v1")
openai_client = AsyncOpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY", "dummy"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")

)

#VARIABLE SETUP
MILVUS_URI = os.getenv("MILVUS_URI", "milvus_demo_hybrid.db")  # Lite file path or http://localhost:19530 for server
COLLECTION = os.getenv("COLLECTION","webpages")
EMBED_MODEL=os.getenv("EMBED_MODEL" ,"nomic-embed-text")
EMBED_DIM = os.getenv("EMBED_DIM")
print("EMBED DIM iS: ", EMBED_DIM)
milvus_client = MilvusClient(MILVUS_URI)

# Create the collection if it doesn't exist (with dynamic fields so we can store arbitrary metadata)
#
"""
if not milvus_client.has_collection(COLLECTION):
    idx = milvus_client.prepare_index_params()
    idx.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
    # Tip: auto_id=True -> Milvus assigns PK for us. Dynamic fields let us store strings/JSON without schema boilerplate.
    milvus_client.create_collection(
        collection_name=COLLECTION,
        dimension=EMBED_DIM,
        index_params=idx,
        auto_id=True,
        enable_dynamic_field=True,  # <-- store url/title/summary/content/metadata freely
        consistency_level="Bounded"  # good default
    )
"""
#This schema enables hybrid search. 
if not milvus_client.has_collection(COLLECTION):
    #Schema for hybrid search. 
    schema = milvus_client.create_schema(auto_id=True, description="webpages hybrid", enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("url", DataType.VARCHAR, max_length=1024)
    schema.add_field("chunk_number", DataType.INT64)

    # Raw text to be searched with BM25
    schema.add_field("content", DataType.VARCHAR, max_length=65535, enable_analyzer=True)

    # Dense vector (same dim you already use)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBED_DIM)

    # Sparse vector to hold BM25 output
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

    # Tell Milvus to compute BM25(content) → sparse at insert time
    schema.add_function(Function(
        name="content_bm25",
        input_field_names=["content"],
        output_field_names=["sparse"],
        function_type=FunctionType.BM25,
    ))

    # Index both dense and sparse
    idx = MilvusClient.prepare_index_params()
    idx.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
    idx.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_bm25_idx",
        metric_type="BM25",                 # default BM25 scoring
        params={"inverted_index_algo": "DAAT_MAXSCORE"}
    )

    milvus_client.create_collection(
        collection_name=COLLECTION,
        schema=schema,
        index_params=idx,
        consistency_level="Bounded"
    )
@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    #title: str
    #summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

# ================== STORAGE (MILVUS) ==================
async def insert_chunk(chunk: ProcessedChunk):
    """
    Insert a processed chunk into Milvus.
    Uses asyncio.to_thread to avoid blocking the event loop with a sync Milvus call.
    """
    """row = {
        "url": chunk.url,
        "chunk_number": chunk.chunk_number,
        "title": chunk.title,
        "summary": chunk.summary,
        "content": chunk.content,
        "metadata": chunk.metadata,
        "vector": chunk.embedding,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    """
    row = {
        "url": chunk.url,
        "chunk_number": chunk.chunk_number,
        "content": chunk.content,
        "vector": chunk.embedding,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Optional: Dedup by (url, chunk_number) before insert.
    # Milvus doesn't enforce unique constraints; quick query helps avoid duplicates.
    try:
        existing = await asyncio.to_thread(
            milvus_client.query,
            collection_name=COLLECTION,
            filter=f'url == "{chunk.url}" and chunk_number == {chunk.chunk_number}',
            output_fields=["url", "chunk_number"],
            limit=1
        )
        if existing:
            print(f"Skip duplicate: {chunk.url} [chunk {chunk.chunk_number}]")
            return
    except Exception as e:
        # Query might fail if the collection is brand new and hasn’t loaded; we’ll just attempt insert.
        print(f"Dedup check error (continuing to insert): {e}")

    try:
        await asyncio.to_thread(
            milvus_client.insert,
            collection_name=COLLECTION,
            data=[row]
        )
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
    except Exception as e:
        print(f"Error inserting into Milvus: {e}")

# ============= TEXT CHUNKING / SUMMARIZATION / EMBEDDING (unchanged) =============
def chunk_text(text: str, chunk_size: int = 3000) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block
        elif '\n\n' in chunk:
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:
                end = start + last_break
        elif '. ' in chunk:
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(start + 1, end)
    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    system_prompt = (
        "You are an AI that extracts titles and summaries from documentation chunks. "
        "Return a JSON object with 'title' and 'summary' keys. "
        "Keep both concise."
    )
    try:
        response = await openai_client.chat.completions.create(
            model="llama3.2:latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title/summary: {e}")
        return {"title": "Untitled", "summary": "Summary unavailable."}

async def get_embedding(text: str) -> List[float]:
    try:
        resp = await openai_client.embeddings.create(
            model=EMBED_MODEL,
            input=text
        )
        return resp.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0.0] * EMBED_DIM

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    #extracted = await get_title_and_summary(chunk, url)
    embedding = await get_embedding(chunk)
    metadata = {
        "source": "pydantic_ai_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        #title=extracted.get("title", "Untitled"),
        #summary=extracted.get("summary", ""),
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )

async def process_and_store_document(url: str, markdown: str):
    chunks = chunk_text(markdown)
    processed = await asyncio.gather(
        *[process_chunk(c, i, url) for i, c in enumerate(chunks)]
    )
    await asyncio.gather(*[insert_chunk(pc) for pc in processed])

# ============= CRAWLER (unchanged) =============
async def crawl_parallel(urls: List[str], max_concurrent: int = 1):
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    try:
        sem = asyncio.Semaphore(max_concurrent)
        async def process_url(url: str):
            async with sem:
                result = await crawler.arun(url=url, config=crawl_config, session_id="session1")
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        await asyncio.gather(*[process_url(u) for u in urls])
    finally:
        await crawler.close()

def get_pydantic_ai_docs_urls() -> List[str]:
    try:
        with open("sitemap.xml", "r") as f:
            xml_data = f.read()
        root = ElementTree.fromstring(xml_data)
        ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        return [loc.text for loc in root.findall('.//ns:loc', ns)]
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

# ============= SIMPLE SEMANTIC SEARCH (Milvus) =============
async def search_milvus(query: str, top_k: int = 5):
    vec = await get_embedding(query)
    try:
        results = milvus_client.search(
            collection_name=COLLECTION,
            data=[vec],
            anns_field="vector",
            limit=top_k,
            output_fields=["url", "chunk_number", "title", "summary"]
        )
        # results is a list with one element (for one query vector)
        hits = results[0] if results else []
        return [
            {
                "score": hit.get("distance", None),
                "url": hit["entity"].get("url"),
                "chunk_number": hit["entity"].get("chunk_number"),
                "title": hit["entity"].get("title"),
                "summary": hit["entity"].get("summary"),
            }
            for hit in hits
        ]
    except Exception as e:
        print(f"Search error: {e}")
        return []

# ============= MAIN =============
async def main():
    urls = get_pydantic_ai_docs_urls()
    if not urls:
        print("No URLs found to crawl")
        return
    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())

