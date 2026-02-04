
# keyword_search_milvus.py
# Sparse-only BM25 keyword search against Milvus ("sparse" field)
import os
import sys
import asyncio
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv, find_dotenv
from pymilvus import MilvusClient

# ---------- .env loading ----------
# Finds the nearest .env (from current working directory upward) and loads it.
load_dotenv(find_dotenv(usecwd=True), override=True)

# ---------- Config from env (UPPERCASE keys) ----------
MILVUS_URI  = os.getenv("MILVUS_URI", "milvus_demo_hybrid.db")   # e.g. http://localhost:19530 or Lite file path
COLLECTION  = os.getenv("COLLECTION", "webpages")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")       # not used here but kept for consistency
EMBED_DIM   = int(os.getenv("EMBED_DIM", "768"))                  # loaded correctly (example/debug)

print("EMBED_DIM is:", EMBED_DIM, flush=True)

# ---------- Milvus client ----------
milvus_client = MilvusClient(MILVUS_URI)

async def keyword_search_milvus(
    query: str,
    top_k: int = 8,
    drop_ratio: float = 0.2,
    filter_expr: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    BM25 keyword search using Milvus 'sparse' field only.
    - drop_ratio: drop low-weight terms during search (0.0 ~ 0.5)
    - filter_expr: optional scalar filter, e.g. 'chunk_number >= 0 and url like "https://example.com/%"'
    """
    try:
        search_params = {"drop_ratio_search": drop_ratio}

        kwargs = dict(
            collection_name=COLLECTION,
            data=[query],                 # raw text; Milvus builds sparse BM25 query internally
            anns_field="sparse",
            limit=top_k,
            search_params=search_params,
            output_fields=["url", "chunk_number", "content"],
        )
        # Only include 'filter' when provided (some client versions double-pass expr if not careful)
        if filter_expr:
            kwargs["filter"] = filter_expr

        res = milvus_client.search(**kwargs)
        hits = res[0] if res else []

        results = []
        for h in hits:
            ent = h.get("entity", {})
            results.append({
                "score": h.get("distance"),
                "url": ent.get("url"),
                "chunk_number": ent.get("chunk_number"),
                "snippet": (ent.get("content") or "")
            })
        return results

    except Exception as e:
        print(f"[sparse keyword search] error: {e}", flush=True)
        return []


async def main():
    # Allow: python3 keyword_search_milvus.py "your query here"
    query = "415  Operating Systems prerequisites"
    print(f"Query: {query}", flush=True)
    results = await keyword_search_milvus(query, top_k=8)

    for r in results:
        print("------------------------------------------------------\n")
        print(f"{r['score']:.6f}  {r['url']}  {r['snippet']}", flush=True)
        print("\n------------------------------------------------------\n")

if __name__ == "__main__":
    asyncio.run(main())

