from pymilvus import MilvusClient

client = MilvusClient("milvus_demo.db")

rows = client.query(
    collection_name="webpages",
    # omit filter or use a trivially true one:
    filter="chunk_number >= 0",
    output_fields=["id", "url", "chunk_number", "vector"],
    limit=5
)

for r in rows:
    v = r.get("vector")
    print({
        "id": r["id"],
        "url": r.get("url"),
        "chunk": r.get("chunk_number"),
        "vec_len": (len(v) if isinstance(v, (list, tuple)) else None),
        "vec_preview": (v[:6] if isinstance(v, (list, tuple)) else None),
    })

