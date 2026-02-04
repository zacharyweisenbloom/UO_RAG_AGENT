from pymilvus import MilvusClient
#setup
METRIC = "IP"
COLLECTION = "webpages"
client = MilvusClient("milvus_demo.db")

if client.has_collection(collection_name=COLLECTION):
    client.drop_collection(collection_name=COLLECTION)

if not client.has_collection(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        dimension=5,
        metric_type=METRIC,
        auto_id=AUTO_ID,
    )
    # optional but recommended
    client.create_index(
        collection_name=COLLECTION,
        field_name="vector",
        index_params={"index_type": "AUTOINDEX", "metric_type": METRIC},
    )#insert

res = client.insert(
  collection_name=COLLECTION,
  data=[
    {"id": 0, "vector": [0.1, 0.2, 0.3, 0.2, 0.1], "text": "AI was proposed in 1956.", "subject": "history"},
    {"id": 1, "vector": [0.1, 0.2, 0.3, 0.2, 0.1], "text": "Alan Turing was born in London.", "subject": "history"},
    {"id": 2, "vector": [0.1, 0.2, 0.3, 0.2, 0.1], "text": "Computational synthesis with AI algorithms predicts molecular properties.", "subject": "biology"},
  ]
)

#Search
res = client.search(
  collection_name=COLLECTION,
  data=[[0.1, 0.2, 0.3, 0.2, 0.1]],
  filter="subject == 'history'",
  limit=2,
  output_fields=["text", "subject"],
)
print(res)

res = client.delete(
  collection_name=COLLECTION,
  ids=[0,2]
)
