from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from documents import document_list

encoder = SentenceTransformer("all-MiniLM-L6-v2")

client = QdrantClient(":memory:")

# recreate_collection will first try to remove an existing collection with the same name.
# I think this is a function to instantiate a collection within a Qdrant client
client.recreate_collection(
    collection_name="my_books",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
        distance=models.Distance.COSINE,
    ),
)


# Tell the database to upload documents to the my_books collection. This will give each record an id and a payload. 
# The payload is just the metadata from the dataset.
# I think this is the function for uploading the data (knowledge base) as the basis for vector search
client.upload_points(
    collection_name="my_books",
    points=[
        models.PointStruct(
            id=idx, vector=encoder.encode(doc["description"]).tolist(), payload=doc
        )
        for idx, doc in enumerate(document_list)
    ],
)

# Ask the model a question. Here, we perform the vector search
hits = client.search(
    collection_name="my_books",
    query_vector=encoder.encode("alien invasion").tolist(),
    # Filter the most recent book from 2000s
    query_filter=models.Filter(
        must=[models.FieldCondition(key="year", range=models.Range(gte=2000))]
    ),
    limit=3,
)
for hit in hits:
    print(hit.payload, "score:", hit.score)
