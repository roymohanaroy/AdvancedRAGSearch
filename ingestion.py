from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",  # 384 dimensions
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
# Test it
test_embedding = embeddings.embed_query("What is artificial intelligence?")
print(f"Embedding dimension: {len(test_embedding)}")  # 384con