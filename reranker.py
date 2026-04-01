from sentence_transformers import CrossEncoder
from retreiver import vectorstore

class CrossEncoderReranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        print(f"Loading cross-encoder: {model_name}...")
        self.model = CrossEncoder(model_name)
        print("Re-ranker ready")
    
    def rerank(self, query, documents, top_k=3):
        """Re-rank documents using cross-encoder"""
        # Create (query, doc) pairs
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Score with cross-encoder
        scores = self.model.predict(pairs)
        
        # Sort by score
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked[:top_k]
    
# Initialize re-ranker
reranker = CrossEncoderReranker()
# Compare before/after re-ranking
query = "Explain how attention mechanisms work in transformers"
# Get initial candidates (more documents)
initial_docs = vectorstore.similarity_search(query, k=10)
print("Before Re-ranking (top 3):")
for i, doc in enumerate(initial_docs[:3], 1):
    print(f"  [{i}] {doc.page_content[:100]}...")
# Re-rank
reranked = reranker.rerank(query, initial_docs, top_k=3)
print("\nAfter Re-ranking (top 3):")
for i, (doc, score) in enumerate(reranked, 1):
    print(f"  [{i}] Score: {score:.4f}")
    print(f"      {doc.page_content[:100]}...")