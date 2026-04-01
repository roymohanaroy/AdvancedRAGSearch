from rank_bm25 import BM25Okapi
from retreiver import vectorstore
from loader import chunks
import numpy as np

class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        # Tokenize documents (simple word splitting)
        self.tokenized_docs = [
            doc.page_content.lower().split() 
            for doc in documents
        ]
        # Create BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print(f"BM25 index created with {len(documents)} documents")
    
    def search(self, query, k=3):
        """Search and return top k documents"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_k_idx = np.argsort(scores)[::-1][:k]
        
        return [(self.documents[i], scores[i]) for i in top_k_idx]
# Create BM25 retriever
bm25_retriever = BM25Retriever(chunks)

# Compare with vector search
query = "transformers attention mechanism"
print("Vector Search:")

vector_results = vectorstore.similarity_search(query, k=3)
for doc in vector_results:
    print(f"  - {doc.page_content[:80]}...")
print("\n BM25 Search:")

bm25_results = bm25_retriever.search(query, k=3)
for doc, score in bm25_results:
    print(f"  - [Score: {score:.2f}] {doc.page_content[:80]}...")