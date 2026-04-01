from langchain_community.vectorstores import Chroma
from ingestion import embeddings
from loader import chunks

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="ai_knowledge_base"
)
# Search for similar documents
