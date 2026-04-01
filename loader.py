from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from pypdf import PdfReader

# Sample documents about AI
documents = [
    Document(page_content="""
    Large Language Models (LLMs) are AI systems trained on vast amounts 
    of text data. They use transformer architectures with self-attention 
    mechanisms to understand and generate human language. Models like 
    GPT-4, Claude, and LLaMA have billions of parameters...
    """, metadata={"source": "ai_intro", "topic": "llms"}),
    # ... more documents
]
def load_pdf(path):
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return pages

pdf_text = load_pdf("data/Thinking_Like_Transformers.pdf")
docs = [Document(page_content=page) for page in pdf_text]

# Smart chunking with overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # ~125 tokens
    chunk_overlap=100,    # Preserve context between chunks
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks from {len(docs)} documents")

