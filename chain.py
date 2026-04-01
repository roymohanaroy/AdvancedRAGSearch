from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from retreiver import vectorstore

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=256,
    
)
llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"max_length": 256, "do_sample": False} )
# Create custom prompt
prompt_template = """Use the following context to answer the question. 
If you don't know, say you don't know.
Context: {context}
Question: {question}
"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


# Build the chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" = concatenate all docs
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

result = qa_chain.invoke({"query": "What are LLMs?"})    
for i, doc in enumerate(result, 1):
    print(f"\n[{i}] {doc}...")

# Ask questions!

print(f"Answer--------------------------------: {result['result']}")
print(f"Sources: {len(result['source_documents'])} documents")