# day_26.py
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import FakeEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import FakeLLM

# Sample document
text = """
Artificial Intelligence (AI) is a wonderful field.
Machine Learning (ML) is a subset of AI.
Deep Learning uses neural networks.
"""

# Split text
splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)
texts = splitter.split_text(text)

# Create vector store (using fake embeddings for demo)
embeddings = FakeEmbeddings(size=50)
vectorstore = FAISS.from_texts(texts, embeddings)

# Fake LLM (in real: use OpenAI or HuggingFace)
llm = FakeLLM()

# Create RAG chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Ask question
query = "What is Machine Learning?"
result = qa.run(query)
print("Query:", query)
print("Result:", result)