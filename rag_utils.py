from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import os

def load_documents(data_path="data/"):
    all_docs = []
    for file in os.listdir(data_path):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(data_path, file))
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            all_docs.extend(splitter.split_documents(documents))
    return all_docs

def setup_vectorstore(docs, persist_directory="chroma_db"):
    embedding = OllamaEmbeddings(model="llama3")
    db = Chroma.from_documents(docs, embedding, persist_directory=persist_directory)
    return db

def get_qa_chain():
    embedding = OllamaEmbeddings(model="llama3")
    db = Chroma(persist_directory="chroma_db", embedding_function=embedding)
    retriever = db.as_retriever()
    llm = Ollama(model="llama3")
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return chain
