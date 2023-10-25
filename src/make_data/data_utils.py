from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os

EMBEDDINGS = HuggingFaceEmbeddings()

def get_data():
    current_path = os.path.dirname(__file__)
    data_path = os.path.join(current_path, 'raw_data','ResultsTest.txt')
    loader = TextLoader(data_path)
    docs = loader.load()

    return docs


def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2024,
        chunk_overlap=64,
        # separators=['\n\n', '\n', '(?=>\. )', ' ', '']
    )
    split_docs = text_splitter.split_documents(docs)
    
    return split_docs


def load_vectorstore():
    docs = get_data()
    split_docs = split_text(docs)
    vdb = FAISS.from_documents(split_docs, EMBEDDINGS)

    return vdb



    