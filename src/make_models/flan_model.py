from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
import os

REPO_ID = os.getenv('REPO_ID')
TEMPERATURE = os.getenv('TEMPERATURE')
MAX_LENGTH = os.getenv('MAX_LENGTH')
CHAIN_TYPE = os.getenv('CHAIN_TYPE')
K = os.getenv('K')
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# show how to use dotenv too

def get_llm(vdb):

    llm = HuggingFaceHub(repo_id=REPO_ID, 
                         model_kwargs={
                             "temperature":TEMPERATURE, 
                             "max_length":MAX_LENGTH
                             }
                        )
    chain = load_qa_chain(llm, chain_type=CHAIN_TYPE)
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever = vdb.as_retriever(search_kwargs={"k": K})
        )

    return qa
