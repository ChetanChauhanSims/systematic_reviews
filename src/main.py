from src.make_data import load_vectorstore
from src.make_models import get_llm

print("loading vector database")
vdb = load_vectorstore()

print("loading LLM")
qa = get_llm(vdb)

print("running QA with LLM")
query = """
How many papers were included in the systematic review of this study?
"""
print(query)
print(qa.run(query))
