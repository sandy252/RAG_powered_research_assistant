from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def cleaner(contents):
    corpus = " ".join(contents)
    pass


def load_paper(file):
    loader = PyPDFLoader(file)
    docs = loader.load()
    contents = list()
    for doc in docs:
        contents.append(doc.page_content)
    cleaned_contents = cleaner(contents)

docs = load_paper(r"D:\RAG_assistant\attention_paper.pdf")
