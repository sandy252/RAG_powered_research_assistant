from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv
import re

load_dotenv()


def cleaner(page_contents):
    text = page_contents

    text = re.split(r'\b(references|bibliography)\b', text, flags=re.IGNORECASE)[0]
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\[[0-9,\s]+\]', ' ', text)
    text = re.sub(r'\([A-Za-z].*?\d{4}.*?\)', ' ', text)
    text = re.sub(r'(Figure|Fig\.|Table)\s*\d+.*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'Page\s*\d+', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\n\d+\n', ' ', text)
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = " ".join(text.split())

    return text
    

def load_paper(file):
    loader = PyPDFLoader(file)
    docs = loader.load()
    cleaned_documents = list()

    for doc in docs:
        raw_content = doc.page_content
        cleaned_content = cleaner(raw_content)
        new_document = Document(
            page_content = cleaned_content,
            metadata = doc.metadata
        )
        cleaned_documents.append(new_document)
        

    return cleaned_documents

docs = load_paper(r"D:\RAG_assistant\attention_paper.pdf")
print(docs[4].page_content)

"""
checking whether my git is connected to the right repository or not, and also checking whether the code is being pushed to the repository or not.
"""
