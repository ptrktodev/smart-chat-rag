from langchain_community.document_loaders import PyPDFLoader
from io import BytesIO
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
import tempfile
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore  
import random
import streamlit as st
import string

emb_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=st.secrets["LLM"]["OPENAI_API_KEY"])

def split_text(document_input):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(document_input.read())
        tmp_path = tmp_file.name
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    enc = tiktoken.get_encoding("cl100k_base")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, # limite de tokens por chunk
        chunk_overlap=50,  # overlap entre chunks
        length_function=lambda x: len(enc.encode(x)) # contar tokens
    )
    chunks = text_splitter.split_documents(pages)
    return chunks

def create_collection(chunks, name):
    qdrant = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=emb_model,
        collection_name=name,
        api_key=st.secrets["QDRANT"]["QDRANT_API_KEY"],
        url=st.secrets["QDRANT"]["QDRANT_API_URL"],
        prefer_grpc=True # protocolo de comunicação
    )

def gerar_string_aleatoria(tamanho=10):
    caracteres = string.ascii_letters + string.digits
    return ''.join(random.choice(caracteres) for _ in range(tamanho))

def connect_to_collection(name):
    server = QdrantVectorStore.from_existing_collection(
        collection_name=name,
        api_key=st.secrets["QDRANT"]["QDRANT_API_KEY"],
        url=st.secrets["QDRANT"]["QDRANT_API_URL"],
        embedding=emb_model,
    )
    return server