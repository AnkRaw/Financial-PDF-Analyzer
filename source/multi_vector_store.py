# multi_vector_store.py

from typing import List, Tuple
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from config import TEXT_EMBEDDING_MODEL_NAME, TABLE_EMBEDDING_MODEL_NAME

class TextVectorStoreBuilder:
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=TEXT_EMBEDDING_MODEL_NAME
        )
        
    def get_retriever(self, vector_store: Chroma):
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "lambda_mult": 0.7}
        )
        return retriever

    def build_store_and_retriever(self, text_chunks: List, text_summaries: List[str]) -> Tuple[Chroma, any]:
        documents = []
        for chunk, summary in zip(text_chunks, text_summaries):
            doc = Document(
                page_content=chunk.text,
                metadata={
                    **chunk.metadata,
                    "summary": summary
                }
            )
            documents.append(doc)

        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )
        retriever = self.get_retriever(vector_store)
        return vector_store, retriever


class TableVectorStoreBuilder:
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=TABLE_EMBEDDING_MODEL_NAME
        )

    def get_retriever(self, vector_store: Chroma):
        retriever = vector_store.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 5, "lambda_mult": 0.8}
            )
        return retriever
        
    def build_store_and_retriever(self, table_chunks: List, table_summaries: List[str]) -> Tuple[Chroma, any]:
        documents = []
        for chunk, summary in zip(table_chunks, table_summaries):
            doc = Document(
                page_content=chunk.text,
                metadata={
                    **chunk.metadata,
                    "summary": summary
                }
            )
            documents.append(doc)

        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )
        retriever = self.get_retriever(vector_store)
        return vector_store, retriever
