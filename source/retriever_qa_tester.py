from langchain_core.runnables import RunnableMap
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from config import GEMINI_API_KEY, CHAT_MODEL_NAME


class RetrieverQATester:
    def __init__(self, text_retriever, table_retriever):
        self.text_retriever = text_retriever
        self.table_retriever = table_retriever
        self.llm = GoogleGenerativeAI(
            model=CHAT_MODEL_NAME,
            google_api_key=GEMINI_API_KEY
        )

        self.prompt = PromptTemplate.from_template("""
        You are a financial question answering assistant.
        Use the context below to answer the question accurately.
        If applicable, include references such as page numbers or section types.

        Context:
        {context}

        Question: {question}

        Answer:
        """)

        self.chain = (
            RunnableMap({
                "context": self._combine_retrievers,
                "question": lambda x: x["question"]
            }) |
            self.prompt |
            self.llm
        )

    def _combine_retrievers(self, inputs):
        query = inputs["question"]
        text_docs = self.text_retriever.get_relevant_documents(query)
        table_docs = self.table_retriever.get_relevant_documents(query)

        all_docs = text_docs + table_docs
        formatted_context = []

        for doc in all_docs:
            metadata = doc.metadata
            formatted_context.append(
                f"{metadata}\n{doc.page_content.strip()}\n\n"
            )

        return "\n\n".join(formatted_context)

    def ask(self, question: str):
        return self.chain.invoke({"question": question}).strip()
