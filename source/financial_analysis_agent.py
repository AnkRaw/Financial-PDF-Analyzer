from langchain_google_genai import GoogleGenerativeAI
from config import GEMINI_API_KEY, CHAT_MODEL_NAME

class FinancialAnalysisAgent:
    def __init__(self, text_retriever, table_retriever):
        self.text_retriever = text_retriever
        self.table_retriever = table_retriever
        self.llm = GoogleGenerativeAI(
            model=CHAT_MODEL_NAME,
            google_api_key=GEMINI_API_KEY
        )

    def generate_full_report(self):
        section_titles = {
        "Executive summary of financial performance": "## Executive Summary",
        "Key financial highlights and metrics": "## Key Highlights",
        "Management discussion and analysis": "## Management Discussion",
        "Overall financial health summary": "## Financial Health",
        "Revenue and profit trends summary": "## Revenue & Profit Trends",
        "Liquidity and solvency overview": "## Liquidity & Solvency",
        "Earnings highlights": "## Earnings Overview",
        "Summary of consolidated financial statements": "## Consolidated Statements",
    }

        final_report = ""
        for query, title in section_titles.items():
            context = self._retrieve_financial_sections([query])  # Use only that query
            # prompt = f"""
            # You are a financial analyst. Generate a markdown report section titled '{title}' based only on the context below.
            
            # Support with page numbers if available.

            # Context:
            # {context}
            # """
            prompt = f"""
                You are a financial analyst generating a structured markdown report section titled **'{title}'**.

                The input context was extracted from financial documents using an automated parser and is presented in a lightweight HTML-like format. Please note:

                ### Format Notes:
                - Text elements are wrapped in custom tags such as `<H>`, `<T>`, `<NT>`, `<TX>`, etc., followed by page numbers like `[P3]`. These indicate semantic roles (e.g., Header, Title, Narrative Text, etc.).
                - Table elements are provided as raw HTML strings representing the table structure.
                - Due to extraction limitations, formatting may be **incomplete, inconsistent, or noisy** (e.g., broken lines, misclassified text, malformed tables).
                - These tags and formats are only hints — **use them to guide interpretation**, not as absolute structure.

                ### Your Task:
                - Extract and summarize **only the relevant facts and figures** tied to the section title.
                - **Avoid speculation** or inclusion of boilerplate/general content.
                - Focus on:
                    - Quantitative financial data (e.g., revenue, net income, EPS, margins, debt)
                    - Key strategic actions (e.g., acquisitions, divestitures, expansions)
                - Reference page numbers (e.g., *(p. 12)*) where applicable.

                ### Output Format:
                - Markdown
                - Use clear headings and subheadings
                - Bullet points or short analytical paragraphs
                - Keep the tone concise, objective, and fact-driven

                Ignore formatting noise or irrelevant content. Prioritize clarity and accuracy in summarizing relevant financial insights.

                **Context:**
                {context}
                """
            response = self.llm.invoke(prompt)
            final_report += f"\n{response.strip()}\n\n"
            print(final_report)
        
        return final_report

    def _retrieve_financial_sections(self, queries):
        combined_context = ""

        for query in queries:
            # Get results from both retrievers
            text_docs = self.text_retriever.get_relevant_documents(query)
            table_docs = self.table_retriever.get_relevant_documents(query)            
            
            for doc in text_docs[:3] + table_docs[:3]:
                # Include metadata in the context
                meta_info = doc.metadata
                content_block = f"{meta_info}\n{doc.page_content.strip()}\n\n"
                combined_context += content_block
        return combined_context

    