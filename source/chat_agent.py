# agents/chat_agent.py

import getpass
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from config import GEMINI_API_KEY


class PDFChatAgent:
    def __init__(self, retriever):
        print("[INFO] Initializing PDF Retriever...")
        self.retriever = retriever

        print("[INFO] Initializing Gemini LLM (Google AI Studio)...")
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-1.5-pro-002",
            google_api_key=GEMINI_API_KEY,
            temperature=0,
        )

        # Define the tool using a closure to capture self
        @tool
        def search_pdf(query: str) -> str:
            """Searches the financial PDFs for relevant content based on the user query."""
            try:
                result = self.retriever.query(query)
                documents = result.get("documents", [[]])[0]
                if not documents:
                    return "No relevant information found in the financial documents."
                return "\n\n".join(documents[:3])  # Return top 3 results
            except Exception as e:
                return f"Error during document retrieval: {str(e)}"

        self.tools = [search_pdf]

        print("[INFO] Initializing LangChain Agent...")
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
        )

    def chat(self):
        """Starts an interactive command-line chat with the PDF analysis agent."""
        print("\n=== Interactive Financial Document Chat ===")
        print("Ask any question related to the ingested PDFs. Type 'exit' to quit.\n")

        while True:
            query = input("You: ")
            if query.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            try:
                response = self.agent_executor.run(query)
                print(f"\nAgent: {response}\n")
            except Exception as e:
                print(f"Error: {e}\n")


# Optional CLI execution
if __name__ == "__main__":
    agent = PDFChatAgent()
    agent.chat()
