# summary_generator.py

import time
from typing import List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from config import GEMINI_API_KEY, SUMMARY_MODEL_NAME


class Chunk(BaseModel):
    type: str
    text: str
    metadata: dict = {}


class SummaryGenerator:
    def __init__(self, api_key: str = GEMINI_API_KEY, model_name: str = SUMMARY_MODEL_NAME, sleep_interval: int = 5):
        self.sleep_interval = sleep_interval

        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0,
            timeout=30
        )

        self.prompt = ChatPromptTemplate.from_template(
        """You are an intelligent assistant tasked with summarizing structured document content.

        The input consists of either:
        1. **Textual chunks** with custom HTML-like tags:
        - <H> = Header
        - <T> = Title
        - <NT> = Narrative Text
        - <TX> = Text
        - <LI> = List Item
        - <IM> = Image
        - <FC> = Figure Caption
        - <F> = Formula  
        Each element ends with its source page number, e.g., [P1], [P2].

        2. **Tables** provided as raw HTML strings.

        Please note:  
        - The structure may be **imperfect or partially extracted**, as the content was processed by an automated extractor.  
        - Your goal is to interpret the content **flexibly**, focusing on the main ideas, key data points, and general meaning, rather than exact formatting.

        Your tasks:
        - For **text chunks**: Identify and summarize the key points, grouping related information logically.
        - For **tables**: Summarize the most important insights, trends, or data relationships the table is trying to convey.

        Chunk:
        {element}
        """
        )


        self.summarize_chain = {"element": lambda x: x} | self.prompt | self.model | StrOutputParser()

    def summarize_chunks(self, chunks: List[Chunk], label: str = "element") -> List[str]:
        summaries = []
        for i, chunk in enumerate(chunks):
            print(f"Summarizing {label} #{i}")
            try:
                summary = self.summarize_chain.invoke({"element": chunk.text})
                summaries.append(summary)
            except Exception as e:
                print(f"Error summarizing {label} #{i}: {e}")
                summaries.append("")
            time.sleep(self.sleep_interval)
        return summaries
