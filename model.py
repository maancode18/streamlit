import os
from typing import Dict
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.tools import Tool


class SmartSummarizerPro:
    """
    Summarizes text from a given source (URL, file path, or raw text) using LangChain.
    """

    def run(self, source: Dict[str, str]) -> str:
        if not isinstance(source, dict) or len(source) != 1 or "source" not in source:
            raise ValueError("Input must be a dictionary with a single key: 'source'.")

        source_value = source["source"]
        print(f"Summarizing source: {source_value}")

        llm = Ollama(model="mistral",temperature=0)

        try:
            if source_value.startswith("http://") or source_value.startswith("https://"):
                loader = WebBaseLoader(source_value)
                docs = loader.load()

            elif os.path.exists(source_value):
                if source_value.lower().endswith(".pdf"):
                    loader = PyPDFLoader(source_value)
                    docs = loader.load_and_split()
                elif source_value.lower().endswith(".txt"):
                    loader = TextLoader(source_value, autodetect_encoding=True)
                    docs = loader.load_and_split()
                else:
                    raise ValueError("Unsupported file type. Must be .pdf or .txt")

            else:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.create_documents([source_value])

            if not docs:
                return "No text found to summarize."

            chain = load_summarize_chain(llm, chain_type="map_reduce")
            return chain.run(docs)

        except Exception as e:
            raise Exception(f"Summarization failed: {e}")


# Instantiate your summarizer logic
summarizer = SmartSummarizerPro()

# Wrap it using LangChain Tool
smart_summarizer_tool = Tool(
    name="smart_summarizer_pro",
    func=summarizer.run,
    description="Summarizes text from a URL, PDF, TXT file, or raw text string. Input: {'source': '...'}"
)

# Example usage (you can integrate this tool into an Agent)
if __name__ == "__main__":
    result = smart_summarizer_tool.run({"source": "https://www.bbc.com/news/world-us-canada-67822224"})
    print(f"\nSummary:\n{result}")
