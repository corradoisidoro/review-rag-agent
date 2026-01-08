import os
import logging
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import initialize_retriever
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

db_path = os.getenv("DB_PATH")
data_path = os.getenv("DATA_PATH")
embed_model = os.getenv("EMBED_MODEL")
llm_model = os.getenv("LLM_MODEL")

if not os.path.exists(db_path):
    logger.info("Database not found. Initializing from CSV...")
else:
    logger.info("Loading existing database...")

# Initialize
retriever = initialize_retriever(data_path, db_path, embed_model)
model = OllamaLLM(model=llm_model)

template = """
You are an expert in answering questions about pizza restaurants. 
Use the following reviews to answer the question.
Relevant reviews: {reviews}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

if __name__ == "__main__":
    print("\n🚀 RAG Agent Ready.")
    while True:
        question = input("\nAsk your question (q to quit): ").strip()
        if question.lower() == "q":
            break

        logger.info(f"Querying vector store for: '{question}'")
        reviews = retriever.invoke(question)
        print(chain.invoke({"reviews": reviews, "question": question}))
