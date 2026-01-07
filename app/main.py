from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about pizza restaurants. 
Use the following reviews to answer the question. If the answer isn't in the reviews, 
honestly state that you don't know.
Relevant reviews: {reviews}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

print("RAG Agent initialized. Database ready.")

while True:
    print("\n" + "-"*30)

    # question = input("Ask your question (q to quit): ")
    question = input("Ask your question (q to quit): ").strip()

    if question.lower() == "q":
        break

    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)
