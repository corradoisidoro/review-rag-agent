import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


def initialize_retriever(csv_path, db_location, model_name):
    embeddings = OllamaEmbeddings(model=model_name)

    add_documents = not os.path.exists(db_location)

    vector_store = Chroma(
        collection_name="restaurant_reviews",
        persist_directory=db_location,
        embedding_function=embeddings,
    )

    if add_documents:
        df = pd.read_csv(csv_path)
        documents = []
        ids = []

        for i, row in df.iterrows():
            document = Document(
                page_content=str(row["Title"]) + " " + str(row["Review"]),
                metadata={"rating": row["Rating"], "date": row["Date"]},
                id=str(i)
            )
            ids.append(str(i))
            documents.append(document)

        vector_store.add_documents(documents=documents, ids=ids)

    return vector_store.as_retriever(search_kwargs={"k": 5})
