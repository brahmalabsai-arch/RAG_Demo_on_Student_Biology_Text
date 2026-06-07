import os
import chromadb
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

generation_model_name = "llama-3.3-70b-versatile"

client = chromadb.PersistentClient(path="./chroma_biology")
collection = client.get_or_create_collection(name="biology_knowledge_base")


def retrieve_relevant_documents(query, top_k=3):
    # query_texts lets ChromaDB embed the query with the same default model
    # used during ingestion — keeps the vector spaces consistent
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return results


def generate_answer(query, retrieved_docs):
    context = "\n\n".join(retrieved_docs['documents'][0])
    system_prompt = (
        f"You are a biology tutor for class 7-8 students. "
        f"Answer the following question using only the context provided below in a single paragraph\n\n"
        f"Additionally, try to cite the page numbers from the chunks for reference"
        f"Assume that you are explaining to a 10 year old student. Avoid using excessive complicated jargons."
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        f"If the answer is not present in the context, say: "
        f"'This topic is not covered in the provided material.'"
    )
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = groq_client.chat.completions.create(
        model=generation_model_name,
        messages=[{"role": "user", "content": system_prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    for i in range(2):
        user_query = input(f"Ask your query {i+1}: ")
        retrieved_doc = retrieve_relevant_documents(user_query)
        answer = generate_answer(user_query, retrieved_doc)
        print(f"\nAnswer: {answer}\n")
