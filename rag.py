from policy_loader import load_policy_documents
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

documents = load_policy_documents()

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name="university_support",
    embedding_function=embeddings,
)

batch_size = 100
for i in range(0, len(documents), batch_size):
    chunk = documents[i : i + batch_size]
    vector_store.add_documents(chunk)
    print(f"already add {i + len(chunk)} / {len(documents)} ")


def retrieve(query: str, k: int = 2):
    return vector_store.similarity_search(query, k=k)