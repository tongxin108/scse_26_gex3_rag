from langchain_core.tools import tool
from rag import retrieve


@tool
def search_policy(query: str) -> str:
    """Search the university IT policy knowledge base for relevant content.

    Args:
        query: The user's question.

    Returns:
        Relevant policy text, or a message saying nothing was found.
    """
    docs = retrieve(query, k=2)
    if not docs:
        return "No relevant information found in the knowledge base."
    return "\n\n".join(d.page_content for d in docs)