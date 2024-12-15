from langchain_core.tools import tool


@tool
def rag_tool():
    """
    saves a note to a local file

    Args:
        note: the text note to save
    """
    return 