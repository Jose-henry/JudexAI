from langchain_core.tools import tool


@tool
def google_search_tool():
    """
    saves a note to a local file

    Args:
        note: the text note to save
    """
