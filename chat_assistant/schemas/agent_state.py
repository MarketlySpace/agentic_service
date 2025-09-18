from langgraph.graph import StateGraph


class AgentState(StateGraph):
    query: str
    result: str