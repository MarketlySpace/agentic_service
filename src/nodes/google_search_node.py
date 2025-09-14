from typing import Dict
from src.tools.google_search import GoogleSearch
from src.schemas.agent_state import AgentState


class GoogleNode:

    def __init__(self):
        self.tool = GoogleSearch()

    def __call__(self, state: AgentState) -> Dict[str, str]:
        return {"result": self.tool.run(state.query)}