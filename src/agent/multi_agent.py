from src.nodes.router_node import RouterNode
from src.nodes.google_search_node import GoogleSearch
from src.global_config import TOOLS_AVAILABLE
from src.schemas.agent_state import AgentState



class MultiAgent:

    def __init__(self) -> None:

        self.router = RouterNode()
        self.nodes = {t['name']: t['class']() for t in TOOLS_AVAILABLE}

    async def run(self, query: str):
        state = AgentState(query=query)
        state, selected_tools = await self.router(state)
        for tool_name in selected_tools:
            node = self.nodes.get(tool_name)
            if node:
                state.result = await node.run(state)
        return state.result
