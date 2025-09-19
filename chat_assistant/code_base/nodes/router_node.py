import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from chat_assistant.code_base.schemas.agent_state import AgentState
from chat_assistant.global_config import TOOLS_AVAILABLE, AVAILABLE_MODEL_URL, TEMPERATURE_ROUTER, GOOUGLE_SEARCH_TOOL_NAME


class RouterNode:

    def __init__(self, model="openrouter/auto") -> None:
        os.environ['OPEN_API_KEY'] = os.getenv("OPENROUTER_API_KEY")
        os.environ['OPEN_API_BASE'] = AVAILABLE_MODEL_URL
        self.llm = ChatOpenAI(
            model=model,
            temperature=TEMPERATURE_ROUTER,
        )

    def __call__(self, state: AgentState):
        tool_list = "\n".join([f"{t['name']}: {t['description']}" for t in TOOLS_AVAILABLE])

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a smart router. Choose the best tools for the query."),
            HumanMessage(content=f"""
        User query: {state.query}
        Available tools:
        {tool_list}
        
        Return a JSON with a list of tools to use, e.g.:
        {{"tools": ["{GOOUGLE_SEARCH_TOOL_NAME}", ]}}
        """)
        ])
        messages = prompt_template.format_prompt().to_messages()
        decision = self.llm(messages)
        try:
            data = json.loads(decision[0].content)
            tools = data.get("tools", [])
        except Exception:
            tools = []
        return state, tools
