from pydantic import BaseModel

class AgentState(BaseModel):
    query: str
    result: str