from pydantic import BaseModel


class Source(BaseModel):
    source_type: str
    path_or_url: str
