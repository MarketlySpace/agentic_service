import os
from typing import Any
from langchain.tools import BaseTool
from langchain.utilities import GoogleSearchAPIWrapper
from src.global_config import (
    GOOUGLE_SEARCH_TOOL_NAME,
    GOOUGLE_SEARCH_TOOL_DESCRIPTION,
    NUM_SEARCH_RESULT,
)


class GoogleSearch(BaseTool):

    name = GOOUGLE_SEARCH_TOOL_NAME
    description = GOOUGLE_SEARCH_TOOL_DESCRIPTION

    def __init__(
            self,
            num_results: int=NUM_SEARCH_RESULT,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        google_cse_id = os.environ.get("GOOGLE_CSE_ID")

        if (
            not google_api_key or
            not google_cse_id
        ):
            raise ValueError("Google API Key and CSE ID must be set in environment variables")

        self.api_wrapper = GoogleSearchAPIWrapper(
            google_api_key=google_api_key,
            google_cse_id=google_cse_id,
            k=num_results
        )

    def _run(
            self,
            query: str,
            *args: Any,
            **kwargs: Any
    ) -> str:
        return self.api_wrapper.run(query)