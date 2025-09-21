from pydantic import BaseModel, confloat, conint

from chat_assistant.global_config import (
    LOWER_BOUND_TEMPERATURE,
    UPPER_BOUND_TEMPERATURE,
    TEMPERATURE,
    K_SAMPLES,
    LOWER_BOUND_K_SAMPLES,
    UPPER_BOUND_K_SAMPLES,
)


class QueryRequest(BaseModel):
    query: str
    k: conint(
        ge=LOWER_BOUND_K_SAMPLES,
        le=UPPER_BOUND_K_SAMPLES
    ) = K_SAMPLES
    temperature: confloat(
        ge=LOWER_BOUND_TEMPERATURE,
        le=UPPER_BOUND_TEMPERATURE
    ) = TEMPERATURE