

GOOUGLE_SEARCH_TOOL_NAME        = 'Google search'
GOOUGLE_SEARCH_TOOL_DESCRIPTION = "Synchronous tool for searching general information on the Internet and returning relevant results."
NUM_SEARCH_RESULT               = 1


TOOLS_AVAILABLE = [
    {
        "name": GOOUGLE_SEARCH_TOOL_NAME,
        "description": GOOUGLE_SEARCH_TOOL_DESCRIPTION,
    }
]

AVAILABLE_MODEL_URL    = "https://openrouter.ai/api/v1"
TEMPERATURE_ROUTER     = 0.4


VECTOR_DB_DIR        = 'src\db'
EMBEDDER_CHUNK_SIZE  = 3
RETRY_REQUEST_TIME   = 15
TIMEOUT              = 60
NAME_EMBEDDER_MODEL  = "text-embedding-3-small"
SPLITER_CHUNK_SIZE   = 512
SPLITER_OVERLAP      = 100
K_SAMPLES            = 3






