from fastapi import FastAPI
import uvicorn
from chat_assistant.api.routes.rag_router import router as rag_router

app = FastAPI(
    title="Agentic service",
)

@app.get("/")
async def root():
    return {"message": "Hello world!"}



if __name__ == "__main__":
    app.include_router(rag_router)

    uvicorn.run("main:app", host="0.0.0.0", port=8100, reload=True)