from fastapi import FastAPI
import uvicorn
app = FastAPI(
    title="Agentic service",
)

@app.get("/")
async def root():
    return {"message": "Hello world!"}



if __name__ == "__main__":

    uvicorn.run("main:app", host="0.0.0.0", port=8100, reload=True)