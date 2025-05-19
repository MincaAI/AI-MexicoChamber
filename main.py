from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from Agent.agent2 import agent_response


app = FastAPI()

# Define a request body schema
class ChatRequest(BaseModel):
    message: str
    user_id: str

@app.get("/")
async def root():
    return {"message": "🤖 Agent CCI (événements + base vectorielle + mémoire longue) — prêt !"}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        response = agent_response(req.message, user_id=req.user_id)
        return JSONResponse(content={"reply": response})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)