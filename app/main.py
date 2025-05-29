from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from Agent.agent2005 import *
from typing import Optional

app = FastAPI()

# Define a request body schema
class ChatRequest(BaseModel):
    message: Optional[str] = None
    chat_id: str

@app.get("/")
async def root():
    return {"message": "🤖 Agent CCI (événements + base vectorielle + mémoire longue) — prêt !"}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        response = await agent_response(req.message, chat_id=req.chat_id)
        return JSONResponse(content={"reply": response})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.post("/surveillance_inactivite")
async def surveillance(req: ChatRequest):
    try:
        response = surveillance_inactivite(req.chat_id)
        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)