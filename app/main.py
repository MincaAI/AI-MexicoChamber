from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from Agent.agent2005 import *
from typing import Optional
from app.service.chat.storeMessageWithChatId import store_message_and_reply
import traceback

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
        # 1. Get agent response
        response = await agent_response(req.message, chat_id=req.chat_id)

        # 2. Store both user message and agent reply
        await store_message_and_reply(chat_id=req.chat_id, message=req.message, reply=response)

        # 3. Return response
        return JSONResponse(content={"reply": response})

    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": tb
            }
        )
    
@app.post("/surveillance_inactivite")
async def surveillance(req: ChatRequest):
    try:
        response = await surveillance_inactivite(req.chat_id)
        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)