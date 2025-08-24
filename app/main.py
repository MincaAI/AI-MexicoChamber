import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from typing import Optional
import traceback
import httpx

# Import agent seulement si les variables d'environnement sont définies
AGENT_ERROR = None
try:
    from Agent.agent2005 import *
    from app.service.chat.storeMessageWithChatId import store_message_and_reply
    AGENT_AVAILABLE = True
except Exception as e:
    print(f"Warning: Agent not available: {e}")
    AGENT_AVAILABLE = False
    AGENT_ERROR = str(e)

app = FastAPI()

# Define a request body schema
class ChatRequest(BaseModel):
    message: Optional[str] = None
    chat_id: str

@app.get("/")
async def root():
    response = {
        "message": "🤖 Agent CCI (événements + base vectorielle + mémoire longue) — prêt !",
        "status": "healthy",
        "agent_available": AGENT_AVAILABLE,
        "timestamp": "2025-01-22"
    }
    if AGENT_ERROR:
        response["agent_error"] = AGENT_ERROR
    return response

@app.get("/health")
async def health_check():
    """Endpoint de santé pour AWS App Runner."""
    from datetime import datetime
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "whatsapp-agent-cci"
    }

@app.post("/chat")
async def chat(req: ChatRequest):
    if not AGENT_AVAILABLE:
        return JSONResponse(
            status_code=503, 
            content={"error": "Agent service temporarily unavailable. Please check environment variables."}
        )
    
    try:
        # 1. Get chatId from phone number via middle API
        async with httpx.AsyncClient() as client:
            chatid_response = await client.post(
                "https://api.mincaai-franciamexico.com/chat/getChatIdFromPhone",
                json={"phoneNumber": req.chat_id}  # assuming req.chat_id is a phone number
            )
            chatid_response.raise_for_status()
            chatid = chatid_response.json().get("chatId")

        if not chatid:
            return JSONResponse(status_code=404, content={"error": "Unable to retrieve chatId from phone number"})

        # 2. Get agent response using the resolved chatid
        response = await agent_response(req.message, chat_id=chatid)

        # 3. Store messages in DB
        await store_message_and_reply(chat_id=chatid, message=req.message, reply=response)

        # 4. Return reply
        return JSONResponse(content={"reply": response})

    except httpx.HTTPStatusError as e:
        return JSONResponse(
            status_code=e.response.status_code,
            content={"error": "Failed to retrieve chatId", "details": e.response.text}
        )

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
    if not AGENT_AVAILABLE:
        return JSONResponse(
            status_code=503, 
            content={"error": "Agent service temporarily unavailable. Please check environment variables."}
        )
    
    try:
        response = await surveillance_inactivite(req.chat_id)
        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)