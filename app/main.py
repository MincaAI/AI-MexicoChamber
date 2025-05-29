from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from Agent.agent2005 import *



app = FastAPI()

# Define a request body schema
class ChatRequest(BaseModel):
    message: str
    chat_id: str

@app.get("/")
async def root():
    return {"message": "🤖 Agent CCI (événements + base vectorielle + mémoire longue) — prêt !"}

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        response = agent_response(req.message, chat_id=req.chat_id)
        return JSONResponse(content={"reply": response})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
class ChatRequest(BaseModel):
    chat_id: str

@app.post("/surveillance-inactivite")
def surveillance_inactivite_api(request: ChatRequest):
    try:
        history = get_full_conversation(request.chat_id)
        if has_calendly_link(history):
            lead = extract_lead_info(history)
            if lead.get("prenom") != "inconnu" and lead.get("email") != "inconnu":
                store_lead_to_google_sheet(lead)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))