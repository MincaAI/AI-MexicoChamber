import os
import sys
import time
import threading
import asyncio
from datetime import datetime, timezone
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from pinecone import Pinecone
from Agent.storage import store_lead_to_google_sheet
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
from app.service.chat.getAllChat import get_full_conversation_postgre

import redis
import pickle

load_dotenv()

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=int(os.getenv("REDIS_PORT")),
    username=os.getenv("REDIS_USERNAME"),
    password=os.getenv("REDIS_PASSWORD"),
    ssl=os.getenv("REDIS_USE_SSL", "false").lower() == "true"
)

class StreamPrintCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)

# === 1. Charger les variables d'environnement ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# === 2. Initialisation ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
llm = ChatOpenAI(
    temperature=0.6,
    model="gpt-4o",
    streaming=True,
    callbacks=[StreamPrintCallback()]
)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
retriever = PineconeVectorStore(
    index=index,
    embedding=embeddings
).as_retriever(search_kwargs={"k": 2})
summary_store = PineconeVectorStore(index=index, embedding=embeddings,namespace="summaries")


def save_or_update_summary(chat_id: str, summary: str):
    doc = Document(
        page_content=summary,
        metadata={
            "chat_id": chat_id,
            "summary_id": f"summary_{chat_id}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )
    # On remplace l'ancien résumé
    summary_store.delete(ids=[f"summary_{chat_id}"])
    summary_store.add_documents([doc], ids=[f"summary_{chat_id}"])

def load_summary_from_pinecone(chat_id: str) -> str:
    docs = summary_store.similarity_search("", k=1, filter={"chat_id": chat_id, "summary_id": f"summary_{chat_id}"})
    return docs[0].page_content if docs else ""

def generate_summary_from_memory(memory: ConversationSummaryBufferMemory, chat_id: str, max_messages: int = 15) -> str:
    new_messages = memory.chat_memory.messages[-max_messages:]
    convo_text = "\n".join([f"{msg.type.capitalize()} : {msg.content}" for msg in new_messages])

    if not convo_text.strip():
        return ""

    old_summary = load_summary_from_pinecone(chat_id)

    prompt = f"""
    Voici le résumé actuel d’une conversation WhatsApp entre toi et l'utilisateur  :
    {old_summary or '[aucun résumé pour le moment]'}

    Voici les nouveaux messages à intégrer :
    {convo_text}

    Génére un **nouveau résumé synthétique mis à jour**, qui conserve les éléments utiles de l’ancien et ajoute les nouvelles informations importantes. Résume en 7 phrases maximum.
    """

    summary = llm.invoke(prompt).content.strip()
    save_or_update_summary(chat_id, summary)
    return summary

def save_summary_memory_to_redis(chat_id, memory):
    state = {
        "messages": memory.chat_memory.messages,
        "summary": memory.moving_summary_buffer
    }
    redis_client.set(chat_id, pickle.dumps(state))

def load_summary_memory_from_redis(chat_id):
    data = redis_client.get(chat_id)
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=800,
        return_messages=True,
        memory_key="chat_history"
    )
    if data:
        state = pickle.loads(data)
        memory.chat_memory.messages = state.get("messages", [])
        memory.moving_summary_buffer = state.get("summary", "")
    return memory


def load_prompt_template():
    with open("prompt_base.txt", encoding="utf-8") as f:
        return f.read().strip()
    
def load_extraction_prompt_template():
    with open("prompt_extraction.txt", encoding="utf-8") as f:
        return f.read().strip()
    

def has_calendly_link(text: str) -> bool:
    return "https://calendly.com/" in text


def extract_lead_info(history: str) -> dict:
    template = load_extraction_prompt_template()
    prompt = template.replace("{{history}}", history)

    try:
        response = llm.invoke(prompt).content
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1]
        if response.endswith("```"):
            response = response.rsplit("\n", 1)[0]
        response = response.strip()

        import json
        data = json.loads(response)
        data["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return data
    except Exception as e:
        return {
            "prenom": "inconnu",
            "nom": "inconnu",
            "entreprise": "inconnu",
            "email": "inconnu",
            "interet": "inconnu",
            "score": 1,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d")
        }
        
def get_and_increment_counter(chat_id: str) -> int:
    key = f"msg_counter:{chat_id}"
    counter = redis_client.incr(key, 1)
    redis_client.expire(key, 86400)  # 24h
    return counter

prompt_template = load_prompt_template()

async def agent_response(user_input: str, chat_id: str) -> str:
    today = datetime.now().strftime("%d %B %Y")
    memory = load_summary_memory_from_redis(chat_id)
    short_term_memory = "\n".join([f"{msg.type.capitalize()} : {msg.content}" for msg in memory.buffer])
    long_term_memory = load_summary_from_pinecone(chat_id)
    

    base_cci_context_docs = retriever.invoke(user_input)
    base_cci_context = "\n\n".join(doc.page_content for doc in base_cci_context_docs) if base_cci_context_docs else "[Pas d'information pertinente dans la base.]"
    

    prompt = prompt_template.replace("{{today}}", today)\
                            .replace("{{user_input}}", user_input)\
                            .replace("{{short_term_memory}}", short_term_memory or "[Aucune mémoire courte]")\
                            .replace("{{long_term_memory}}", long_term_memory or "[Aucune mémoire longue]")\
                            .replace("{{cci_context}}", base_cci_context)

    reply = await llm.ainvoke(prompt)
    reply_text = reply.content if hasattr(reply, "content") else str(reply)
    print(reply_text)  # Affiche la réponse de l'agent dans la console

    memory.chat_memory.add_user_message(user_input)
    memory.chat_memory.add_ai_message(reply_text)
    save_summary_memory_to_redis(chat_id, memory)

    # Incrément du compteur et résumé auto tous les 30 messages
    counter = get_and_increment_counter(chat_id)
    if counter >= 15:
        redis_client.delete(f"msg_counter:{chat_id}")
        generate_summary_from_memory(memory, chat_id)

    print(reply_text)  # Affiche la réponse de l'agent dans la console
    return reply_text

    
async def surveillance_inactivite(chat_id: str):
    try:
        result = await get_full_conversation_postgre(chat_id)
        history = result.get("history", "")
        chatType = result.get("type", "undefined")

        if has_calendly_link(history):
            lead = extract_lead_info(history)
            store_lead_to_google_sheet(lead, type=chatType)

        return {"status": "success"}

    except Exception as e:
        return {"status": "error", "message": str(e)}
# if __name__ == "__main__":
#     import uuid
#     chat_id = str(uuid.uuid4())  # 👈 Generate a unique ID per session
#     print(f"🆔 Session Chat ID: {chat_id}")
#     threading.Thread(target=surveillance_inactivite, args=(chat_id,), daemon=True).start()

#     try:
#         while True:
#             user_input = input("💬 Vous : ")
#             inactivity_event.set()
#             reply = asyncio.run(agent_response(user_input, chat_id=chat_id))
#             print(f"\n🧠 Agent :\n{reply}\n")
#     except KeyboardInterrupt:
#         pass

