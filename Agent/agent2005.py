import os
import sys
import time
import threading
import asyncio
from datetime import datetime, timezone
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from pinecone import Pinecone
from Agent.Lead_extraction.storage import *
from Agent.Lead_extraction.extraction import *
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import RedisChatMessageHistory
from app.service.chat.getAllChat import get_full_conversation_postgre
import redis
from app.service.chat.store_calendy_link import *


load_dotenv()

redis_url = os.getenv("REDIS_URL")
redis_client = redis.Redis.from_url(redis_url)

inactivity_event = threading.Event()

class StreamPrintCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)

# === 1. Charger les variables d'environnement ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# === 2. Initialisation ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Configuration Claude
#llm = ChatAnthropic(
  #  temperature=0.6,
   # model="claude-3-5-sonnet-20241022",
    #streaming=False,
    #anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    #callbacks=[StreamPrintCallback()]
#)

llm = ChatOpenAI(
    temperature=0.2,
    model="gpt-4o",
    streaming=True,
    max_tokens=350,
    callbacks=[StreamPrintCallback()]
)

# === 3. Pinecone ===

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
retriever = PineconeVectorStore(
    index=index,
    embedding=embeddings
).as_retriever(search_kwargs={"k": 2})
#new
#summary_store = PineconeVectorStore(index=index, embedding=embeddings,namespace="summaries")  # new


def get_memory(chat_id: str) -> ConversationSummaryBufferMemory:
    redis_url = os.getenv("REDIS_URL")
    history = RedisChatMessageHistory(
        session_id=f"chat:{chat_id}",
        url=redis_url,
        key_prefix="message_store" 
    )

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        memory_key="chat_history",
        chat_memory=history,
        return_messages=True,
        max_token_limit=2000
    )
    return memory


def load_prompt_template():
    with open("prompt_base.txt", encoding="utf-8") as f:
        return f.read().strip()


async def get_user_profile_summary(chat_id: str, messages: list) -> str:
    redis_key = f"profile_summary:{chat_id}"
    if redis_client.exists(redis_key):
        return redis_client.get(redis_key).decode("utf-8")

    if len(messages) < 11:
        return ""

    full_chat = "\n".join([f"{msg.type.upper()} : {msg.content}" for msg in messages])
    prompt_resume = (
        "Voici une conversation WhatsApp. Résume-la en 5 phrases claires :\n"
        "1. Qui est la personne, son entreprise ? (Nom si connu)\n"
        "2. Que fait-elle dans la vie ?\n"
        "3. Quels sont ses besoins ou objectifs ?\n"
        "4. Quoi l'intéresse avec la CCI ?\n"
        "5. Un fait important à retenir pour la suite ? (budget, localisation, urgence...)\n\n"
        f"Conversation :\n{full_chat}"
    )

    try:
        summary = await llm.ainvoke(prompt_resume)
        summary_text = summary.content if hasattr(summary, "content") else str(summary)
        redis_client.set(redis_key, summary_text, ex=60 * 60 * 24 * 60)  # expire dans 30 jours
        return summary_text
    except Exception as e:
        print("Erreur génération résumé utilisateur :", e)
        return ""

prompt_template = load_prompt_template()

async def agent_response(user_input: str, chat_id: str) -> str:
    today = datetime.now().strftime("%d %B %Y")
    memory = get_memory(chat_id)
    
    memory.chat_memory.add_message(HumanMessage(content=user_input))
    messages = memory.chat_memory.messages
    short_term_memory = "\n".join([f"{msg.type.capitalize()} : {msg.content}" for msg in messages[-30:]])
    user_profile_summary = await get_user_profile_summary(chat_id, messages)
    
    # Affichage de la mémoire courte
    #print("\n📝 Short-term memory qui sera utilisée:")
    #print(short_term_memory or "[Aucune mémoire]")
    #print("="*50)
    #print("\n📝 long-term qui sera utilisée:")
    #print(user_profile_summary or "[Aucune mémoire]")
    #print("="*50)


    # Récupération du contexte
    base_cci_context_docs = retriever.invoke(user_input)
    base_cci_context = "\n\n".join(doc.page_content for doc in base_cci_context_docs) if base_cci_context_docs else "[Pas d'information pertinente dans la base.]"
    
    prompt = prompt_template.replace("{{today}}", today)\
                            .replace("{{user_input}}", user_input)\
                            .replace("{{short_term_memory}}", short_term_memory or "[Aucune mémoire courte]")\
                            .replace("{{user_profile}}", user_profile_summary or "[Profil encore inconnu]")\
                            .replace("{{cci_context}}", base_cci_context)
    
    # Affichage du prompt
    #print("\n📝 Prompt qui sera utilisé:")
    #print(prompt)
    #print("="*50)
    
    try:
        # Timeout de 9 secondes pour la réponse de l'agent
        reply = await asyncio.wait_for(llm.ainvoke(prompt), timeout=9.0)
        reply_text = reply.content if hasattr(reply, "content") else str(reply)

        # ⬅️ Ajout **synchronisé** du message AI (ordre garanti)
        memory.chat_memory.add_message(AIMessage(content=reply_text))

        return reply_text

    except asyncio.TimeoutError:
        return "Je n'ai pas pu répondre à temps. Pouvez-vous reformuler ou poser une question plus simple ?"
    except Exception as e:
        return f"Erreur interne : {str(e)}"


    
async def surveillance_inactivite(chat_id: str):
    try:
        result = await get_full_conversation_postgre(chat_id)
        history = result.get("history", "")
        chatType = result.get("type", "undefined")
        value = result.get("value", "")

        if has_calendly_link(history):
            store_calendy_link(chat_id)
            lead = extract_lead_info(history)
            store_lead_to_google_sheet(lead, type=chatType, chat_id=chat_id, numberWhatsapp=value)

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

