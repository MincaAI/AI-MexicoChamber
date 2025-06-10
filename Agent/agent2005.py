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
llm = ChatAnthropic(
    temperature=0.6,
    model="claude-3-5-sonnet-20241022",
    streaming=False,
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    callbacks=[StreamPrintCallback()]
)

#llm = ChatOpenAI(
    #temperature=0.6,
    #model="gpt-4o",
    #streaming=False,
    #callbacks=[StreamPrintCallback()]
#)




# === 3. Pinecone ===

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
retriever = PineconeVectorStore(
    index=index,
    embedding=embeddings
).as_retriever(search_kwargs={"k": 2})
#new
summary_store = PineconeVectorStore(index=index, embedding=embeddings,namespace="summaries")  # new

#new
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
# new 
def load_summary_from_pinecone(chat_id: str) -> str:
    docs = summary_store.similarity_search("", k=1, filter={"chat_id": chat_id, "summary_id": f"summary_{chat_id}"})
    return docs[0].page_content if docs else ""
# new
def generate_summary_from_memory(memory: ConversationSummaryBufferMemory, chat_id: str, max_messages: int = 15) -> str:
    new_messages = memory.chat_memory.messages[-max_messages:]
    convo_text = "\n".join([f"{msg.type.capitalize()} : {msg.content}" for msg in new_messages])

    if not convo_text.strip():
        return ""

    old_summary = load_summary_from_pinecone(chat_id)

    prompt = f"""
    Voici le résumé actuel d'une conversation WhatsApp entre toi et l'utilisateur  :
    {old_summary or '[aucun résumé pour le moment]'}

    Voici les nouveaux messages à intégrer :
    {convo_text}

    Génére un **nouveau résumé synthétique mis à jour**, qui conserve les éléments utiles de l'ancien et ajoute les nouvelles informations importantes. Résume en 5 phrases maximum.
    """

    summary = llm.invoke(prompt).content.strip()
    save_or_update_summary(chat_id, summary)
    return summary
#new

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
        max_token_limit=600
    )
    return memory


def load_prompt_template():
    with open("prompt_base.txt", encoding="utf-8") as f:
        return f.read().strip()
    

#new 
def get_and_increment_counter(chat_id: str) -> int:
    key = f"msg_counter:{chat_id}"
    counter = redis_client.incr(key, 1)
    redis_client.expire(key, 86400)  # 24h
    return counter

prompt_template = load_prompt_template()
async def agent_response(user_input: str, chat_id: str) -> str:
    
    today = datetime.now().strftime("%d %B %Y")
    memory = get_memory(chat_id)
    messages = memory.chat_memory.messages
    short_term_memory = "\n".join([f"{msg.type.capitalize()} : {msg.content}" for msg in messages[-10:]])
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

    memory.chat_memory.add_message(HumanMessage(content=user_input))
    memory.chat_memory.add_message(AIMessage(content=reply_text))

    counter = get_and_increment_counter(chat_id)
    if counter >= 15:
        redis_client.delete(f"msg_counter:{chat_id}")
        generate_summary_from_memory(memory, chat_id)

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

