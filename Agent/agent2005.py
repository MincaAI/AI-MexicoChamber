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
from Agent.storage import store_lead_to_google_sheet  #
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler

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
long_term_store = PineconeVectorStore(index=index, embedding=embeddings, namespace="memory")

# Mémoire conversationnelle par utilisateur (chat history)
chat_histories = {}
inactivity_event = threading.Event()

#def load_evenements_context():
#    try:
#        with open("Data/evenements_structures.txt", encoding="utf-8") as f:
#            return f.read().strip()
#    except FileNotFoundError:
#        return ""
#    except Exception as e:
#        return ""

def load_prompt_template():
    with open("prompt_base.txt", encoding="utf-8") as f:
        return f.read().strip()
    
def load_extraction_prompt_template():
    with open("prompt_extraction.txt", encoding="utf-8") as f:
        return f.read().strip()
    
# evenements_context = load_evenements_context()

def get_full_conversation(chat_id: str) -> str:
    # Utilise un filtre Pinecone directement au lieu de tout rapatrier et filtrer ensuite
    docs = long_term_store.similarity_search(" ", k=50, filter={"chat_id": chat_id})
    docs.sort(key=lambda d: d.metadata.get("timestamp", ""))
    return "\n".join(doc.page_content for doc in docs)


def save_message_to_long_term_memory(role: str, message: str, chat_id: str):
    now = datetime.now(timezone.utc).isoformat()
    doc = Document(
        page_content=f"{role} : {message}",
        metadata={"chat_id": chat_id, "timestamp": now, "type": role}
    )
    long_term_store.add_documents([doc])


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

def get_chat_history(chat_id: str):
    if chat_id not in chat_histories:
        chat_histories[chat_id] = InMemoryChatMessageHistory()
    return chat_histories[chat_id]


chain = RunnableWithMessageHistory(llm, get_chat_history)
prompt_template = load_prompt_template()

async def agent_response(user_input: str, chat_id: str) -> str:
    today = datetime.now().strftime("%d %B %Y")
    history = get_full_conversation(chat_id)

    # ➕ Ajouter la recherche de contexte CCI à partir de la question posée
    base_cci_context_docs = retriever.invoke(user_input)
    base_cci_context = "\n\n".join(doc.page_content for doc in base_cci_context_docs) if base_cci_context_docs else "[Pas d'information pertinente dans la base.]"

    # ➕ Injecter dans le prompt
    prompt = prompt_template.replace("{{today}}", today)\
                            .replace("{{user_input}}", user_input)\
                            .replace("{{history}}", history or "[Aucune conversation précédente]")\
                            .replace("{{cci_context}}", base_cci_context)
    print("\n---🧠 CONTEXT WINDOW (prompt complet envoyé au LLM) ---\n")
    print(prompt)
    print("\n--- FIN CONTEXTE ---\n")
    print("\n🧠 Agent :\n", end="", flush=True)
    
    reply = await chain.ainvoke(input=prompt, config={"configurable": {"session_id": chat_id}})
    reply_text = reply.content if hasattr(reply, "content") else str(reply)

    save_message_to_long_term_memory("Utilisateur", user_input, chat_id)
    save_message_to_long_term_memory("Agent", reply_text, chat_id)
    return reply_text


def surveillance_inactivite(chat_id: str, timeout=50):
    while True:
        inactivity_event.clear()
        if not inactivity_event.wait(timeout):
            try:
                history = get_full_conversation(chat_id)
                if has_calendly_link(history):
                    lead = extract_lead_info(history)
                    if lead.get("prenom") != "inconnu" and lead.get("email") != "inconnu":
                        store_lead_to_google_sheet(lead)
            except Exception:
                pass
            sys.exit()
        time.sleep(1)


if __name__ == "__main__":
    import uuid
    chat_id = str(uuid.uuid4())  # 👈 Generate a unique ID per session
    print(f"🆔 Session Chat ID: {chat_id}")
    threading.Thread(target=surveillance_inactivite, args=(chat_id,), daemon=True).start()

    try:
        while True:
            user_input = input("💬 Vous : ")
            inactivity_event.set()
            reply = asyncio.run(agent_response(user_input, chat_id=chat_id))
            print(f"\n🧠 Agent :\n{reply}\n")
    except KeyboardInterrupt:
        pass

