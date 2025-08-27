from __future__ import annotations
import os
import sys
import time
import threading
import asyncio
from datetime import datetime, timezone
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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

# Chargement unique des variables d'environnement (seulement si fichier .env existe)
try:
    load_dotenv()
except:
    pass  # Pas de fichier .env en production AWS App Runner

# Initialisation sécurisée de Redis
redis_url = os.getenv("REDIS_URL")
redis_client = None
try:
    if redis_url:
        redis_client = redis.Redis.from_url(redis_url)
        # Test de connexion
        redis_client.ping()
        # Redis connecté avec succès (print supprimé pour WhatsApp latence)
except Exception as e:
    print(f"⚠️ Redis non disponible: {e}")
    redis_client = None

inactivity_event = threading.Event()

# StreamPrintCallback supprimé pour optimiser la latence WhatsApp
# class StreamPrintCallback(BaseCallbackHandler):
#     def on_llm_new_token(self, token: str, **kwargs):
#         print(token, end="", flush=True)

def get_redis_client():
    """Retourne un client Redis basé sur REDIS_URL, ou None en cas d'erreur."""
    try:
        if redis_client:
            return redis_client
    except Exception:
        pass
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    try:
        return redis.Redis.from_url(url)
    except Exception as e:
        # Erreur connexion Redis (print supprimé pour WhatsApp latence)
        return None

# === 1. Variables d'environnement (déjà chargées) ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# === 2. Initialisation ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Tentative d'utilisation de GPT-4.1 avec fallback vers GPT-4o (optimisé pour AWS)
try:
    # Initialisation GPT-4.1 (prints supprimés pour WhatsApp latence)
    llm = ChatOpenAI(
        temperature=0.2,
        model="gpt-4.1-mini",
        streaming=False,  # Désactivé pour WhatsApp latence
        max_tokens=350
    )
    # GPT-4.1 initialisé (test de connexion différé)
except Exception as e:
    print(f"⚠️ GPT-4.1 non disponible ({str(e)[:50]}...), fallback vers GPT-4o")
    llm = ChatOpenAI(
        temperature=0.2,
        model="gpt-4o-mini",
        streaming=False,  # Désactivé pour WhatsApp latence
        max_tokens=350
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
        max_token_limit=800
    )
    return memory

def load_prompt_template():
    """Charge le template de prompt français avec chemin absolu."""
    try:
        # Chemin absolu pour AWS App Runner
        base_path = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(os.path.dirname(base_path), "prompt_base.txt")
        with open(prompt_path, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fallback pour développement local
        with open("prompt_base.txt", encoding="utf-8") as f:
            return f.read().strip()

def load_prompt_template_es():
    """Charge le template de prompt espagnol avec chemin absolu."""
    try:
        # Chemin absolu pour AWS App Runner
        base_path = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(os.path.dirname(base_path), "prompt_base_es.txt")
        with open(prompt_path, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fallback pour développement local
        with open("prompt_base_es.txt", encoding="utf-8") as f:
            return f.read().strip()



async def detect_language(text: str) -> str:
    """
    Détecte la langue d'un message texte avec GPT-4o-mini.
    Returns: 'fr' pour français, 'es' pour espagnol
    """
    # LLM spécifique pour la détection de langue (plus rapide et moins cher)
    # Essayer GPT-4.1-mini s'il existe, sinon fallback vers GPT-4o-mini
    try:
        language_detector = ChatOpenAI(
            temperature=0,
            model="gpt-4.1-mini",
            max_tokens=10
        )
    except:
        language_detector = ChatOpenAI(
            temperature=0,
            model="gpt-4o-mini",
            max_tokens=10
        )
    
    prompt = f"""Détecte la langue de ce message et réponds uniquement par 'fr' pour français ou 'es' pour espagnol:

Message: "{text}"

Réponse:"""
    
    try:
        result = await language_detector.ainvoke(prompt)
        detected = (result.content or "").strip().lower()
        # Normalisation robuste des sorties potentielles
        if "es" in detected or "espa" in detected:
            return 'es'
        if "fr" in detected or "fran" in detected:
            return 'fr'
        return 'fr'
    except Exception as e:
        # Erreur détection langue (print supprimé pour WhatsApp latence)
        # Fallback en cas d'erreur
        return 'fr'

async def get_or_detect_user_language_and_prompt(chat_id: str, user_input: str) -> tuple[str, str]:
    """
    Récupère ou détecte la langue de l'utilisateur et retourne (langue, prompt_template).
    Ne fait la détection qu'une seule fois par utilisateur.
    """
    redis_key = f"user_language:{chat_id}"
    
    # Vérifier si la langue est déjà stockée
    redis_client = get_redis_client()
    if redis_client:
        stored_language = redis_client.get(redis_key)
        if stored_language:
            language = stored_language.decode('utf-8')
            # Charger le bon prompt depuis les fichiers
            prompt = load_prompt_template_es() if language == 'es' else load_prompt_template()
            return language, prompt
    
    # Première fois : détecter la langue du premier message
    detected_language = await detect_language(user_input)
    
    # Stocker pour 30 jours
    if redis_client:
        redis_client.set(redis_key, detected_language, ex=60 * 60 * 24 * 30)
    
    # Charger le bon prompt depuis les fichiers
    prompt = load_prompt_template_es() if detected_language == 'es' else load_prompt_template()
    return detected_language, prompt


async def get_user_profile_summary(chat_id: str, messages: list) -> str:
    redis_key = f"profile_summary:{chat_id}"
    if redis_client.exists(redis_key):
        return redis_client.get(redis_key).decode("utf-8")

    if len(messages) < 24:
        return ""

    full_chat = "\n".join([f"{msg.type.upper()} : {msg.content}" for msg in messages[-40:]])
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
        # Erreur génération résumé utilisateur (print supprimé pour WhatsApp latence)
        return ""

async def agent_response(user_input: str, chat_id: str) -> str:
    today = datetime.now().strftime("%d %B %Y")
    memory = get_memory(chat_id)
    
    memory.chat_memory.add_message(HumanMessage(content=user_input))
    messages = memory.chat_memory.messages
    short_term_memory = "\n".join([f"{msg.type.capitalize()} : {msg.content}" for msg in messages[-12:]])
    user_profile_summary = await get_user_profile_summary(chat_id, messages)
    
    # Optimisation : récupération directe langue depuis Redis si déjà stockée
    redis_lang_key = f"user_language:{chat_id}"
    cached_language = redis_client.get(redis_lang_key) if redis_client else None
    
    if cached_language:
        # Langue déjà en cache - récupération directe du prompt
        user_language = cached_language.decode('utf-8')
        if user_language == 'es':
            prompt_template = load_prompt_template_es()
        else:
            prompt_template = load_prompt_template()
    else:
        # Première fois - détection et stockage
        user_language, prompt_template = await get_or_detect_user_language_and_prompt(chat_id, user_input)
    
    # Affichage de la mémoire courte (debug)
    #print(f"\n📝 Langue détectée/stockée: {user_language}")
    #print("\n📝 Short-term memory qui sera utilisée:")
    #print(short_term_memory or "[Aucune mémoire]")
    #print("="*50)
    #print("\n📝 long-term qui sera utilisée:")
    #print(user_profile_summary or "[Aucune mémoire]")
    #print("="*50)

    # Récupération du contexte
    base_cci_context_docs = retriever.invoke(user_input)
    base_cci_context = "\n\n".join(doc.page_content for doc in base_cci_context_docs) if base_cci_context_docs else "[Pas d'information pertinente dans la base.]"
    
    # Messages par défaut selon la langue
    default_memory = "[Ninguna memoria corta]" if user_language == 'es' else "[Aucune mémoire courte]"
    default_profile = "[Perfil aún desconocido]" if user_language == 'es' else "[Profil encore inconnu]"
    
    prompt = prompt_template.replace("{{today}}", today)\
                            .replace("{{user_input}}", user_input)\
                            .replace("{{short_term_memory}}", short_term_memory or default_memory)\
                            .replace("{{user_profile}}", user_profile_summary or default_profile)\
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
if __name__ == "__main__":
    import uuid
    chat_id = str(uuid.uuid4())  # 👈 Generate a unique ID per session
    print(f"🆔 Session Chat ID: {chat_id}")
    print("💬 Tapez vos messages (français ou espagnol). Ctrl+C pour quitter.\n")
    
    try:
        while True:
            user_input = input("💬 Vous : ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
            reply = asyncio.run(agent_response(user_input, chat_id=chat_id))
            print(f"\n🧠 Agent :\n{reply}\n")
    except KeyboardInterrupt:
        print("\n👋 Au revoir !")
        pass

