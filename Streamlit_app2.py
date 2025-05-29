import streamlit as st
import asyncio
import uuid
from Agent.agent2005 import agent_response, extract_lead_info, get_full_conversation

# === Config Streamlit ===
st.set_page_config(
    page_title="CCI Mexico AI Assistant",
    layout="centered"
)

# === Initialisation de session ===
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())
    
if st.sidebar.button("🆕 Nouvelle session"):
    st.session_state.clear()  # Réinitialise toute la session proprement
    st.rerun()

# === SIDEBAR : Analyse du lead ===
with st.sidebar:
    st.markdown("### 🔎 Analyse lead")
    if st.button("Analyser le lead maintenant"):
        with st.spinner("Analyse en cours..."):
            history = get_full_conversation(st.session_state.chat_id)
            lead = extract_lead_info(history)

            if lead.get("email") != "inconnu":
                st.success("✅ Lead détecté")
                st.json(lead)
            else:
                st.warning("Pas de lead qualifié détecté.")

# === Interface principale ===
st.title("Assistant CCI Mexico 🇲🇽🤖")

st.markdown("""
Bienvenue ! Je suis l'assistant virtuel de la Chambre de Commerce et d'Industrie Franco-mexicaine.  
**Comment puis-je vous aider aujourd'hui ?**

<br>

¡Bienvenido! Soy el asistente virtual de la Cámara de Comercio e Industria Franco-Mexicana.  
¿En qué puedo ayudarle hoy?
""", unsafe_allow_html=True)

# === Affichage historique des messages ===
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# === Entrée utilisateur ===
async def get_assistant_response(prompt, chat_id):
    return await agent_response(prompt, chat_id)

prompt = st.chat_input("Votre message...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = asyncio.run(get_assistant_response(prompt, st.session_state.chat_id))
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
