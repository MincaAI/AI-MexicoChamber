import streamlit as st
from Agent.agent2 import agent_response
import uuid

# Configuration de la page
st.set_page_config(
    page_title="CCI Mexico Chat Assistant",
    layout="centered"
)

# Initialisation de la session
if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# Titre et description
st.title("Assistant CCI Mexico")
st.markdown("""
Bienvenue ! Je suis l'assistant virtuel de la Chambre de Commerce et d'Industrie Franco-mexicaine.  
Comment puis-je vous aider aujourd'hui ?  

<br>

¡Bienvenido! Soy el asistente virtual de la Cámara de Comercio e Industria Franco-Mexicana.  
¿En qué puedo ayudarle hoy?
""", unsafe_allow_html=True)

# Affichage des messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Zone de saisie
if prompt := st.chat_input("Votre message..."):
    # Ajout du message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Réponse de l'agent
    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            response = agent_response(prompt, st.session_state.user_id)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response}) 