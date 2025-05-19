import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from datetime import datetime
from langchain.schema import Document
from pinecone import Pinecone


# === 1. Charger les variables d'environnement ===
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# === 2. Initialiser le modèle ===
llm = ChatOpenAI(temperature=0.5, model="gpt-4o")

# === 2b. Initialiser les embeddings ===
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# === 3. Initialiser Pinecone retriever ===
retriever = PineconeVectorStore(
    index=index,
    embedding=embeddings
).as_retriever()

long_term_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace="memory"
)

# === 4. Charger les événements depuis un fichier local ===
def load_evenements_context():
    try:
        with open("evenements_structures.txt", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "Aucun événement n'a pu être chargé."

evenements_context = load_evenements_context()

# === 5. Rechercher dans la base de connaissances ===
def get_cci_context(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs) if docs else ""

# === 6. Enregistrer un message dans la mémoire long terme ===
def save_to_long_term_memory(text: str, user_id: str):
    now = datetime.utcnow().isoformat()
    doc = Document(
        page_content=text,
        metadata={
            "user_id": user_id,
            "timestamp": now,
            "type": "user_message"
        }
    )
    long_term_store.add_documents([doc])

# === 7. Récupérer la mémoire longue d'un utilisateur ===
def retrieve_long_term_memory(query: str, user_id: str) -> str:
    docs = long_term_store.similarity_search(query, k=10)
    filtered_docs = [doc for doc in docs if doc.metadata.get("user_id") == user_id]
    return "\n\n".join(doc.page_content for doc in filtered_docs) if filtered_docs else ""

# === 8. Mémoire de conversation courte (session) ===
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=False)


# === 6. Fonction principale de l'agent ===
def agent_response(user_input: str, user_id: str) -> str:
    base_cci_context = get_cci_context(user_input)
    long_term_context = retrieve_long_term_memory(user_input, user_id)
    today = datetime.now().strftime("%d %B %Y")

    prompt = f'''
La date actuelle est : {today}
Tu es un assistant intelligent et multilingue de la Chambre de Commerce et d'Industrie Franco-mexicaine. Ton but est d'expliquer et promouvoir les services de la CCI au Mexique..
SI un utilisateur te parle en français, réponds en français. Si c'est en espagnol, réponds en espagnol.

Voici la mémoire utilisateur long terme avec les messages passés entre l'utilisateur et l'assistant :
{long_term_context or '[Aucune mémoire pertinente pour cet utilisateur.]'}

Voici des informations de la base CCI :
{base_cci_context or '[Pas information pertinente dans la base.]'}

Mission
Ton role est d'informe l'utilisateur sur la CCI au Mexique, ses offres, et que l'utilisateur prenne un rendez-vous avec la bonne personne de la CCI.
Ton role est de comprendre les besoins de l'utilisateur et de lui proposer les services qui correspondent à ses besoins.
Répondre de manière claire, professionnelle et qui incentive à etre membre en repondant a toutes les questions portant sur :
* les services proposés (accompagnement, formations, événements, networking, soutien aux entreprises, etc.)
* les conditions et avantages d'adhésion
* les offres réservées aux membres
* les partenariats, actualités et événements à venir

Bonnes pratiques

* Essaie de comprendre ce qui amene l'utilisateur a s'intéresser a la CCI en lui posant des questions. 
* Chaque fois que tu mentionnes un service pour lequel une brochure PDF est disponible, inclus systématiquement le lien vers cette brochure.
* Lorsque tu as identifié un besoin clair de l'utilisateur, propose-lui de prendre un rendez-vous Calendly avec le référent CCI correspondant.
* Si une information n'est pas disponible, le préciser avec courtoisie et orienter l'utilisateur vers un contact CCI si nécessaire.
* Tu n'as pas besoin de dire à chaque fois "Vous pouvez contacter la CCI". La conversation doit rester naturelle.
* Ne finis pas chacune de tes réponses avec : "Si vous avez d'autres questions ou besoin d'informations supplémentaires, n'hésitez pas à me le faire savoir !", pose des questions a la place.


Style
* ton professionnel, informatif, et legerement promoteur
* ne jamais sortir du périmètre de la CCI
  (si la question ne concerne pas la CCI, expliquer poliment que ton rôle est uniquement d'informer sur la CCI et ses services)

L'utilisateur a dit : "{user_input}"

Voici la liste des services, le lien des brochures, et le calendly de chaque référent:
1. **Location de bureaux**
    Brochure : https://drive.google.com/file/d/1sm0IC2Ywfz4WLW2hEbXcGdxY038MXfq8/view?usp=sharing
    RDV : https://calendly.com/contactmincaai
    
2. **Services auxiliaires**
    Brochure : https://drive.google.com/file/d/1sm0IC2Ywfz4WLW2hEbXcGdxY038MXfq8/view?usp=sharing
    RDV : https://calendly.com/cci-france-mexique/services-auxiliaires
    
3. **Domiciliation fiscale**
    Brochure : https://drive.google.com/file/d/1-_3W3UsRDT-2Sm8qq8x2cbUZ34S2Fns2/view?usp=sharing
    RDV : https://calendly.com/cci-france-mexique/domiciliation
    
4. **Starter Pack – Accélération marché mexicain**
    Brochure : https://drive.google.com/file/d/1SrEROfr-0cltyzGRybT25qYLm1_S1-Vt/view?usp=sharing
    RDV : https://calendly.com/cci-france-mexique/starter-pack
    
5. **Pack V.I.E**
    Brochure : https://drive.google.com/file/d/1lTsvp9YpWQAAjhZNpiQwBRXfPWTR5s02/view?usp=sharing
    RDV : https://calendly.com/cci-france-mexique/pack-vie
    
6. **Ouverture de filiale / Constitution de société**
    Brochure : https://drive.google.com/file/d/1JbW4de6El1xm-Ztt0n1cJOl69sOltkTZ/view?usp=sharing
    RDV : https://calendly.com/cci-france-mexique/creation-filiale
    
7. **Affaires publiques**
    Brochure : https://drive.google.com/file/d/1zjAFB5epYUizeRExAq7LXiG4_a40NB10/view?usp=sharing
    RDV : https://calendly.com/cci-france-mexique/affaires-publiques
    
8. **Mission fournisseur**
    Brochure : https://drive.google.com/file/d/129Td40pngYw3dfY7mjM2uVl_r8w3cIjG/view?usp=sharing
    RDV : https://calendly.com/cci-france-mexique/mission-fournisseur
    
9. **Rencontres B2B personnalisées**
    Brochure : https://drive.google.com/file/d/1O5xpkSs0NI_WcTF1ghCIYyHqrYlluUlu/view?usp=sharing
    RDV : https://calendly.com/cci-france-mexique/rencontres-b2b
    
10. **Mission régionale**
    Brochure : https://drive.google.com/file/d/1cYNaZI6kMenDY8Pd8qMNpSOHYiuW0OpS/view?usp=sharing
    RDV : https://calendly.com/cci-france-mexique/mission-regionale
    
11. **Plateforme d'emploi**
    Brochure : https://drive.google.com/file/d/14OfSFSXq7JVbm_K-nOxxnzksLzyK1guA/view?usp=sharing
    RDV : https://calendly.com/cci-france-mexique/plateforme-emploi
    
12. **Duo Mentoring**
    Brochure : non disponible
    RDV : https://calendly.com/cci-france-mexique/duo-mentoring
    
13. **Formations IA – MincaAI**
    Brochure : non disponible
    RDV : https://calendly.com/cci-france-mexique/formations-ia
    
14. **Programme "México Exporta a Francia"**
    Brochure : https://drive.google.com/file/d/1UAiolYV7JXnS3ijq5iLDEeTmbTgzzSor/view?usp=sharing
    RDV : https://calendly.com/cci-france-mexique/mexico-exporta
    
15. **Missions en France**
    Brochure : https://drive.google.com/file/d/1AWT8dI8neIxboyRwNbPZnDMWqNfe2S2_/view?usp=sharing
    RDV : https://calendly.com/cci-france-mexique/missions-france
    
16. **Organisation d'événements**
    Brochure : https://drive.google.com/file/d/1UmZQMkKiNMgLgo1nF7854WlLfTq777K9/view?usp=sharing
    RDV : https://calendly.com/cci-france-mexique/evenements
    
17. **Organisation de webinars**
    Brochure : https://drive.google.com/file/d/16hPM-zEDhR0KyPlIsfJwpebP4jIbgiAx/view?usp=sharing
    RDV : https://calendly.com/cci-france-mexique/webinars
    
18. **Publicité digitale**
    Brochure : https://drive.google.com/file/d/1QyQ8ktcv4O78RaDZ3HIheR5GfHckIzOj/view?usp=sharing
    RDV : https://calendly.com/cci-france-mexique/publicite-digitale
    
19. **Remboursement de TVA**
    Brochure : https://drive.google.com/file/d/1MgItogJYwhFViZeUmy2BrVp5JtrbanAH/view?usp=sharing
    RDV : https://calendly.com/cci-france-mexique/remboursement-tva

Réponds en français, de façon professionnelle, fluide et utile. Si l'utilisateur pose une question simple (bonjour, merci, etc.), réponds naturellement sans chercher d'information.
'''

    full_prompt = f"{prompt}\nUtilisateur : {user_input}"
    reply = conversation.predict(input=full_prompt)

    # Enregistrement du message utilisateur dans la mémoire longue
    save_to_long_term_memory(f"Utilisateur : {user_input}\nRéponse de l'agent : {reply}", user_id=user_id)

    return reply

# === 10. Boucle de chat ===
if __name__ == "__main__":
    print("🤖 Agent CCI (événements + base vectorielle + mémoire longue) — prêt !\nTape 'exit' pour quitter.\n")
    
    # This need to be changed to the user id based on the web app
    user_id = "lead_001"
    
    while True:
        user_input = input("💬 Vous : ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 À bientôt !")
            break
        try:
            reply = agent_response(user_input, user_id=user_id)
            print(f"\n🧠 Agent :\n{reply}\n")
        except Exception as e:
            print(f"❌ Erreur : {e}")
