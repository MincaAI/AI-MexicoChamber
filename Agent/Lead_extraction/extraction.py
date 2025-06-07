import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(
    temperature=0.6,
    model="gpt-4",
)

def load_extraction_prompt_template():
    with open("prompt_extraction.txt", encoding="utf-8") as f:
        return f.read().strip()
    
    
def has_calendly_link(text: str) -> bool:
    return "https://calendly.com/" in text

# same
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
        