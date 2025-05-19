import json
import openai
from dotenv import load_dotenv
import os

def structurer_evenements_via_llm(json_path, openai_api_key):
    with open(json_path, "r", encoding="utf-8") as f:
        events = json.load(f)
    prompt = f"""
Voici des événements extraits d'un site :
{json.dumps(events, ensure_ascii=False, indent=2)}

Organise-les proprement sous forme de tableau ou de liste structurée, en mettant en avant le plus d'informations disponibles.
"""
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Tu es un assistant qui structure des listes d'événements pour les rendre lisibles et organisées. Garde un Max d'info sur les evenements."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3000,
        temperature=0.2
    )
    with open("evenements_structures.txt", "w", encoding="utf-8") as f:
        f.write(response.choices[0].message.content)
    print("✅ Résultat structuré sauvegardé dans evenements_structures.txt")

if __name__ == "__main__":
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    structurer_evenements_via_llm("evenements.json", openai_api_key)