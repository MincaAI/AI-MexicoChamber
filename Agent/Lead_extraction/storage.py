# storage.py
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import os
import pytz

def store_lead_to_google_sheet(lead_data: dict, type: str = 'undefined', numberWhatsapp: str ='',chat_id :str =''):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    # Utiliser le chemin relatif à la racine du projet
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    creds_path = os.path.join(project_root, "google-credentials.json")
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
    client = gspread.authorize(creds)

    # Nom exact du fichier Google Sheet
    sheet = client.open("CCI_support_agent_lead").sheet1

    # Date actuelle (en UTC)
    #date_now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    mexico_tz = pytz.timezone("America/Mexico_City")
    date_now = datetime.now(mexico_tz).strftime("%Y-%m-%d %H:%M:%S")

    # Envoie des données dans l'ordre des colonnes A → F
    sheet.append_row([
        lead_data.get("prenom", "inconnu"),
        lead_data.get("nom", "inconnu"),
        lead_data.get("entreprise", "inconnu"),
        lead_data.get("email", "inconnu"),
        lead_data.get("interet", "inconnu"),
        lead_data.get("score", 1),
        date_now,        
        type,
        numberWhatsapp,
        chat_id
    ])
