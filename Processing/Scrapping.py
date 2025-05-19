import requests
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import json

BASE_URL = "https://www.franciamexico.com"

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--window-size=1920,1080')
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def get_event_links():
    driver = setup_driver()
    try:
        url = f"{BASE_URL}/evenements/prochains-evenements.html"
        print("🌐 Chargement de la page...")
        driver.get(url)
        
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "main")))
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        events = []
        # Cibler la structure exacte des événements
        for article in soup.select('article.thumbnail.thumbnail-inline'):
            a = article.select_one('div.caption a[href]')
            if a:
                href = a['href']
                full_url = BASE_URL + href if not href.startswith('http') else href
                if full_url not in events:
                    events.append(full_url)
                    print(f"📌 Événement trouvé : {full_url}")
        return events
    except Exception as e:
        print(f"Erreur lors de la récupération des événements: {e}")
        return []
    finally:
        driver.quit()

def parse_event_details(event_url):
    driver = setup_driver()
    try:
        driver.get(event_url)
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Titre
        title = soup.find('h1')
        title = title.text.strip() if title else "Titre non trouvé"

        # Scraper tout le body, mais filtrer intelligemment
        body_content = soup.find('body')
        page_content = {}
        
        textes_a_ignorer = [
            'cookies', 'cookie', 'politique de confidentialité', 'mentions légales',
            'inscrivez-vous', 'ajouter à mon agenda', 'voir sur la carte', 'fermer',
            'partager', 'imprimer', 'retour', 'accueil', 'menu', 'rechercher',
            'suivez-nous', 'newsletter', 's\'abonner', 'se connecter',
            'sélectionnez votre fuseau horaire', 'fuseau horaire', 'timezone', 'dropdown', 'liste déroulante', 'select timezone'
        ]

        if body_content:
            for element in body_content.find_all(['div', 'p', 'span', 'li', 'h2', 'h3', 'h4']):
                # Ignorer les éléments avec des classes/id spécifiques
                if element.get('class'):
                    if any(cls in str(element.get('class')).lower() for cls in ['cookie', 'footer', 'header', 'nav', 'menu', 'social', 'timezone', 'dropdown', 'select']):
                        continue
                if element.get('id'):
                    if any(idx in str(element.get('id')).lower() for idx in ['cookie', 'footer', 'header', 'nav', 'menu', 'social', 'timezone', 'dropdown', 'select']):
                        continue
                text = element.get_text(strip=True)
                if text and len(text) > 1:
                    # Filtres avancés
                    if any(mot in text.lower() for mot in textes_a_ignorer):
                        continue
                    if text.startswith("Sélectionnez votre fuseau horaire"):
                        continue
                    if text.count("UTC") > 10:
                        continue
                    if len(text) > 1000 and text.count("\n") > 10:
                        continue  # Probablement une grosse liste déroulante
                    if element.name == 'select' or 'option' in str(element):
                        continue  # Éviter les listes déroulantes
                    if element.parent and element.parent.name == 'select':
                        continue
                    if element.get('aria-label') and 'fuseau' in element.get('aria-label').lower():
                        continue
                    if element.name in ['h2', 'h3', 'h4'] or any(marker in text.lower() for marker in [':', 'le ', 'à ', 'adresse', 'lieu', 'date']):
                        key = text
                        page_content[key] = ""
                    else:
                        if page_content:
                            last_key = list(page_content.keys())[-1]
                            if page_content[last_key]:
                                page_content[last_key] += " " + text
                            else:
                                page_content[last_key] = text

        return {
            "titre": title,
            "contenu_complet": page_content,
            "url": event_url
        }
    except Exception as e:
        print(f"Erreur lors de la récupération des détails de l'événement {event_url}: {e}")
        return None
    finally:
        driver.quit()

def get_all_event_details():
    events = []
    links = get_event_links()

    for url in links:
        try:
            details = parse_event_details(url)
            if details:
                events.append(details)
        except Exception as e:
            print(f"Erreur sur {url} : {e}")
    
    if events:
        with open("evenements.json", "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)
        print("✅ Les événements ont été sauvegardés dans evenements.json")
    else:
        print("❌ Aucun événement n'a pu être récupéré.")

if __name__ == "__main__":
    get_all_event_details()
