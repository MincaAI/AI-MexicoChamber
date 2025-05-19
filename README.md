# 🤖 AI Agent – CCI France Mexico

This project delivers an AI system for an french institution. It includes three AI agents, starting with the **Expert Agent** deployed on their official website to assist visitors with accurate and context-aware responses about CCI's services and events.

---

## 📁 Project Structure

cci_ai_project/
├── Agent/
│ └── agent2.py
│
├── Data/
│ ── evenements.json
│ └── evenements_structures.txt
│
├── Processing/
│ ├── scraper.py
│ └── formatter.py
│
├── .env
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## 📦 Component Breakdown

### `Agent/agent2.py`

The main LangChain-powered AI agent:

- Handles user queries
- Uses a Pinecone vector database for CCI knowledge
- Injects structured events into the prompt
- Stores long-term memory by `user_id`
- Key function: `agent_response(message, user_id)`

---

### `Data/raw/evenements.json`

Raw events scraped from the CCI website (monthly).  
Used as the source file for formatting.

---

### `Data/processed/evenements_structures.txt`

Cleaned and formatted version of the event data.  
This is **directly injected into the prompt** to inform the AI of upcoming events.

---

### `Processing/scraper.py`

Script that scrapes event information from the CCI website.  
It writes the output to `evenements.json`.

> To be run **monthly** (e.g. manually or via CRON job).

---

### `Processing/formatter.py`

Cleans and formats `evenements.json` into the prompt-ready `evenements_structures.txt`.

---

### `.env`

Environment variables:

- `OPENAI_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_ENV`
- `PINECONE_INDEX`

> ⚠️ This file must **not be committed** to Git (contains secrets).

---

### `requirements.txt`

All required Python dependencies for the AI agent, Pinecone, OpenAI, and environment handling.

---

## 🚀 Usage

### Local Testing

1. Clone the repository
2. Add your `.env` file with valid keys
3. Run the agent:

```bash
python Agent/agent2.py
```
