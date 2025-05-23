🤖 AI Agent – CCI France Mexico
This project implements an intelligent conversational assistant for the Franco-Mexican Chamber of Commerce and Industry (CCI).
It helps website visitors understand CCI services, upcoming events, and connect with the right people — while generating qualified leads automatically.

cci_ai_project/
├── Agent/
│   ├── agent2005.py              # Main LangChain agent (session + long-term memory)
│   ├── storage.py                # Handles lead storage to Google Sheets
│   └── __init__.py               # (optional) to make Agent a Python package
│
├── Data/
│   ├── evenements_raw.json       # Raw event data scraped from the CCI website
│   └── evenements_structures.txt # Formatted version injected into the agent prompt
│
├── Processing/
│   ├── Scrapping.py              # Scraper for CCI website events (to be run monthly)
│   └── Formatter.py              # Cleans and formats event data for prompt use
│
├── prompt_base.txt               # 🔥 Prompt template injected dynamically (at root)
├── Streamlit_app.py              # Streamlit UI interface for the chatbot
├── google-credentials.json       # Google Sheets service account (DO NOT COMMIT)
├── .env                          # API keys and environment variables
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation



🧠 Key Features
🌐 Multilingual – Responds in French or Spanish based on the user's input

🧠 Memory-enabled:

Session memory: via InMemoryChatMessageHistory

Long-term memory: stored in Pinecone and retrieved on each interaction

🔍 Context-aware:

Vector search (retriever) brings relevant CCI info into the prompt

Current and upcoming events are injected from pre-formatted text

📅 Monthly updates:

Event data is scraped monthly from the official CCI website

🤝 Lead generation automation:

When interest is shown (e.g. Calendly link clicked), lead info is extracted

Data is stored in a shared Google Sheet used by CCI staff

🤖 Agent Behavior
Starts by asking 1–2 questions to understand the user's context (business, project, needs)

Answers questions about services or events in a clear and informative way

Only shares Calendly booking links when interest is clearly expressed

Ends the conversation by politely requesting the user’s name and email (opt-in)

All answers are generated with a professional and human tone, not robotic

🚀 Running the Agent Locally
1. Clone the repo
bash
Copy
Edit
git clone https://github.com/your-org/cci_ai_project.git
cd cci_ai_project
2. Create a .env file with:
env
Copy
Edit
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_ENV=...
PINECONE_INDEX=cci-index
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Start the terminal-based chatbot
bash
Copy
Edit
python Agent/agent2005.py
🌐 Streamlit Web Interface
Launch locally:

bash
Copy
Edit
streamlit run Streamlit_app.py
Each session has a unique session_id using uuid

The conversation runs in a web chat UI

A background thread detects inactivity and automatically triggers lead extraction

📤 Lead Storage (Google Sheets)
Qualified leads are extracted and pushed to a Google Sheet:
CCI_support_agent_lead

Format: first name, last name, company, email, interest, date

The lead is only stored when a Calendly link has been sent and info is complete

The system uses OpenAI to extract this info based on full conversation history

🔐 Security & Best Practices
.env and google-credentials.json must be excluded from Git (add to .gitignore)

On Streamlit Cloud, use st.secrets for secure credential handling

Logs and extracted leads are never stored locally

Pinecone data includes timestamps and session-level metadata

✅ Dependencies (requirements.txt)
txt
Copy
Edit
streamlit
langchain
openai
gspread
oauth2client
python-dotenv
pinecone-client