# TicketQuery — Natural Language Interface for Support Ticket Data

Ask a plain-English question about a customer support ticket dataset. The system
classifies the question and routes it to the right kind of answer:

- **text** — semantic questions about what's *in* tickets (RAG over ticket descriptions, via Chroma)
- **numeric** — aggregations (counts, averages, percentages) computed directly with pandas
- **chart** — breakdowns/comparisons across a category, rendered with Chart.js
- (bonus) **predict** — a trained ML model estimates a satisfaction rating for a hypothetical ticket

## Why this architecture (read before you present it)

The LLM (Groq) is used for two things only: **classifying intent**, and **generating
the pandas expression / chart spec / RAG answer**. It never invents the numeric answer
itself — numbers always come from actually running pandas on the real dataframe. This
matters: if you route numeric questions through an LLM directly, it will hallucinate
plausible-looking wrong numbers. Keep this distinction in your head when you explain
the project — "the LLM decides *how* to look, pandas *looks*."

RAG (Chroma) is used only for the "lookup" intent — semantic questions over free-text
fields (`Ticket Description`, `Resolution`). It is deliberately **not** used for
counts/averages; that would be the wrong tool.

## Setup

### 1. Get the real dataset
This build ships with a **synthetic placeholder CSV** (`data/customer_support_tickets.csv`,
1000 rows, same schema as the real Kaggle set) because the sandbox that built this has no
internet access to Kaggle. Download the real dataset and replace the file:

```
https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset
```

Keep the same filename/columns and everything downstream keeps working. Note: because
the placeholder's satisfaction ratings are random, the trained model's accuracy (~20%,
basically chance) is meaningless — retrain on the real CSV before you report any number.

### 2. Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env        # then add your GROQ_API_KEY
python train_model.py       # trains and saves model.joblib (needed for "predict" intent)
uvicorn main:app --reload --port 8000
```

Get a free Groq API key at https://console.groq.com/keys.

First run will download the Chroma embedding model (needs internet, one-time).

### 3. Frontend

No build step — it's static HTML/CSS/JS.

```bash
cd frontend
python3 -m http.server 5500
```

Open `http://localhost:5500`. It talks to the backend at `http://localhost:8000`
(CORS is already open in `main.py`).

## Project structure

```
support-nlq/
├── backend/
│   ├── main.py            # FastAPI app: routing, pandas exec, RAG, prediction
│   ├── train_model.py     # Trains the satisfaction-rating classifier
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── data/
│   └── customer_support_tickets.csv   # ← replace with the real Kaggle CSV
└── README.md
```

## What's still on you

- Swap in the real dataset and retrain the model.
- The "stat" and "chart" handlers execute an LLM-generated pandas expression inside a
  restricted `eval` (no imports/builtins). It blocks the obvious dangerous keywords, but
  don't expose this endpoint on the open internet without hardening it further — treat
  it as a prototype, not production-hardened.
- Groq occasionally returns malformed JSON for the router — there's a regex-based
  extractor to clean it up, but if you see parsing errors, tighten the system prompts or
  add a retry.
