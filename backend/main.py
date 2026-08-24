import os
import json
import re
import traceback

import pandas as pd
import numpy as np
import joblib
import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "customer_support_tickets.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

app = FastAPI(title="Support Ticket NLQ API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load data ----------
def _add_derived_resolution_duration(df):
    """Some dataset variants (the real Kaggle CSV included) don't have a
    ready-made numeric resolution-time column — only two raw timestamps,
    'First Response Time' and 'Time to Resolution'. If both are present and
    genuinely parse as datetimes, derive an hours-duration column from their
    difference. This dataset's timestamps are synthetically generated and
    aren't always causally consistent (Time to Resolution can land before
    First Response Time), so negative durations are treated as bad data and
    dropped rather than silently producing a nonsensical average."""
    if "First Response Time" not in df.columns or "Time to Resolution" not in df.columns:
        return df

    frt = pd.to_datetime(df["First Response Time"], errors="coerce")
    ttr = pd.to_datetime(df["Time to Resolution"], errors="coerce")

    if frt.notna().mean() < 0.05 or ttr.notna().mean() < 0.05:
        return df  # not meaningfully parseable as timestamps — leave as-is

    duration_hrs = (ttr - frt).dt.total_seconds() / 3600.0
    valid_before = duration_hrs.notna().sum()
    duration_hrs = duration_hrs.where(duration_hrs >= 0)  # drop impossible negatives
    valid_after = duration_hrs.notna().sum()
    if valid_before > 0:
        dropped = valid_before - valid_after
        print(f"[info] Derived 'Resolution Duration (hrs)' from timestamps: "
              f"{valid_after} usable rows ({dropped} dropped for negative/impossible duration).")

    df["Resolution Duration (hrs)"] = duration_hrs
    return df


df = pd.read_csv(DATA_PATH)
df = _add_derived_resolution_duration(df)
COLUMNS = list(df.columns)

# ---------- Load prediction model (optional, if trained) ----------
ml_bundle = None
if os.path.exists(MODEL_PATH):
    ml_bundle = joblib.load(MODEL_PATH)

# ---------- Chroma RAG over ticket text ----------
# ---------- Chroma RAG over ticket text (TF-IDF embeddings — no model download) ----------
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbeddingFunction:
    """Lightweight embedding function so Chroma doesn't need to download a
    ~80MB ONNX model over the network. Fit once on the ticket corpus."""

    def __init__(self, corpus: list[str]):
        self.vectorizer = TfidfVectorizer(max_features=384, stop_words="english")
        self.vectorizer.fit(corpus)

    def name(self) -> str:
        return "tfidf-local"

    def __call__(self, input: list[str]):
        vecs = self.vectorizer.transform(input).toarray()
        return [v.astype(float).tolist() for v in vecs]

    def embed_documents(self, input: list[str]):
        return self(input)

    def embed_query(self, input):
        texts = [input] if isinstance(input, str) else input
        return self(texts)[0] if isinstance(input, str) else self(texts)


def _safe_str(val):
    return "" if pd.isna(val) else str(val)


_corpus = [
    f"{_safe_str(row.get('Ticket Subject'))}. {_safe_str(row.get('Ticket Description'))}"
    + (f" Resolution: {_safe_str(row.get('Resolution'))}" if _safe_str(row.get("Resolution")) else "")
    for _, row in df.iterrows()
]
_embed_fn = TfidfEmbeddingFunction(_corpus)

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("tickets", embedding_function=_embed_fn)

RAG_READY = True
if collection.count() == 0:
    try:
        docs, ids, metas = [], [], []
        for _, row in df.iterrows():
            text = f"{row['Ticket Subject']}. {row['Ticket Description']} Resolution: {row['Resolution']}"
            docs.append(text)
            ids.append(str(row["Ticket ID"]))
            metas.append({
                "priority": str(row["Ticket Priority"]),
                "status": str(row["Ticket Status"]),
                "product": str(row["Product Purchased"]),
            })
        BATCH = 200
        for i in range(0, len(docs), BATCH):
            collection.add(
                documents=docs[i:i + BATCH],
                ids=ids[i:i + BATCH],
                metadatas=metas[i:i + BATCH],
            )
    except Exception as e:
        RAG_READY = False
        print(f"[warn] Chroma indexing failed ({e}). 'lookup' intent will fall back to keyword search.")


class QueryRequest(BaseModel):
    query: str


def call_groq(system_prompt: str, user_prompt: str) -> str:
    if client is None:
        raise RuntimeError("GROQ_API_KEY not set")
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        timeout=20,  # without this, a stalled Groq response hangs the whole
                     # request indefinitely — fetch() has no default timeout
                     # either, so the frontend spinner would never resolve
    )
    return resp.choices[0].message.content.strip()


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


STAT_SYSTEM = f"""You write a single pandas expression to answer a question about a dataframe called df.
Columns available: {COLUMNS}
Rules:
- Only use pandas/numpy operations on df. No imports, no file/os access, no assignments beyond the final expression.
- The expression must evaluate to a single scalar (number, string, or short list).
- Do not use exec, eval, __, open, import.
Respond with ONLY compact JSON: {{"expression": "<python expression using df>", "explanation": "<one short sentence>"}}
"""

CHART_SYSTEM = f"""You design a chart spec to answer a question about a support ticket dataframe called df.
Columns available: {COLUMNS}
Respond with ONLY compact JSON:
{{
  "groupby": "<column name to group by>",
  "agg_column": "<column to aggregate, or null if using count>",
  "agg_func": "count" | "mean" | "sum",
  "chart_type": "bar" | "line" | "pie",
  "title": "<short chart title>"
}}
Choose groupby/agg columns strictly from the dataset columns above.
"""

def _build_predict_system_prompt():
    if ml_bundle is None:
        return "Prediction model not available."

    categorical = ml_bundle.get("categorical", [])
    numeric = ml_bundle.get("numeric", [])
    lines = []
    for c in categorical:
        options = list(ml_bundle["encoders"][c].classes_)
        lines.append(f'- {c} (one of: {", ".join(options)}; pick the closest match)')
    for n in numeric:
        lines.append(f"- {n} (number; make a reasonable estimate if not mentioned)")

    keys_json = ", ".join(f'"{f}": <value>' for f in categorical + numeric)

    return (
        "You extract hypothetical ticket attributes from a user's question for a prediction model.\n"
        "Fields needed (use sensible defaults if not mentioned):\n"
        + "\n".join(lines)
        + "\nRespond with ONLY compact JSON with exactly these keys:\n"
        + "{" + keys_json + "}\n"
    )


PREDICT_SYSTEM = _build_predict_system_prompt()

FORBIDDEN = ["import", "exec", "eval", "__", "open(", "os.", "sys.", "subprocess"]


def safe_eval_expression(expr: str):
    if any(bad in expr for bad in FORBIDDEN):
        raise ValueError("Unsafe expression rejected.")
    allowed_names = {"df": df, "pd": pd, "np": np}
    return eval(expr, {"__builtins__": {}}, allowed_names)


import re as _re_intent

# ---------- Classical NLP intent classification (no LLM call) ----------
# Small, closed intent set -> keyword/pattern matching is the right tool here,
# not an LLM call. Order of checks matters: predict > numeric > chart > lookup,
# since a predict question can also contain numeric-sounding words ("rating").

PREDICT_PATTERNS = [
    r"\bpredict\b", r"\bestimate\b", r"\bwhat would\b", r"\bhypothetical\b",
    r"\blikely (rating|satisfaction)\b", r"\bwould (get|receive|have)\b",
]

NUMERIC_PATTERNS = [
    r"\bhow many\b", r"\bcount\b", r"\bnumber of\b", r"\btotal\b",
    r"\baverage\b", r"\bavg\b", r"\bmean\b", r"\bmedian\b",
    r"\bpercent(age)?\b", r"\bproportion\b", r"\bratio\b",
    r"\b(highest|lowest|max(imum)?|min(imum)?)\b", r"\bsum of\b",
]

CHART_PATTERNS = [
    r"\bby\b.{0,15}\b(channel|product|priority|type|status|gender|subject)s?\b",
    r"\bcompare\b", r"\bbreakdown\b", r"\bbreak down\b", r"\bdistribution\b",
    r"\btrend\b", r"\bover time\b", r"\bacross\b", r"\bversus\b", r"\bvs\.?\b",
    r"\bchart\b", r"\bgraph\b", r"\bplot\b", r"\bshow me\b.*\b(by|per)\b",
]

LOOKUP_PATTERNS = [
    r"\bwhat (issues?|problems?|complaints?)\b", r"\bwhat.*(reporting|saying|mentioning)\b",
    r"\bdescribe\b", r"\bexplain\b", r"\btell me about\b", r"\bfeedback\b",
    r"\babout\b.*\b(battery|login|refund|delivery|payment|setup|warranty)\b",
]


def _matches_any(patterns, text):
    return any(_re_intent.search(p, text) for p in patterns)


def classify_intent_classical(q: str) -> dict:
    text = q.lower().strip()

    is_predict = _matches_any(PREDICT_PATTERNS, text)
    needs_numeric = is_predict or _matches_any(NUMERIC_PATTERNS, text)
    needs_chart = _matches_any(CHART_PATTERNS, text)
    is_lookup = _matches_any(LOOKUP_PATTERNS, text)

    # Fallback: if nothing matched at all, default to lookup so the question
    # still gets a grounded text answer via RAG rather than an empty response.
    if not (needs_numeric or needs_chart or is_lookup):
        is_lookup = True

    return {
        "needs_numeric": needs_numeric,
        "needs_chart": needs_chart,
        "is_predict": is_predict,
        "is_lookup": is_lookup,
    }


@app.post("/api/query")
def handle_query(req: QueryRequest):
    q = req.query.strip()
    if not q:
        return {"text": "Please enter a question.", "numeric": None, "chart": None}

    flags = classify_intent_classical(q)

    numeric_result = None
    chart_result = None
    lookup_context = None
    errors = []

    try:
        if flags.get("is_predict"):
            numeric_result = compute_predict(q)
        elif flags.get("needs_numeric"):
            numeric_result = compute_stat(q)
    except Exception as e:
        traceback.print_exc()
        errors.append(f"numeric part failed: {e}")

    try:
        if flags.get("needs_chart"):
            chart_result = compute_chart(q)
    except Exception as e:
        traceback.print_exc()
        errors.append(f"chart part failed: {e}")

    try:
        if flags.get("is_lookup") or (not numeric_result and not chart_result):
            lookup_context = retrieve_context(q)
    except Exception as e:
        traceback.print_exc()

    text_answer = synthesize_text(q, numeric_result, chart_result, lookup_context, errors)

    return {
        "text": text_answer,
        "numeric": numeric_result,
        "chart": chart_result,
    }


def compute_stat(q: str):
    raw = call_groq(STAT_SYSTEM, q)
    spec = extract_json(raw)
    expr = spec["expression"]
    result = safe_eval_expression(expr)

    if isinstance(result, (pd.Series, pd.DataFrame)):
        result = result.to_dict()
    elif isinstance(result, (np.integer,)):
        result = int(result)
    elif isinstance(result, (np.floating,)):
        result = round(float(result), 2)

    return {
        "value": result,
        "explanation": spec.get("explanation", ""),
        "query_used": expr,
        "kind": "stat",
    }


def compute_chart(q: str):
    raw = call_groq(CHART_SYSTEM, q)
    spec = extract_json(raw)
    groupby = spec["groupby"]
    agg_col = spec.get("agg_column")
    agg_func = spec.get("agg_func", "count")
    title = spec.get("title", q)

    if groupby not in df.columns:
        raise ValueError(f"Unknown column: {groupby}")

    if agg_func == "count" or not agg_col:
        grouped = df.groupby(groupby).size().sort_values(ascending=False)
    else:
        if agg_col not in df.columns:
            raise ValueError(f"Unknown column: {agg_col}")
        grouped = df.groupby(groupby)[agg_col].agg(agg_func).sort_values(ascending=False)

    grouped = grouped.head(12)
    labels = [str(x) for x in grouped.index.tolist()]
    values = [round(float(v), 2) for v in grouped.values.tolist()]

    return {
        "chart_type": spec.get("chart_type", "bar"),
        "title": title,
        "labels": labels,
        "values": values,
    }


def retrieve_context(q: str):
    if RAG_READY:
        results = collection.query(query_texts=[q], n_results=5)
        docs = results.get("documents", [[]])[0]
    else:
        words = [w.lower() for w in re.findall(r"[a-zA-Z]{4,}", q)]
        desc = df["Ticket Description"].fillna("").str.lower()
        mask = desc.apply(lambda t: any(w in t for w in words))
        subset = df[mask].head(5)
        docs = [
            f"{_safe_str(r.get('Ticket Subject'))}. {_safe_str(r.get('Ticket Description'))}"
            + (f" Resolution: {_safe_str(r.get('Resolution'))}" if _safe_str(r.get("Resolution")) else "")
            for _, r in subset.iterrows()
        ]
    return docs


def compute_predict(q: str):
    if ml_bundle is None:
        return None

    raw = call_groq(PREDICT_SYSTEM, q)
    attrs = extract_json(raw)

    model = ml_bundle["model"]
    encoders = ml_bundle["encoders"]
    features = ml_bundle["features"]

    row = {}
    for f in features:
        val = attrs.get(f)
        if f in encoders:
            le = encoders[f]
            val = str(val) if val in le.classes_ else le.classes_[0]
            val = le.transform([val])[0]
        row[f] = val

    X = pd.DataFrame([row])[features]
    pred = int(model.predict(X)[0])

    return {
        "value": pred,
        "explanation": f"Predicted satisfaction rating (1-5) based on: {attrs}",
        "query_used": None,
        "kind": "predict",
    }


def synthesize_text(q, numeric_result, chart_result, lookup_context, errors):
    """Always produces the text-tile answer. Grounded in whatever was actually
    computed — the model is told the exact numbers/labels so it can't invent
    different ones."""
    facts = []
    if numeric_result:
        facts.append(f"Computed numeric answer: {numeric_result['value']} "
                      f"({numeric_result.get('explanation', '')})")
    if chart_result:
        facts.append(f"Computed chart '{chart_result['title']}': "
                      f"{dict(zip(chart_result['labels'], chart_result['values']))}")
    if lookup_context:
        joined = "\n---\n".join(lookup_context)
        facts.append(f"Relevant ticket excerpts:\n{joined}")
    if errors:
        facts.append(f"Note: {'; '.join(errors)}")

    if not facts:
        return "I couldn't find enough information in the dataset to answer that."

    system = (
        "You answer a question about a customer support ticket dataset. "
        "You are given facts that were already computed or retrieved — use ONLY "
        "these facts, do not invent numbers. Write a concise, natural 2-4 sentence answer. "
        "If a computed numeric answer or chart data is given, reference it in your own words "
        "(don't just repeat raw numbers robotically)."
    )
    user_prompt = f"Question: {q}\n\n" + "\n\n".join(facts)
    try:
        return call_groq(system, user_prompt)
    except Exception as e:
        # Fallback: if Groq synthesis fails but we have a numeric/chart result, still show something.
        if numeric_result:
            return numeric_result.get("explanation") or f"Answer: {numeric_result['value']}"
        if chart_result:
            return f"Here's the breakdown for {chart_result['title']}."
        return f"Couldn't generate a text summary: {e}"


@app.get("/api/health")
def health():
    return {"status": "ok", "rows": len(df), "groq_configured": client is not None, "model_trained": ml_bundle is not None}


def _first_present(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_numeric_mean(df, col_name, round_to=1):
    if col_name is None or col_name not in df.columns:
        return None
    coerced = pd.to_numeric(df[col_name], errors="coerce").dropna()
    # If almost nothing parsed as a number, this column probably isn't
    # actually numeric in this dataset variant (e.g. it's a raw timestamp
    # string) — better to report unavailable than a meaningless average.
    if len(coerced) == 0 or coerced.notna().mean() < (0.05 if len(df) else 0):
        return None
    return round(float(coerced.mean()), round_to)


@app.get("/api/stats")
def stats():
    """Quick dashboard KPIs — pure pandas, no LLM call, so this loads instantly.
    Schema-agnostic: checks each column exists and is actually numeric before
    computing on it, since exact column names/types vary across dataset
    variants (e.g. the real Kaggle CSV vs. a regenerated one)."""
    total = len(df)

    status_col = _first_present(df, ["Ticket Status", "Status"])
    open_tickets = int((df[status_col] != "Closed").sum()) if status_col else None

    resolution_col = _first_present(df, ["Resolution Duration (hrs)", "Time to Resolution (hrs)", "Time to Resolution"])
    avg_resolution_val = _safe_numeric_mean(df, resolution_col, round_to=1)

    csat_col = _first_present(df, ["Customer Satisfaction Rating"])
    avg_csat_val = _safe_numeric_mean(df, csat_col, round_to=2)

    priority_col = _first_present(df, ["Ticket Priority", "Priority"])
    critical = int((df[priority_col] == "Critical").sum()) if priority_col else None

    return {
        "total_tickets": total,
        "open_tickets": open_tickets,
        "avg_resolution_hrs": avg_resolution_val,
        "avg_satisfaction": avg_csat_val,
        "critical_tickets": critical,
    }

