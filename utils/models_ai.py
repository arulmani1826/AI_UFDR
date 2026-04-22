import os
import time
import json
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict, Counter
import re
from groq import Groq
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


# ===== CHATBOT MODULE =====
client = None
model_id = None

# Token safety constants (conservative estimates)
TOKENS_PER_CHAR = 0.25  # Average across most content
MAX_REQUEST_TOKENS = 12000  # Increased for richer context
CONTEXT_MAX_CHARS = 5000  # Max chars for context inclusion
HISTORY_MAX_CHARS = 1500  # Max chars for chat history

def setup_model(api_key=None, model_name=None):
    global client, model_id
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        client = None
        model_id = None
        return None
    model_id = model_name or os.getenv("GROQ_MODEL") or "openai/gpt-oss-20b"
    client = Groq(api_key=api_key)
    return client

def _truncate_text(text, max_chars):
    """Safely truncate text to max characters, preserving structure."""
    if not text or len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    return truncated + "\n[... truncated for length ...]"

def _estimate_tokens(text):
    """Estimate token count (conservative)."""
    return int(len(str(text)) * TOKENS_PER_CHAR)

def _call_model(prompt):
    if client is None or not model_id:
        return "Model not configured. Set GROQ_API_KEY in your .env file."
    last_error = None
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}".strip()
            time.sleep(2)
    return f"Error: {last_error or 'Unknown error from model'}"

def ask_chat(question, chat_history, system_prompt=None, context=None):
    if system_prompt is None:
        system_prompt = "You are a helpful NLP assistant. Be concise and practical."
    
    # Truncate chat history to prevent token overflow
    history_str = str(chat_history) if chat_history else "None"
    history_str = _truncate_text(history_str, HISTORY_MAX_CHARS)
    
    # Format context safely
    context_str = "None"
    if context:
        if isinstance(context, dict):
            # For dict context, use a compact representation
            context_str = json.dumps(context, separators=(',', ':'))
        else:
            context_str = str(context)
        context_str = _truncate_text(context_str, CONTEXT_MAX_CHARS)
    
    prompt = f"""{system_prompt}

Context:
{context_str}

History:
{history_str}

Question:
{question}
"""
    
    # Verify token estimate won't exceed limit
    estimated_tokens = _estimate_tokens(prompt)
    if estimated_tokens > MAX_REQUEST_TOKENS:
        return f"Request too large ({estimated_tokens} tokens, limit {MAX_REQUEST_TOKENS}). Please use shorter questions or smaller context."
    
    return _call_model(prompt)

def ask_file_question(question, context, chat_history):
    system_prompt = "You are a forensic AI assistant for UFDR device data."
    return ask_chat(question, chat_history, system_prompt=system_prompt, context=context)

def detect_intent(question, columns):
    if client is None or not model_id:
        return None, "Model not configured. Set GROQ_API_KEY in your .env file."
    schema_hint = {
        "intent": "nearby_contacts|row_count|list_columns|summary|top_n|unique_values|filter_contains",
        "column": "string or null",
        "n": "integer or null",
        "radius_km": "number or null",
        "lat": "number or null",
        "lon": "number or null",
        "value": "string or null"
    }
    system_prompt = (
        "You extract a structured intent for local CSV analysis. "
        "Return ONLY valid JSON matching this schema: " + json.dumps(schema_hint)
    )
    prompt = (
        f"{system_prompt}\n\n"
        f"Available columns: {columns}\n"
        f"User question: {question}\n"
        "Return JSON only."
    )
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content
        if not text:
            return None, "Empty response from model."
        data = json.loads(text)
        return data, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ===== MODELS MODULE =====
def tfidf_embeddings(
    texts,
    max_features: int = 5000,
    min_df: int = 2,
    max_df: float = 0.95,
    ngram_range: tuple = (1, 2),
):
    if texts is None or len(texts) == 0:
        return np.zeros((0, 0), dtype=float)
    safe_texts = ["" if t is None else str(t) for t in texts]
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        lowercase=True,
        strip_accents="unicode",
    )
    X = vectorizer.fit_transform(safe_texts)
    if X.shape[1] == 0:
        return np.zeros((len(safe_texts), 0), dtype=float)
    return X.toarray()


def run_call_log_isolation(df: pd.DataFrame, max_samples: int = 5000):
    use_cols = [c for c in ["duration_sec", "is_night_call", "call_type", "transaction_status"] if c in df.columns]
    if not use_cols:
        return df, None
    X = pd.get_dummies(df[use_cols], dummy_na=True)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    if len(X) > max_samples:
        sample_idx = np.random.RandomState(42).choice(len(X), size=max_samples, replace=False)
        X_fit = X.iloc[sample_idx]
    else:
        X_fit = X
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit)
    X_scaled = scaler.transform(X)
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.03,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_fit_scaled)
    scores = iso.decision_function(X_scaled)
    df = df.copy()
    df["anomaly_score"] = scores
    df["is_anomaly"] = scores < np.percentile(scores, 3)
    return df, iso


def run_message_clustering(embeddings, n_clusters: int = 4):
    if len(embeddings) < 2:
        labels = np.zeros(len(embeddings), dtype=int)
        coords = np.zeros((len(embeddings), 2))
        return labels, coords, None
    unique_count = np.unique(np.asarray(embeddings), axis=0).shape[0]
    if unique_count < 2:
        labels = np.zeros(len(embeddings), dtype=int)
        coords = np.zeros((len(embeddings), 2))
        return labels, coords, None
    n_clusters = min(n_clusters, unique_count)
    if len(embeddings) < n_clusters:
        n_clusters = max(2, min(3, len(embeddings)))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        labels = kmeans.fit_predict(embeddings)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    return labels, coords, kmeans


def run_gps_models(df: pd.DataFrame, eps: float = 0.6, min_samples: int = 20):
    numeric = df.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(numeric.median())
    if numeric.empty:
        return df, None, None, None
    scaler = StandardScaler()
    X = scaler.fit_transform(numeric)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = db.fit_predict(coords)
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X)
    scores = iso.decision_function(X)
    df = df.copy()
    df["dbscan_cluster"] = clusters
    df["gps_anomaly_score"] = scores
    df["gps_is_anomaly"] = scores < np.percentile(scores, 2)
    return df, coords, db, iso


# ===== GRAPH ANALYZER MODULE =====
def detect_message_suspicion(df: pd.DataFrame) -> pd.DataFrame:
    """Detect suspicious messages based on content, patterns, and metadata."""
    df = df.copy()
    
    # Primary suspicious keywords — high confidence forensic indicators
    high_weight_keywords = [
        'drug', 'sell', 'buy', 'deal', 'money', 'transfer', 'bank',
        'password', 'otp', 'crypto', 'bitcoin', 'payment',
        'delete', 'destroy', 'hide', 'erase', 'clean',
        'midnight', 'secret', 'private', 'anonymous',
        'urgent', 'immediately', 'urgently', 'now',
        'drop', 'pickup', 'deliver', 'package', 'shipment',
        'weapon', 'gun', 'kill', 'threat', 'blackmail',
        'hack', 'breach', 'exploit', 'phishing',
    ]
    
    # Secondary keywords — moderate suspicion
    medium_weight_keywords = [
        'price', 'cash', 'account', 'verify', 'confirm',
        'click', 'link', 'prize', 'win', 'claim', 'congratulations',
        'debit', 'credit', 'transaction',
        'meet', 'location', 'spot', 'place',
        'call me', 'don\'t tell', 'no one',
    ]
    
    df["suspicion_score"] = 0
    
    if "message_clean" in df.columns:
        message_text = df["message_clean"].astype(str).str.lower()
        
        # Check for high-weight suspicious keywords
        for keyword in high_weight_keywords:
            df["suspicion_score"] += message_text.str.contains(keyword, regex=False).astype(int) * 5
        
        # Check for medium-weight keywords
        for keyword in medium_weight_keywords:
            df["suspicion_score"] += message_text.str.contains(keyword, regex=False).astype(int) * 3
        
        # Check for unusual patterns
        df["suspicion_score"] += message_text.str.contains(r'\d{6,}').astype(int) * 3  # Long numbers
        df["suspicion_score"] += message_text.str.contains(r'[A-Z]{5,}').astype(int) * 2  # ALL CAPS sequences
        df["suspicion_score"] += message_text.str.contains(r'http|www|\.(com|org|net)').astype(int) * 10  # URLs
        
        # Check for excessive punctuation
        df["special_char_ratio"] = message_text.apply(
            lambda x: len(re.findall(r'[!?#@*$%&]', str(x))) / max(len(str(x)), 1)
        )
        df["suspicion_score"] += (df["special_char_ratio"] > 0.2).astype(int) * 8
    
    if "char_len" in df.columns:
        # Very short messages might be automated
        df["suspicion_score"] += (df["char_len"] < 5).astype(int) * 2
    
    df["is_suspicious_message"] = df["suspicion_score"] >= 5
    return df


def detect_call_suspicion(df: pd.DataFrame) -> pd.DataFrame:
    """Detect suspicious call patterns."""
    df = df.copy()
    
    if "suspicious_call" not in df.columns:
        df["suspicious_call"] = False
    
    if "suspicious_reason" not in df.columns:
        df["suspicious_reason"] = ""
    
    df["suspicion_score"] = 0
    
    # Duration-based suspicion
    if "duration_sec" in df.columns:
        df.loc[df["duration_sec"] >= 600, "suspicion_score"] += 20  # Long calls
        df.loc[df["duration_sec"] == 0, "suspicion_score"] += 15  # Zero duration (dropped/missed)
    
    # Night calls suspicion
    if "is_night_call" in df.columns:
        df.loc[df["is_night_call"] == 1, "suspicion_score"] += 10
    
    # Call type suspicion
    if "call_type" in df.columns:
        ct = df["call_type"].astype(str).str.lower()
        df.loc[ct == "missed", "suspicion_score"] += 5
        df.loc[ct == "rejected", "suspicion_score"] += 8
    
    # Frequency-based: calls to same number in short timespan
    if "number" in df.columns and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        
        # Group by number and time gaps
        for number in df["number"].unique():
            mask = df["number"] == number
            if mask.sum() > 5:  # If more than 5 calls to same number
                df.loc[mask, "suspicion_score"] += 5
    
    df["is_suspicious_call"] = df["suspicion_score"] > 0
    return df


def build_network_graph(messages_df=None, calls_df=None, contacts_df=None):
    """Build network graph data from messages, calls, and contacts."""
    nodes = {}
    edges = []
    edge_map = {}
    node_id_counter = 0
    node_map = {}  # Map (type, identifier) to node ID
    
    # Add contact nodes
    if contacts_df is not None and not contacts_df.empty:
        for idx, row in contacts_df.iterrows():
            phone = str(row.get("phone", "")) if "phone" in row else ""
            name = str(row.get("name", "Unknown")) if "name" in row else "Unknown"
            
            if phone:
                key = ("contact", phone)
                node_id = node_id_counter
                node_id_counter += 1
                node_map[key] = node_id
                
                nodes[node_id] = {
                    "id": node_id,
                    "label": name,
                    "type": "contact",
                    "phone": phone,
                    "size": 25,
                    "color": "#FF6B6B",
                    "suspicion_score": row.get("suspicion_score", 0)
                }
    
    # Add message connections
    if messages_df is not None and not messages_df.empty:
        for idx, row in messages_df.iterrows():
            sender = str(row.get("sender", "Unknown")) if "sender" in row else "Unknown"
            receiver = str(row.get("receiver", "Unknown")) if "receiver" in row else "Unknown"
            
            # Create nodes for sender/receiver if they don't exist and aren't in contacts
            for party, party_type in [(sender, "sender"), (receiver, "receiver")]:
                if party:
                    key = ("participant", party)
                    if key not in node_map:
                        node_id = node_id_counter
                        node_id_counter += 1
                        node_map[key] = node_id
                        nodes[node_id] = {
                            "id": node_id,
                            "label": party,
                            "type": party_type,
                            "size": 20,
                            "color": "#4ECDC4",
                            "suspicion_score": row.get("suspicion_score", 0)
                        }
            
            # Create edge
            sender_id = node_map.get(("participant", sender))
            receiver_id = node_map.get(("participant", receiver))
            
            if sender_id is not None and receiver_id is not None:
                key = (sender_id, receiver_id, "message")
                suspicious = row.get("is_suspicious_message", False)
                if key not in edge_map:
                    edge_map[key] = {
                        "from": sender_id,
                        "to": receiver_id,
                        "type": "message",
                        "label": "MSG",
                        "color": "#95E1D3",
                        "suspicious": suspicious,
                        "count": 1
                    }
                else:
                    edge_map[key]["count"] += 1
                    if suspicious:
                        edge_map[key]["suspicious"] = True
    
    # Add call connections
    if calls_df is not None and not calls_df.empty:
        for idx, row in calls_df.iterrows():
            caller = str(row.get("caller_id", "")) or str(row.get("number", "Unknown"))
            receiver = str(row.get("receiver_id", "Unknown"))
            
            # Create nodes for participants
            for party, party_type in [(caller, "caller"), (receiver, "callee")]:
                if party:
                    key = ("participant", party)
                    if key not in node_map:
                        node_id = node_id_counter
                        node_id_counter += 1
                        node_map[key] = node_id
                        nodes[node_id] = {
                            "id": node_id,
                            "label": party,
                            "type": party_type,
                            "size": 20,
                            "color": "#FFE66D",
                            "suspicion_score": row.get("suspicion_score", 0)
                        }
            
            # Create edge
            caller_id = node_map.get(("participant", caller))
            receiver_id = node_map.get(("participant", receiver))
            
            if caller_id is not None and receiver_id is not None:
                key = (caller_id, receiver_id, "call")
                suspicious = row.get("suspicious_call", False)
                if key not in edge_map:
                    edge_map[key] = {
                        "from": caller_id,
                        "to": receiver_id,
                        "type": "call",
                        "label": "CALL",
                        "color": "#FF6B6B" if suspicious else "#FFE66D",
                        "suspicious": suspicious,
                        "count": 1
                    }
                else:
                    edge_map[key]["count"] += 1
                    if suspicious:
                        edge_map[key]["suspicious"] = True
                        edge_map[key]["color"] = "#FF6B6B"

    edges = list(edge_map.values())
    
    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "contact_nodes": sum(1 for n in nodes.values() if n.get("type") == "contact"),
            "participant_nodes": sum(1 for n in nodes.values() if n.get("type") in ["sender", "receiver", "caller", "callee"]),
            "message_edges": sum(1 for e in edges if e.get("type") == "message"),
            "call_edges": sum(1 for e in edges if e.get("type") == "call"),
            "suspicious_nodes": sum(1 for n in nodes.values() if n.get("suspicion_score", 0) > 0),
            "suspicious_edges": sum(1 for e in edges if e.get("suspicious", False))
        }
    }


def generate_suspicious_activities_summary(messages_df=None, calls_df=None, contacts_df=None):
    """Generate summary of suspicious activities."""
    suspicious_items = []
    
    # Suspicious messages
    if messages_df is not None and not messages_df.empty:
        if "is_suspicious_message" in messages_df.columns:
            sus_msgs = messages_df[messages_df["is_suspicious_message"] == True]
            for idx, msg in sus_msgs.iterrows():
                suspicious_items.append({
                    "type": "message",
                    "severity": min(5, max(1, int(msg.get("suspicion_score", 0) / 10))),
                    "timestamp": str(msg.get("timestamp", "")),
                    "description": f"Suspicious message from {msg.get('sender', 'Unknown')} to {msg.get('receiver', 'Unknown')}",
                    "content": str(msg.get("message", ""))[:100],
                    "suspicion_score": msg.get("suspicion_score", 0),
                    "keywords": msg.get("message", "").split()[:5]
                })
    
    # Suspicious calls
    if calls_df is not None and not calls_df.empty:
        if "is_suspicious_call" in calls_df.columns:
            sus_calls = calls_df[calls_df["is_suspicious_call"] == True]
            for idx, call in sus_calls.iterrows():
                duration = call.get("duration_sec", 0)
                timestamp = str(call.get("timestamp", "") or call.get("start_time", ""))
                suspicious_items.append({
                    "type": "call",
                    "severity": min(5, max(1, int(call.get("suspicion_score", 0) / 10))),
                    "timestamp": timestamp,
                    "description": f"Suspicious call with {call.get('number', 'Unknown')} for {duration}s",
                    "duration": duration,
                    "suspicion_score": call.get("suspicion_score", 0),
                    "reason": call.get("suspicious_reason", "")
                })
    
    # Suspicious contacts
    if contacts_df is not None and not contacts_df.empty:
        if "suspicion_score" in contacts_df.columns:
            sus_contacts = contacts_df[contacts_df["suspicion_score"] > 0]
            for idx, contact in sus_contacts.iterrows():
                suspicious_items.append({
                    "type": "contact",
                    "severity": min(5, max(1, int(contact.get("suspicion_score", 0) / 10))),
                    "description": f"Suspicious contact: {contact.get('name', 'Unknown')}",
                    "phone": contact.get("phone", ""),
                    "suspicion_score": contact.get("suspicion_score", 0),
                    "flags": [f for f in contact.index if f.startswith("flag_") and contact[f] == True]
                })
    
    # Sort by severity and then by suspicion score
    suspicious_items.sort(
        key=lambda x: (x.get("severity", 0), x.get("suspicion_score", 0)), 
        reverse=True
    )
    
    return suspicious_items[:50]  # Return top 50 suspicious activities


def calculate_overall_risk_score(messages_df=None, calls_df=None, contacts_df=None):
    """Calculate overall risk score from all data sources."""
    risk_score = 0
    risk_breakdown = {
        "messages": 0,
        "calls": 0,
        "contacts": 0,
        "total": 0
    }
    
    # Messages risk
    if messages_df is not None and not messages_df.empty:
        if "is_suspicious_message" in messages_df.columns:
            sus_count = (messages_df["is_suspicious_message"] == True).sum()
            avg_score = messages_df.get("suspicion_score", pd.Series()).mean() if "suspicion_score" in messages_df.columns else 0
            msg_risk = min(30, sus_count * 2 + int(avg_score * 0.5))
            risk_breakdown["messages"] = msg_risk
            risk_score += msg_risk
    
    # Calls risk
    if calls_df is not None and not calls_df.empty:
        if "is_suspicious_call" in calls_df.columns:
            sus_count = (calls_df["is_suspicious_call"] == True).sum()
            avg_score = calls_df.get("suspicion_score", pd.Series()).mean() if "suspicion_score" in calls_df.columns else 0
            call_risk = min(30, sus_count * 3 + int(avg_score * 0.5))
            risk_breakdown["calls"] = call_risk
            risk_score += call_risk
    
    # Contacts risk
    if contacts_df is not None and not contacts_df.empty:
        flagged_count = (contacts_df.get("rule_based_flag", pd.Series()) == True).sum()
        avg_score = contacts_df.get("suspicion_score", pd.Series()).mean() if "suspicion_score" in contacts_df.columns else 0
        contact_risk = min(40, flagged_count * 5 + int(avg_score * 0.5))
        risk_breakdown["contacts"] = contact_risk
        risk_score += contact_risk
    
    risk_breakdown["total"] = min(100, risk_score)
    return risk_breakdown


def analyze_interactions(messages_df=None, calls_df=None):
    """Analyze interaction patterns and frequencies."""
    interaction_stats = {
        "message_interactions": {},
        "call_interactions": {},
        "top_contacts": [],
        "interaction_patterns": []
    }
    
    # Message interactions
    if messages_df is not None and not messages_df.empty:
        if "sender" in messages_df.columns and "receiver" in messages_df.columns:
            interactions = defaultdict(int)
            for idx, row in messages_df.iterrows():
                sender = str(row.get("sender", "Unknown"))
                receiver = str(row.get("receiver", "Unknown"))
                key = f"{sender}↔{receiver}"
                interactions[key] += 1
            
            interaction_stats["message_interactions"] = dict(sorted(
                interactions.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10])
    
    # Call interactions
    if calls_df is not None and not calls_df.empty:
        caller_counts = defaultdict(int)
        if "number" in calls_df.columns:
            caller_counts = dict(calls_df["number"].value_counts().head(10))
        interaction_stats["call_interactions"] = caller_counts
        interaction_stats["top_contacts"] = list(caller_counts.keys())
    
    return interaction_stats
