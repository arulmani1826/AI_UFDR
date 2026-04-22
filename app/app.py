import os
import sys
import threading
import time
import uuid
import hashlib

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import json

import math

import io
import pandas as pd
from flask import Flask, render_template, request, send_file
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from utils.data_processing import (
    detect_dataset_type,
    preprocess_messages,
    preprocess_call_logs,
    preprocess_contacts,
    preprocess_gps,
    rule_based_contact_flags,
    analyze_dataframe
)
from utils.models_ai import (
    tfidf_embeddings,
    run_message_clustering,
    setup_model,
    ask_chat,
    detect_message_suspicion,
    detect_call_suspicion,
    build_network_graph,
    generate_suspicious_activities_summary,
    calculate_overall_risk_score,
    analyze_interactions
)

from dotenv import load_dotenv
load_dotenv()
setup_model()

BASE_DIR = os.path.dirname(__file__)
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MAX_DATASET_ROWS = int(os.getenv("MAX_DATASET_ROWS", "2000"))
ALLOWED_EXTENSIONS = {".json", ".ufdr", ".csv"}

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
UPLOAD_DIR = os.path.join(ROOT_DIR, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
JOBS = {}
CACHE = {}
CACHE_MAX = 3


def _read_csv_with_fallback(uploaded_file):
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc, nrows=MAX_DATASET_ROWS)
        except Exception as e:
            last_err = e
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, encoding="utf-8", errors="replace", nrows=MAX_DATASET_ROWS)


def _read_json_with_fallback(uploaded_file):
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            raw = uploaded_file.read()
            text = raw.decode(enc, errors="replace")
            return json.loads(text)
        except Exception as e:
            last_err = e
    raise last_err


def _append_dict_datasets(data, name_prefix, datasets_out):
    list_keys = [
        "messages",
        "calls",
        "contacts",
        "locations",
        "files",
        "browser_history",
        "installed_apps",
        "accounts",
    ]
    appended = False
    for key in list_keys:
        if key in data and isinstance(data[key], list) and data[key]:
            items = data[key][:MAX_DATASET_ROWS]
            df = pd.DataFrame(items)
            dtype = detect_dataset_type(df, f"{name_prefix}:{key}")
            datasets_out.append({"name": f"{name_prefix}:{key}", "type": dtype, "df": df})
            appended = True
    if not appended:
        df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        dtype = detect_dataset_type(df, name_prefix)
        datasets_out.append({"name": name_prefix, "type": dtype, "df": df})


def _parse_json_payload(data, name_prefix):
    datasets_out = []
    if isinstance(data, list):
        df = pd.DataFrame(data[:MAX_DATASET_ROWS])
        dtype = detect_dataset_type(df, name_prefix)
        datasets_out.append({"name": name_prefix, "type": dtype, "df": df})
    elif isinstance(data, dict):
        if "datasets" in data and isinstance(data["datasets"], list):
            for i, ds in enumerate(data["datasets"], start=1):
                if isinstance(ds, dict):
                    _append_dict_datasets(ds, f"{name_prefix}::dataset_{i:04d}", datasets_out)
                else:
                    df = pd.DataFrame(ds[:MAX_DATASET_ROWS])
                    dtype = detect_dataset_type(df, f"{name_prefix}::dataset_{i:04d}")
                    datasets_out.append({"name": f"{name_prefix}::dataset_{i:04d}", "type": dtype, "df": df})
        else:
            _append_dict_datasets(data, name_prefix, datasets_out)
    return datasets_out


def _build_preview(df):
    if df is None or df.empty:
        return []
    row = df.iloc[0].to_dict()
    preview = []
    for k, v in row.items():
        preview.append((str(k), str(v)))
    return preview


def _build_summary(files_count, datasets):
    summary = [f"Files uploaded: {files_count}", f"Datasets parsed: {len(datasets)}"]
    if datasets:
        first = datasets[0]
        df = first["df"]
        summary.append(f"First dataset: {first['name']}")
        summary.append(f"Rows in first dataset: {len(df)}")
        summary.append(f"Columns in first dataset: {len(df.columns)}")
        summary.append(f"Detected type: {first['type']}")
    return summary

def _process_files(file_paths, filenames):
    datasets = []
    for path, fname in zip(file_paths, filenames):
        try:
            ext = os.path.splitext(fname)[1].lower()
            with open(path, "rb") as fh:
                if ext in [".json", ".ufdr"]:
                    data = _read_json_with_fallback(fh)
                    datasets.extend(_parse_json_payload(data, fname))
                else:
                    df = _read_csv_with_fallback(fh)
                    dtype = detect_dataset_type(df, fname)
                    datasets.append({"name": fname, "type": dtype, "df": df})
        except Exception:
            continue

    dashboard_data = {
        "metrics": {"total_messages": 0, "total_calls": 0, "message_clusters": 0},
        "timeline": [],
        "messages": [],
        "calls": [],
        "contacts": [],
        "locations": [],
        "risk_score": 0,
        "has_data": False,
        "suspicious_activities": [],
        "graph_data": None,
        "risk_breakdown": {},
        "interaction_stats": {}
    }
    
    # Store dataframes for later graph analysis
    stored_dfs = {
        "messages": None,
        "calls": None,
        "contacts": None
    }

    for ds in datasets:
        df = ds["df"]
        dtype = ds["type"]
        
        try:
            if dtype == "messages":
                df = preprocess_messages(df)
                df = detect_message_suspicion(df)
                dashboard_data["metrics"]["total_messages"] += len(df)
                stored_dfs["messages"] = df

                try:
                    text_col = "message_clean" if "message_clean" in df.columns else "message"
                    texts = df[text_col].astype(str).tolist()
                    embeddings = tfidf_embeddings(texts, max_features=2000)
                    if embeddings.shape[1] > 0 and len(embeddings) > 0:
                        labels, coords, _ = run_message_clustering(embeddings, n_clusters=4)
                        df["msg_cluster"] = labels
                        if coords is not None and len(coords) == len(df) and coords.shape[1] == 2:
                            df["msg_x"] = coords[:, 0]
                            df["msg_y"] = coords[:, 1]
                        if len(labels) > 0:
                            dashboard_data["metrics"]["message_clusters"] = int(len(set(labels)))
                except Exception:
                    pass
                
                df_clean = df.where(pd.notna(df), None)
                records = df_clean.to_dict(orient="records")
                dashboard_data["messages"].extend(records)
                
                for row in records:
                    timestamp = row.get("timestamp") or row.get("time") 
                    if timestamp is not None:
                        dashboard_data["timeline"].append({
                            "type": "message",
                            "timestamp": str(timestamp),
                            "desc": f"Message from {row.get('sender', 'unknown')} to {row.get('receiver', 'unknown')} via {row.get('app', 'unknown')}",
                            "suspicious": row.get("is_suspicious_message", False)
                        })

            elif dtype == "call_logs":
                df = preprocess_call_logs(df)
                df = detect_call_suspicion(df)
                dashboard_data["metrics"]["total_calls"] += len(df)
                stored_dfs["calls"] = df
                
                df_clean = df.where(pd.notna(df), None)
                records = df_clean.to_dict(orient="records")
                dashboard_data["calls"].extend(records)
                
                for row in records:
                    timestamp = row.get("timestamp") or row.get("start_time")
                    if timestamp is not None:
                        dur = row.get("duration") or row.get("duration_sec") or 0
                        num = row.get("number", "unknown")
                        call_type = row.get("type", row.get("call_type", "call"))
                        suspicious = row.get("suspicious_call")
                        reason = row.get("suspicious_reason")
                        suffix = ""
                        if suspicious:
                            suffix = f" [suspicious{': ' + reason if reason else ''}]"
                        dashboard_data["timeline"].append({
                            "type": "call",
                            "timestamp": str(timestamp),
                            "desc": f"Call: {call_type} with {num} ({dur}s){suffix}",
                            "suspicious": suspicious
                        })

            elif dtype == "contacts":
                df = preprocess_contacts(df)
                df = rule_based_contact_flags(df)

                stored_dfs["contacts"] = df
                
                df_clean = df.where(pd.notna(df), None)
                dashboard_data["contacts"].extend(df_clean.to_dict(orient="records"))

            elif dtype in ["gps", "unknown"] or "locations" in ds["name"].lower():
                df = preprocess_gps(df)
                df_clean = df.where(pd.notna(df), None)
                records = df_clean.to_dict(orient="records")
                dashboard_data["locations"].extend(records)
                
                for row in records:
                    lat = row.get("lat") or row.get("latitude")
                    lon = row.get("lon") or row.get("longitude")
                    if lat is not None and lon is not None:
                        timestamp = row.get("timestamp")
                        if timestamp is not None:
                            dashboard_data["timeline"].append({
                                "type": "location",
                                "timestamp": str(timestamp),
                                "desc": f"Location: {row.get('address', f'{lat},{lon}')}"
                            })

        except Exception as e:
            print(f"Error processing {dtype}: {e}")
            continue

    if datasets:
        dashboard_data["has_data"] = True
        dashboard_data["timeline"] = sorted(dashboard_data["timeline"], key=lambda x: str(x["timestamp"]), reverse=True)
        
        # Generate suspicious activities and network graph
        dashboard_data["suspicious_activities"] = generate_suspicious_activities_summary(
            stored_dfs["messages"], 
            stored_dfs["calls"], 
            stored_dfs["contacts"]
        )
        
        dashboard_data["graph_data"] = build_network_graph(
            stored_dfs["messages"], 
            stored_dfs["calls"], 
            stored_dfs["contacts"]
        )
        
        dashboard_data["interaction_stats"] = analyze_interactions(
            stored_dfs["messages"], 
            stored_dfs["calls"]
        )
        
        dashboard_data["risk_breakdown"] = calculate_overall_risk_score(
            stored_dfs["messages"], 
            stored_dfs["calls"], 
            stored_dfs["contacts"]
        )
        
        # Cap arrays to prevent browser from freezing with massive datasets
        MAX_ROWS = 1500
        dashboard_data["messages"] = dashboard_data["messages"][:MAX_ROWS]
        dashboard_data["calls"] = dashboard_data["calls"][:MAX_ROWS]
        dashboard_data["contacts"] = dashboard_data["contacts"][:MAX_ROWS]
        dashboard_data["locations"] = dashboard_data["locations"][:MAX_ROWS]
        dashboard_data["timeline"] = dashboard_data["timeline"][:MAX_ROWS]
        dashboard_data["suspicious_activities"] = dashboard_data["suspicious_activities"][:50]
        
        dashboard_data["risk_score"] = dashboard_data["risk_breakdown"].get("total", 0)
        
    def clean_nans(obj):
        if isinstance(obj, list):
            return [clean_nans(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: clean_nans(v) for k, v in obj.items()}
        elif isinstance(obj, float) and math.isnan(obj):
            return None
        return obj
    
    dashboard_data = clean_nans(dashboard_data)
    return dashboard_data

def _run_job(job_id, file_paths, filenames):
    try:
        result = _process_files(file_paths, filenames)
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = result
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)
    finally:
        for p in file_paths:
            try:
                os.remove(p)
            except Exception:
                pass

def _save_and_hash(file_storage, dest_path):
    hasher = hashlib.sha256()
    with open(dest_path, "wb") as out:
        while True:
            chunk = file_storage.stream.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            out.write(chunk)
    file_storage.stream.seek(0)
    return hasher.hexdigest()

def _cache_put(cache_key, result):
    if cache_key in CACHE:
        CACHE.pop(cache_key, None)
    CACHE[cache_key] = {"result": result, "ts": time.time()}
    # prune oldest
    if len(CACHE) > CACHE_MAX:
        oldest = sorted(CACHE.items(), key=lambda x: x[1]["ts"])[0][0]
        CACHE.pop(oldest, None)

@app.get("/")
def index():
    return render_template("index.html", dashboard_data=json.dumps({"has_data": False}))


@app.post("/upload")
def upload():
    files = request.files.getlist("files")
    if not files:
        return render_template("index.html", dashboard_data=json.dumps({"has_data": False}))

    file_paths = []
    filenames = []
    file_hashes = []
    for f in files:
        if not f.filename:
            continue
        name = os.path.basename(f.filename)
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return render_template("index.html", dashboard_data=json.dumps({"has_data": False, "error": "invalid_format"}))
        job_file = f"{uuid.uuid4().hex}_{name}"
        dest = os.path.join(UPLOAD_DIR, job_file)
        file_hash = _save_and_hash(f, dest)
        file_paths.append(dest)
        filenames.append(name)
        file_hashes.append(file_hash)

    cache_key = hashlib.sha256(("|".join(filenames) + "|" + "|".join(file_hashes)).encode("utf-8")).hexdigest()
    if cache_key in CACHE:
        payload = CACHE[cache_key]["result"]
        return render_template("index.html", dashboard_data=json.dumps(payload, default=str))

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "processing", "result": None, "error": None, "started_at": time.time(), "cache_key": cache_key}
    thread = threading.Thread(target=_run_job, args=(job_id, file_paths, filenames), daemon=True)
    thread.start()

    payload = {"has_data": False, "processing": True, "job_id": job_id}
    return render_template("index.html", dashboard_data=json.dumps(payload, default=str))

@app.get("/job/<job_id>")
def job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return json.dumps({"status": "error", "error": "job_not_found"})
    if job["status"] == "done":
        if job.get("cache_key"):
            _cache_put(job["cache_key"], job["result"])
        return json.dumps({"status": "done", "result": job["result"]}, default=str)
    if job["status"] == "error":
        return json.dumps({"status": "error", "error": job.get("error")})
    elapsed = None
    if job.get("started_at"):
        elapsed = max(0, int(time.time() - job["started_at"]))
    return json.dumps({"status": "processing", "elapsed": elapsed})

@app.post("/chat")
def chat():
    data = request.json
    question = data.get("question", "")
    context = data.get("context", {})
    system_prompt = """You are a forensic AI assistant embedded in a UFDR (Universal Forensic Data Report) analysis dashboard.

Your role: Answer the user's question directly and naturally using the ANALYSIS DATA provided below.

Rules:
- Answer exactly what the user asks. Do NOT add unsolicited sections or headers.
- If the user asks a simple question (e.g. "how many calls?"), give a short direct answer.
- If the user asks for analysis or deeper insights, provide detailed analysis.
- When data supports it, suggest that a chart or visual might help, and describe the data in a way that could be visualized (e.g. list top 5 contacts with counts).
- Use markdown formatting: **bold** for emphasis, bullet points for lists, tables when comparing data.
- Never output code blocks or programming snippets.
- Always ground your answers in the actual ANALYSIS DATA provided — do not hallucinate numbers.
- Focus on SUSPICIOUS ACTIVITIES and RISK DATA from the analysis, not raw datasets.
"""
    
    # Create a rich representation of context from ANALYSIS not raw data
    if context and isinstance(context, dict):
        summary_parts = []
        
        # Metrics overview
        if "metrics" in context and isinstance(context["metrics"], dict):
            summary_parts.append("## Dataset Metrics")
            for k, v in context["metrics"].items():
                summary_parts.append(f"- {k}: {v}")
        
        # Risk breakdown from analysis
        if "risk_breakdown" in context and isinstance(context["risk_breakdown"], dict):
            summary_parts.append("\n## Risk Analysis Breakdown")
            risk = context["risk_breakdown"]
            if "total" in risk:
                summary_parts.append(f"- Overall Risk Score: {risk.get('total', 0)}")
            if "message_risk" in risk:
                summary_parts.append(f"- Message Risk: {risk.get('message_risk', 0)}")
            if "call_risk" in risk:
                summary_parts.append(f"- Call Risk: {risk.get('call_risk', 0)}")
            if "contact_risk" in risk:
                summary_parts.append(f"- Contact Risk: {risk.get('contact_risk', 0)}")
        
        # Interaction statistics from analysis
        if "interaction_stats" in context and isinstance(context["interaction_stats"], dict):
            summary_parts.append("\n## Interaction Statistics")
            stats = context["interaction_stats"]
            for key, value in list(stats.items())[:8]:
                summary_parts.append(f"- {key}: {value}")
        
        # SUSPICIOUS MESSAGES from analysis
        if "messages" in context and isinstance(context["messages"], list) and context["messages"]:
            suspicious_msgs = [m for m in context["messages"] if m.get('is_suspicious_message', False)]
            
            if suspicious_msgs:
                summary_parts.append(f"\n## SUSPICIOUS MESSAGES DETECTED ({len(suspicious_msgs)} flagged)")
                for m in suspicious_msgs[:20]:
                    sender = m.get('sender', m.get('from', 'Unknown'))
                    receiver = m.get('receiver', m.get('to', 'Unknown'))
                    text = m.get('text', m.get('message', m.get('body', '(no content)')))
                    app = m.get('app', m.get('platform', 'Unknown'))
                    ts = m.get('timestamp', m.get('time', ''))
                    summary_parts.append(f"- [{ts}] {sender} → {receiver} ({app}): \"{str(text)[:70]}\"")
        
        # Suspicious activities from analysis
        if "suspicious_activities" in context and isinstance(context["suspicious_activities"], list):
            summary_parts.append(f"\n## SUSPICIOUS ACTIVITIES ANALYSIS ({len(context['suspicious_activities'])} items)")
            for act in context["suspicious_activities"][:15]:
                if isinstance(act, dict):
                    desc = act.get('description', act.get('activity', act.get('type', str(act))))
                    activity_type = act.get('type', 'Activity')
                    severity = act.get('severity', 'unknown')
                    summary_parts.append(f"- [{activity_type}] {desc} (Severity: {severity})")
                else:
                    summary_parts.append(f"- {act}")
        
        # Call risk analysis
        if "calls" in context and isinstance(context["calls"], list) and context["calls"]:
            suspicious_calls = [c for c in context["calls"] if c.get('suspicious_call', False)]
            
            if suspicious_calls:
                summary_parts.append(f"\n## SUSPICIOUS CALLS ({len(suspicious_calls)} flagged)")
                for c in suspicious_calls[:10]:
                    caller = c.get('caller', c.get('caller_id', c.get('number', 'Unknown')))
                    receiver = c.get('receiver', c.get('receiver_id', 'Unknown'))
                    dur = c.get('duration_sec', c.get('duration', 0))
                    ctype = c.get('call_type', c.get('type', 'unknown'))
                    ts = c.get('timestamp', c.get('start_time', ''))
                    summary_parts.append(f"- [{ts}] {caller} → {receiver} | Type: {ctype} | Duration: {dur}s")
        
        # Graph analysis data
        if "graph_data" in context and isinstance(context["graph_data"], dict):
            graph = context["graph_data"]
            if "stats" in graph and isinstance(graph["stats"], dict):
                summary_parts.append("\n## Network Analysis Stats")
                for key, value in list(graph["stats"].items())[:5]:
                    summary_parts.append(f"- {key}: {value}")
        
        compact_context = "\n".join(summary_parts[:100])
        system_prompt += f"\n\n--- FORENSIC ANALYSIS DATA ---\n{compact_context}\n--- END ANALYSIS ---"
        
    response = ask_chat(question, chat_history=[], system_prompt=system_prompt)
    return json.dumps({"response": response})

@app.get("/graph-data")
def get_graph_data():
    """Retrieve network graph data for visualization."""
    if not CACHE:
        return json.dumps({"error": "no_data", "nodes": [], "edges": []})
    
    # Get the most recent cached result
    latest = max(CACHE.items(), key=lambda x: x[1]["ts"])[1]["result"]
    if not latest or not latest.get("graph_data"):
        return json.dumps({"error": "no_graph_data", "nodes": [], "edges": []})
    
    return json.dumps(latest["graph_data"], default=str)

@app.get("/suspicious-activities")
def get_suspicious_activities():
    """Retrieve suspicious activities summary."""
    if not CACHE:
        return json.dumps({"error": "no_data", "activities": []})
    
    latest = max(CACHE.items(), key=lambda x: x[1]["ts"])[1]["result"]
    if not latest:
        return json.dumps({"error": "no_data", "activities": []})
    
    activities = latest.get("suspicious_activities", [])
    risk_breakdown = latest.get("risk_breakdown", {})
    
    return json.dumps({
        "activities": activities,
        "risk_breakdown": risk_breakdown,
        "total_activities": len(activities)
    }, default=str)

@app.get("/investigation-data")
def get_investigation_data():
    """Retrieve investigation data for charts."""
    if not CACHE:
        return json.dumps({"error": "no_data"})
    
    latest = max(CACHE.items(), key=lambda x: x[1]["ts"])[1]["result"]
    if not latest:
        return json.dumps({"error": "no_data"})
    
    investigation_data = {
        "metrics": latest.get("metrics", {}),
        "risk_breakdown": latest.get("risk_breakdown", {}),
        "interaction_stats": latest.get("interaction_stats", {}),
        "timeline": latest.get("timeline", [])[:100],
        "graph_stats": latest.get("graph_data", {}).get("stats", {}) if latest.get("graph_data") else {}
    }
    
    return json.dumps(investigation_data, default=str)

@app.post("/report")
def report():
    payload = request.json or {}
    summary = payload.get("summary", {})
    alerts = payload.get("alerts", [])

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "UFDR Forensic Analysis Report")
    y -= 30

    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Risk Score: {summary.get('risk_score', 'N/A')}")
    y -= 20
    c.drawString(40, y, f"Messages: {summary.get('messages', 0)}")
    y -= 15
    c.drawString(40, y, f"Calls: {summary.get('calls', 0)}")
    y -= 15
    c.drawString(40, y, f"Contacts: {summary.get('contacts', 0)}")
    y -= 15
    c.drawString(40, y, f"Locations: {summary.get('locations', 0)}")
    y -= 15
    c.drawString(40, y, f"Alerts: {summary.get('alerts', 0)}")
    y -= 25

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Suspicious Findings")
    y -= 18

    c.setFont("Helvetica", 10)
    if not alerts:
        c.drawString(40, y, "No suspicious messages detected.")
        y -= 14
    else:
        for alert in alerts[:25]:
            line = f"{alert.get('time', 'Unknown')}: {alert.get('text', '')}"
            for chunk in [line[i:i+95] for i in range(0, len(line), 95)]:
                if y < 60:
                    c.showPage()
                    y = height - 50
                    c.setFont("Helvetica", 10)
                c.drawString(40, y, chunk)
                y -= 12

    c.showPage()
    c.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="ufdr_report.pdf", mimetype="application/pdf")

if __name__ == "__main__":
    app.run(debug=True)
