import os
import re
import pandas as pd
import numpy as np


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)]


def preprocess_messages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _drop_unnamed(df)
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "message"})
    if "message" not in df.columns:
        # fallback: use the second column as message
        if len(df.columns) >= 2:
            df = df.rename(columns={df.columns[1]: "message"})
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"].str.len() > 0].copy()
    df["message_clean"] = (
        df["message"]
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["char_len"] = df["message"].str.len()
    df["word_count"] = df["message"].str.split().str.len()
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.lower().str.strip()
    return df


def preprocess_call_logs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "duration" in df.columns and "duration_sec" not in df.columns:
        df["duration_sec"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0)
    if "duration_sec" in df.columns:
        df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce").fillna(0)
    if "type" in df.columns and "call_type" not in df.columns:
        df["call_type"] = df["type"]
    ts_col = None
    if "start_time" in df.columns:
        ts_col = "start_time"
    elif "timestamp" in df.columns:
        ts_col = "timestamp"
    if ts_col:
        # Try epoch seconds, fallback to parse datetime strings
        ts = pd.to_numeric(df[ts_col], errors="coerce")
        if ts.notna().any():
            df["start_time_dt"] = pd.to_datetime(ts, unit="s", errors="coerce")
        else:
            df["start_time_dt"] = pd.to_datetime(df[ts_col], errors="coerce")
        df["call_hour"] = df["start_time_dt"].dt.hour
        if "is_night_call" not in df.columns:
            df["is_night_call"] = df["call_hour"].apply(
                lambda h: 1 if pd.notna(h) and (h >= 21 or h <= 5) else 0
            )
    reasons = []
    if "duration_sec" in df.columns:
        reasons.append(df["duration_sec"] >= 600)
    if "is_night_call" in df.columns:
        reasons.append(df["is_night_call"] == 1)
    if "call_type" in df.columns:
        reasons.append(df["call_type"].astype(str).str.lower().eq("missed"))
    if reasons:
        flags = reasons[0]
        for r in reasons[1:]:
            flags = flags | r
        df["suspicious_call"] = flags
        def _reason_row(row):
            labels = []
            if "duration_sec" in row and row["duration_sec"] >= 600:
                labels.append("long_duration")
            if "is_night_call" in row and row["is_night_call"] == 1:
                labels.append("night_call")
            if "call_type" in row and str(row["call_type"]).lower() == "missed":
                labels.append("missed_call")
            return ",".join(labels) if labels else None
        df["suspicious_reason"] = df.apply(_reason_row, axis=1)
    # Normalize categorical columns
    for col in ["call_type", "transaction_status", "fraud_type", "sim_id", "device_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def preprocess_contacts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _drop_unnamed(df)
    if "id" in df.columns and "name" not in df.columns:
        df = df.rename(columns={"id": "name"})
    for col in ["name", "phone", "email"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    # Coerce coordinates if present
    for col in ["latitude", "lat", "longitude", "lon", "lng"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def preprocess_gps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure required columns exist
    required_cols = ["timestamp", "lat", "lon", "address"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    # Convert types
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # Remove invalid rows
    df = df.dropna(subset=["lat", "lon"])

    # Fill address if missing
    df["address"] = df["address"].fillna("Unknown")

    # Sort by time (important for UI)
    df = df.sort_values(by="timestamp")

    return df
def rule_based_contact_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "name" in df.columns:
        df["flag_missing_name"] = df["name"].astype(str).str.len() == 0
        df["flag_short_name"] = df["name"].astype(str).str.len() < 3
        df["flag_nonalpha_name"] = ~df["name"].astype(str).str.match(r"^[A-Za-z .'-]+$")
    else:
        df["flag_missing_name"] = False
        df["flag_short_name"] = False
        df["flag_nonalpha_name"] = False

    if "phone" in df.columns:
        phone = df["phone"].astype(str).str.replace(r"\D", "", regex=True)
        df["flag_invalid_phone"] = phone.str.len().isin([0, 6, 7, 8, 9])
        df["flag_duplicate_phone"] = phone.duplicated(keep=False)
    else:
        df["flag_invalid_phone"] = False
        df["flag_duplicate_phone"] = False

    if "email" in df.columns:
        df["flag_invalid_email"] = ~df["email"].astype(str).str.match(
            r"^[\w\.-]+@[\w\.-]+\.\w+$"
        )
    else:
        df["flag_invalid_email"] = False

    # Flag invalid coordinates if present
    lat_col = "latitude" if "latitude" in df.columns else "lat" if "lat" in df.columns else None
    lon_col = "longitude" if "longitude" in df.columns else "lon" if "lon" in df.columns else "lng" if "lng" in df.columns else None
    if lat_col and lon_col:
        lat = df[lat_col]
        lon = df[lon_col]
        df["flag_invalid_location"] = (
            lat.isna() | lon.isna() | (lat.abs() > 90) | (lon.abs() > 180)
        )
    else:
        df["flag_invalid_location"] = False

    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    df["rule_based_flag"] = df[flag_cols].any(axis=1) if flag_cols else False
    
    # Add suspicion score for contacts
    df["suspicion_score"] = 0
    if "flag_missing_name" in df.columns:
        df["suspicion_score"] += df["flag_missing_name"].astype(int) * 15
    if "flag_duplicate_phone" in df.columns:
        df["suspicion_score"] += df["flag_duplicate_phone"].astype(int) * 20
    if "flag_invalid_phone" in df.columns:
        df["suspicion_score"] += df["flag_invalid_phone"].astype(int) * 25
    if "flag_nonalpha_name" in df.columns:
        df["suspicion_score"] += df["flag_nonalpha_name"].astype(int) * 10
    
    return df


def detect_dataset_type(df: pd.DataFrame, filename: str = "") -> str:
    cols = set([c.lower() for c in df.columns])
    fname = filename.lower()

    # ✅ GPS FIRST
    if {"latitude", "longitude"} <= cols or {"lat", "lon"} <= cols:
        return "gps"

    # ✅ CALL LOGS
    if {"duration_sec", "duration", "call_type", "caller_id", "receiver_id", "caller"} & cols or "call" in fname:
        return "call_logs"

    # ✅ MESSAGES
    if {"v1", "v2", "message"} & cols or "message" in fname:
        return "messages"

    # ✅ CONTACTS
    if "contact" in fname or {"name", "phone"} <= cols:
        return "contacts"

    # ✅ FALLBACK
    if len(df.columns) >= 20 or "gps" in fname:
        return "gps"

    return "unknown"


def analyze_dataframe(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns)
    }
