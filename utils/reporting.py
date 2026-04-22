import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet


# ===== INSIGHTS MODULE =====
def summarize_call_logs(df: pd.DataFrame):
    summary = {}
    summary["total_calls"] = len(df)
    if "is_anomaly" in df.columns:
        summary["anomaly_calls"] = int(df["is_anomaly"].sum())
    if "call_type" in df.columns:
        summary["call_type_counts"] = df["call_type"].value_counts().head(5).to_dict()
    if "transaction_status" in df.columns:
        summary["transaction_status_counts"] = df["transaction_status"].value_counts().head(5).to_dict()
    if "duration_sec" in df.columns:
        summary["duration_sec_mean"] = float(df["duration_sec"].mean())
    return summary


def summarize_messages(df: pd.DataFrame):
    summary = {}
    summary["total_messages"] = len(df)
    if "label" in df.columns:
        summary["label_counts"] = df["label"].value_counts().to_dict()
    if "word_count" in df.columns:
        summary["avg_word_count"] = float(df["word_count"].mean())
    return summary


def summarize_contacts(df: pd.DataFrame):
    summary = {}
    summary["total_contacts"] = len(df)
    if "rule_based_flag" in df.columns:
        summary["flagged_contacts"] = int(df["rule_based_flag"].sum())
    return summary


def summarize_gps(df: pd.DataFrame):
    summary = {}
    summary["total_points"] = len(df)
    if "gps_is_anomaly" in df.columns:
        summary["anomaly_points"] = int(df["gps_is_anomaly"].sum())
    if "dbscan_cluster" in df.columns:
        cluster_counts = df["dbscan_cluster"].value_counts().to_dict()
        summary["cluster_counts"] = cluster_counts
    return summary


def summarize_generic(df: pd.DataFrame):
    summary = {}
    summary["rows"] = len(df)
    summary["columns"] = len(df.columns)
    if len(df) > 0:
        missing = df.isna().mean().sort_values(ascending=False)
        summary["top_missing_columns"] = missing.head(5).round(3).to_dict()
        summary["duplicate_rows"] = int(df.duplicated().sum())

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        text_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()

        if numeric_cols:
            sample_num = numeric_cols[:3]
            numeric_stats = {}
            for c in sample_num:
                series = pd.to_numeric(df[c], errors="coerce")
                numeric_stats[c] = {
                    "min": float(series.min()) if series.notna().any() else None,
                    "max": float(series.max()) if series.notna().any() else None,
                    "mean": float(series.mean()) if series.notna().any() else None,
                }
            summary["numeric_overview"] = numeric_stats

        if text_cols:
            sample_text = text_cols[:3]
            text_stats = {}
            for c in sample_text:
                vc = df[c].astype(str).value_counts().head(5).to_dict()
                text_stats[c] = vc
            summary["top_values"] = text_stats
    return summary


def format_report_section(title: str, summary: dict) -> str:
    lines = [f"{title}"]
    for k, v in summary.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


# ===== PDF REPORT MODULE =====
def create_pdf(text, filename="report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    content = [Paragraph(line, styles["Normal"]) for line in text.split("\n")]
    doc.build(content)
    return filename
