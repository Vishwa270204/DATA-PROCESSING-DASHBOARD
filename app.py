import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import io
import os
import re
import warnings
from datetime import datetime
from scipy import stats
from scipy.stats import boxcox
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="DataPrep Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def nav_buttons(current_page):
    page_order = ["Upload & Inspect", "Statistics & EDA", "Recommendations", "Cleaning & Validation", "Encoding & Outliers", "Visualizations & Insights", "Export"]
    idx = page_order.index(current_page)
    slug = current_page.replace(" ", "_").replace("&", "and")

    st.markdown("""
    <style>
    [data-testid="stButton"] button {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("---")
    c_prev, c_mid, c_next = st.columns([3, 4, 3])

    with c_prev:
        if idx > 0:
            prev_label = f"⬅️ Prev: {page_order[idx-1]}"
            if st.button(prev_label, key=f"nav_prev_{slug}", type="secondary"):
                st.session_state.page = page_order[idx-1]
                st.rerun()
            st.markdown(f"""
            <style>
            div[data-testid="column"]:first-child .stButton button {{
                background: linear-gradient(135deg,#dc2626,#f97316) !important;
                color: white !important;
                border: none !important;
                border-radius: 8px !important;
                font-weight: 600 !important;
                white-space: nowrap !important;
                width: 100% !important;
            }}
            </style>
            """, unsafe_allow_html=True)

    with c_next:
        if idx < len(page_order) - 1:
            next_label = f"Next: {page_order[idx+1]} ➡️"
            if st.button(next_label, key=f"nav_next_{slug}", type="secondary"):
                st.session_state.page = page_order[idx+1]
                st.rerun()
            st.markdown(f"""
            <style>
            div[data-testid="column"]:last-child .stButton button {{
                background: linear-gradient(135deg,#059669,#10b981) !important;
                color: white !important;
                border: none !important;
                border-radius: 8px !important;
                font-weight: 600 !important;
                white-space: nowrap !important;
                width: 100% !important;
            }}
            </style>
            """, unsafe_allow_html=True)    
def save_operation(file_name, operation, details):
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.execute("INSERT INTO processing_history (file_name,operation,details,timestamp) VALUES (?,?,?,?)",
                     (file_name, operation, str(details), datetime.now().isoformat()))
        conn.commit(); conn.close()
    except: pass

def get_processing_history(file_name=None):
    try:
        conn = sqlite3.connect(DB_NAME)
        if file_name:
            df = pd.read_sql("SELECT * FROM processing_history WHERE file_name=? ORDER BY timestamp DESC",
                             conn, params=(file_name,))
        else:
            df = pd.read_sql("SELECT * FROM processing_history ORDER BY timestamp DESC", conn)
        conn.close(); return df
    except: return pd.DataFrame()

# ── File loading ──────────────────────────────
def load_file(buf, name):
    n = name.lower()
    if n.endswith(".csv"):    return pd.read_csv(buf, low_memory=False)
    elif n.endswith(".xlsx"): return pd.read_excel(buf, engine="openpyxl")
    elif n.endswith(".xls"):  return pd.read_excel(buf)
    else: raise ValueError("Unsupported format")

# ── Column type detection ─────────────────────
def identify_column_types(df):
    ct = {"numerical": [], "categorical": [], "datetime": [], "boolean": [], "id": []}
    for col in df.columns:
        cl = col.lower(); s = df[col].dropna()
        if s.empty: ct["categorical"].append(col); continue
        if any(k in cl for k in ["_id","id","index","key","uuid","guid"]) and s.nunique() == len(s):
            ct["id"].append(col); continue
        if s.dtype == bool or set(map(str, s.unique())).issubset({"True","False","true","false","1","0","yes","no","Yes","No"}):
            ct["boolean"].append(col); continue
        if pd.api.types.is_datetime64_any_dtype(s): ct["datetime"].append(col); continue
        if s.dtype == object:
            try: pd.to_datetime(s.head(30), infer_datetime_format=True); ct["datetime"].append(col); continue
            except: pass
        if pd.api.types.is_numeric_dtype(s): ct["numerical"].append(col)
        else: ct["categorical"].append(col)
    return ct

@st.cache_data
def get_dataset_summary(df):
    return {
        "rows": len(df), "columns": len(df.columns),
        "missing_cells": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().sum().sum() / df.size * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 3)
    }

def missing_value_report(df):
    ct = identify_column_types(df); report = []
    for col in df.columns:
        m = df[col].isnull().sum()
        if m > 0:
            if col in ct["numerical"]:  dtype, strats = "numerical",  ["mean","median","mode","drop"]
            elif col in ct["datetime"]: dtype, strats = "datetime",   ["ffill","bfill","drop"]
            else:                       dtype, strats = "categorical", ["mode","ffill","bfill","custom","drop"]
            report.append({"column":col,"missing":m,"pct":round(m/len(df)*100,2),"type":dtype,"strategies":strats})
    return report

def fill_missing_values(df, column, strategy, custom_value=None):
    before = df[column].isnull().sum(); df = df.copy()
    if strategy == "mean":     df[column].fillna(df[column].mean(), inplace=True)
    elif strategy == "median": df[column].fillna(df[column].median(), inplace=True)
    elif strategy == "mode":
        m = df[column].mode()
        if not m.empty: df[column].fillna(m[0], inplace=True)
    elif strategy == "ffill": df[column] = df[column].ffill()
    elif strategy == "bfill": df[column] = df[column].bfill()
    elif strategy == "custom" and custom_value is not None: df[column].fillna(custom_value, inplace=True)
    elif strategy == "drop":  df = df.dropna(subset=[column]).reset_index(drop=True)
    after = df[column].isnull().sum()
    return df, {"before": before, "after": after, "filled": before - after}


def detect_outliers_iqr(df):
    result = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75); IQR = Q3 - Q1
        lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        out = df[(df[col] < lo) | (df[col] > hi)]
        result[col] = {"count": len(out), "pct": round(len(out)/len(df)*100, 2),
                       "lower": lo, "upper": hi, "Q1": Q1, "Q3": Q3, "IQR": IQR,
                       "mean": df[col].mean(), "std": df[col].std(), "rows": out.index.tolist()}
    return result


def detect_outliers_zscore(df, threshold=3):
    result = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        z = np.abs(stats.zscore(s))
        outlier_idx = s.index[z > threshold].tolist()
        result[col] = {"count": len(outlier_idx), "pct": round(len(outlier_idx)/len(df)*100, 2),
                       "mean": s.mean(), "std": s.std(), "threshold": threshold, "rows": outlier_idx}
    return result

def remove_outliers(df, column, method="iqr"):
    before = len(df)
    if method == "iqr":
        Q1, Q3 = df[column].quantile(0.25), df[column].quantile(0.75); IQR = Q3 - Q1
        df = df[(df[column] >= Q1-1.5*IQR) & (df[column] <= Q3+1.5*IQR)]
    elif method == "zscore":
        s = df[column].dropna(); z = np.abs(stats.zscore(s))
        keep = s.index[z <= 3]; df = df.loc[keep]
    df = df.reset_index(drop=True)
    return df, {"removed": before - len(df), "new_count": len(df)}

def cap_outliers(df, column):
    df = df.copy()
    Q1, Q3 = df[column].quantile(0.25), df[column].quantile(0.75); IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    n = ((df[column] < lo) | (df[column] > hi)).sum()
    df[column] = df[column].clip(lower=lo, upper=hi)
    return df, {"capped": int(n)}

def calculate_skewness(df):
    rows = []
    for col in df.select_dtypes(include=[np.number]).columns:
        sk = df[col].skew()
        if sk < -1:       cls = "Highly Left Skewed"
        elif sk < -0.5:   cls = "Moderately Left Skewed"
        elif sk <= 0.5:   cls = "Approximately Normal"
        elif sk <= 1:     cls = "Moderately Right Skewed"
        else:             cls = "Highly Right Skewed"
        rows.append({"Column": col, "Skewness": round(sk, 4), "Classification": cls})
    return pd.DataFrame(rows)

def descriptive_statistics(df):
    rows = []
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        mode_v = s.mode().iloc[0] if not s.mode().empty else np.nan
        rows.append({
            "Column": col, "Count": len(s), "Missing": df[col].isnull().sum(),
            "Unique": df[col].nunique(),
            "Mean": round(s.mean(),4), "Median": round(s.median(),4),
            "Mode": round(mode_v,4), "Std": round(s.std(),4),
            "Variance": round(s.var(),4),
            "Min": round(s.min(),4), "Max": round(s.max(),4),
            "Q1": round(s.quantile(0.25),4), "Q3": round(s.quantile(0.75),4),
            "Skewness": round(s.skew(),4), "Kurtosis": round(s.kurtosis(),4)
        })
    return pd.DataFrame(rows)

def detect_invalid_values(df):
    issues = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        cl = col.lower()
        if any(k in cl for k in ["age"]):
            b = df[(df[col] < 0) | (df[col] > 150)]
            if len(b): issues[col] = {"issue": "Age out of range (0–150)", "count": len(b)}
        elif any(k in cl for k in ["pct","percent","rate"]):
            b = df[(df[col] < 0) | (df[col] > 100)]
            if len(b): issues[col] = {"issue": "Percentage out of range", "count": len(b)}
        elif any(k in cl for k in ["salary","income","revenue","price","cost","amount"]):
            b = df[df[col] < 0]
            if len(b): issues[col] = {"issue": "Negative monetary value", "count": len(b)}
    return issues

NON_NEGATIVE_KEYWORDS = [
    "age","salary","income","revenue","debt","experience","weight","height",
    "price","cost","quantity","amount","months","years","positive_node",
    "positivenode","tumor","size","count","duration","population","rate",
    "pct","percent","score","grade","rank","distance","area","volume",
    "length","width"
]
SIGNED_KEYWORDS = [
    "profit","loss","temperature","balance","change","growth","return",
    "difference","delta","variance","gain","net","flow","deviation","residual"
]

def is_non_negative_column(col_name):
    cl = col_name.lower().replace(" ", "_")
    if any(kw in cl for kw in SIGNED_KEYWORDS):
        return False
    return any(kw in cl for kw in NON_NEGATIVE_KEYWORDS)

def detect_negative_values(df):
    r = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        if is_non_negative_column(col):
            n = df[df[col] < 0]
            if len(n):
                r[col] = {"count": len(n), "reason": "Domain requires non-negative values"}
    return r

def validate_single_value(col_name, value, df_context=None, target_col=None):
    cl = col_name.lower()
    if target_col and col_name == target_col and df_context is not None:
        allowed = list(df_context[col_name].dropna().unique())
        if len(allowed) <= 2 and value not in allowed:
            return "invalid", f"Target column only allows: {allowed}"
    try:
        fval = float(value)
    except (TypeError, ValueError):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "invalid", "Missing / null value"
        return "valid", ""
    if any(k in cl for k in ["age"]):
        if fval < 0 or fval > 150:
            return "invalid", f"Age {fval} out of valid range (0–150)"
    if any(k in cl for k in ["pct","percent","rate"]) and "growth" not in cl and "change" not in cl:
        if fval < 0 or fval > 100:
            return "invalid", f"Percentage {fval} out of range (0–100)"
    if any(k in cl for k in ["salary","income","revenue","price","cost","amount","debt"]):
        if fval < 0:
            return "invalid", f"Monetary value {fval} cannot be negative"
    if any(k in cl for k in ["experience","years","months","duration"]):
        if fval < 0:
            return "invalid", f"Time value {fval} cannot be negative"
    if any(k in cl for k in ["weight","height","tumor","size"]):
        if fval < 0:
            return "invalid", f"Physical measurement {fval} cannot be negative"
    if is_non_negative_column(col_name) and fval < 0:
        return "invalid", f"Column '{col_name}' should not contain negative values"
    return "valid", ""

def detect_invalid_email(df):
    pat = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$"); r = {}
    for col in df.select_dtypes(include="object").columns:
        if any(k in col.lower() for k in ["email","mail","e-mail"]):
            bad = df[col].dropna().apply(lambda x: not bool(pat.match(str(x))))
            if bad.sum(): r[col] = {"count": int(bad.sum())}
    return r

def detect_invalid_phone(df):
    pat = re.compile(r"^[\+]?[\d\s\-\(\)]{7,15}$"); r = {}
    for col in df.select_dtypes(include="object").columns:
        if any(k in col.lower() for k in ["phone","mobile","tel","contact"]):
            bad = df[col].dropna().apply(lambda x: not bool(pat.match(str(x))))
            if bad.sum(): r[col] = {"count": int(bad.sum())}
    return r

def detect_future_dates(df):
    now = pd.Timestamp.now(); r = {}
    for col in df.columns:
        if any(k in col.lower() for k in ["birth","dob","born","date","created","joined"]):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                n = (parsed > now).sum()
                if n: r[col] = {"count": int(n)}
            except: pass
    return r

def detect_duplicate_information_columns(df):
    suggestions = []
    cols = df.columns.tolist()
    ct = identify_column_types(df)

    def norm_set(series):
        return set(series.dropna().astype(str).str.strip().str.lower().unique())

    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] > 1:
        corr = num_df.corr().abs()
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                v = corr.iloc[i, j]
                if v > 0.98:
                    suggestions.append({
                        "col1": corr.columns[i], "col2": corr.columns[j],
                        "reason": f"Near-perfect correlation ({v:.3f})",
                        "score": round(float(v)*100, 1), "action": "Consider dropping one"
                    })

    cat_cols = [c for c in cols if c in ct["categorical"] or c in ct["boolean"]]
    for i in range(len(cat_cols)):
        for j in range(i+1, len(cat_cols)):
            try:
                pair = df[[cat_cols[i], cat_cols[j]]].dropna()
                if len(pair) == 0: continue
                fwd = pair.groupby(cat_cols[i])[cat_cols[j]].nunique()
                bwd = pair.groupby(cat_cols[j])[cat_cols[i]].nunique()
                if fwd.max() == 1 and bwd.max() == 1:
                    s1 = norm_set(df[cat_cols[i]]); s2 = norm_set(df[cat_cols[j]])
                    overlap_count = sum(any(a[0]==b[0] or a in b or b in a for b in s2) for a in s1)
                    overlap_ratio = overlap_count / max(len(s1), 1)
                    suggestions.append({
                        "col1": cat_cols[i], "col2": cat_cols[j],
                        "reason": f"One-to-one value mapping (e.g. {list(s1)[:2]} ↔ {list(s2)[:2]})",
                        "score": round(85 + overlap_ratio*10, 1),
                        "action": "Likely same information encoded differently"
                    })
            except: pass

    for i in range(len(cat_cols)):
        for j in range(i+1, len(cat_cols)):
            already = any((s["col1"]==cat_cols[i] and s["col2"]==cat_cols[j]) or
                          (s["col1"]==cat_cols[j] and s["col2"]==cat_cols[i]) for s in suggestions)
            if already: continue
            try:
                s1 = norm_set(df[cat_cols[i]]); s2 = norm_set(df[cat_cols[j]])
                if not s1 or not s2: continue
                intersection = s1 & s2; union = s1 | s2
                jaccard = len(intersection)/len(union) if union else 0
                if jaccard > 0.8:
                    suggestions.append({
                        "col1": cat_cols[i], "col2": cat_cols[j],
                        "reason": f"High value overlap (Jaccard={jaccard:.2f}): {list(intersection)[:3]}",
                        "score": round(jaccard*100, 1), "action": "Columns may contain duplicate information"
                    })
            except: pass

    for i in range(len(cat_cols)):
        for j in range(i+1, len(cat_cols)):
            already = any((s["col1"]==cat_cols[i] and s["col2"]==cat_cols[j]) or
                          (s["col1"]==cat_cols[j] and s["col2"]==cat_cols[i]) for s in suggestions)
            if already: continue
            try:
                pair = df[[cat_cols[i], cat_cols[j]]].dropna()
                if len(pair) < 10: continue
                a = pair[cat_cols[i]].astype(str).str.strip().str.lower()
                b = pair[cat_cols[j]].astype(str).str.strip().str.lower()
                row_match = (a == b).mean()
                if row_match >= 0.95:
                    suggestions.append({
                        "col1": cat_cols[i], "col2": cat_cols[j],
                        "reason": f"Row-level match: {row_match*100:.1f}% of rows identical",
                        "score": round(row_match*100, 1), "action": "Columns appear to be exact duplicates"
                    })
            except: pass

    seen = set(); unique = []
    for s in suggestions:
        key = tuple(sorted([s["col1"], s["col2"]]))
        if key not in seen:
            seen.add(key); unique.append(s)
    return unique

def recommend_encoding(df, col, is_target=False):
    n = df[col].nunique()
    if is_target:
        if n == 2:
            return "label", "Binary target"
        elif n <= 15:
            return "label", "Multiclass target"
        else:
            return "frequency", "High-cardinality target"
    else:
        if n == 2:
            return "label", "Binary column"
        elif n <= 10:
            return "onehot", "Low-cardinality column"
        else:
            return "frequency", "High-cardinality column"
def apply_encoding(df, col, enc_type, ordinal_order=None):
    df = df.copy(); mapping = None
    if enc_type == "label":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        st.session_state.encoders[col] = {
            cls: int(code)
            for code, cls in enumerate(le.classes_)
        }
        mapping = pd.DataFrame({
            "Original": le.classes_,
            "Encoded": range(len(le.classes_))
        })
    elif enc_type == "onehot":
        dummies = pd.get_dummies(df[col], prefix=col); df = pd.concat([df, dummies], axis=1)
        mapping = pd.DataFrame({"Original": dummies.columns, "Encoded": dummies.columns})
    elif enc_type == "ordinal" and ordinal_order:
        om = {v: i for i, v in enumerate(ordinal_order)}
        df[col] = df[col].map(om)
        st.session_state.encoders[col] = om
    elif enc_type == "frequency":
        freq = df[col].value_counts(normalize=True)
        df[col] = df[col].map(freq)
        st.session_state.encoders[col] = freq.to_dict()
    return df, mapping


def calculate_data_quality_score(df):
    miss_pct = df.isnull().sum().sum() / df.size * 100
    dup_pct  = df.duplicated().sum() / len(df) * 100
    outliers = detect_outliers_iqr(df)
    avg_out  = np.mean([v["pct"] for v in outliers.values()]) if outliers else 0
    inv      = detect_invalid_values(df)
    inv_pct  = sum(v["count"] for v in inv.values()) / len(df) * 100 if inv else 0
    score    = max(0,30-miss_pct*0.6) + max(0,20-dup_pct*0.4) + max(0,20-avg_out*0.4) + max(0,15-inv_pct*0.3) + 15
    return round(min(100, score), 1)

def export_csv(df):
    return df.to_csv(index=False).encode("utf-8")

def export_excel(df):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Processed")
    return out.getvalue()

# ── CSS ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root {
    --bg:#f8f9fc;--surface:#ffffff;--surface2:#f1f3f9;--border:#e2e6f0;
    --accent:#2563eb;--accent2:#3b82f6;--accent-light:#eff6ff;
    --success:#16a34a;--success-light:#f0fdf4;
    --warning:#d97706;--warning-light:#fffbeb;
    --danger:#dc2626;--danger-light:#fef2f2;
    --text:#111827;--text2:#374151;--muted:#6b7280;
    --shadow:0 1px 3px rgba(0,0,0,0.08),0 1px 2px rgba(0,0,0,0.04);
    --shadow-md:0 4px 6px rgba(0,0,0,0.07),0 2px 4px rgba(0,0,0,0.04);
}
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background-color:var(--bg)!important;color:var(--text)!important;}
.stApp{background:var(--bg)!important;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}
.main-header{background:linear-gradient(135deg,#1e3a8a 0%,#2563eb 60%,#3b82f6 100%);border-radius:16px;padding:28px 32px;margin-bottom:24px;box-shadow:var(--shadow-md);}
.main-header h1{font-family:'Inter',sans-serif!important;font-size:1.8rem!important;font-weight:700!important;color:#ffffff!important;margin:0!important;}
.main-header p{color:rgba(255,255,255,0.75)!important;margin:6px 0 0!important;font-size:0.95rem;}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px;text-align:center;box-shadow:var(--shadow);}
.metric-card .val{font-family:'JetBrains Mono',monospace;font-size:1.7rem;font-weight:700;color:var(--accent);display:block;}
.metric-card .label{font-size:0.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-top:4px;}
.section-header{display:flex;align-items:center;gap:10px;padding:12px 18px;background:var(--accent-light);border-left:3px solid var(--accent);border-radius:0 8px 8px 0;margin:20px 0 16px 0;}
.section-header h3{margin:0!important;font-size:0.95rem;font-weight:600;color:var(--accent);}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:0.75rem;font-weight:600;font-family:'JetBrains Mono',monospace;}
.badge-success{background:var(--success-light);color:var(--success);border:1px solid #bbf7d0;}
.badge-warning{background:var(--warning-light);color:var(--warning);border:1px solid #fde68a;}
.badge-danger{background:var(--danger-light);color:var(--danger);border:1px solid #fecaca;}
.badge-info{background:var(--accent-light);color:var(--accent);border:1px solid #bfdbfe;}
.progress-bar-wrap{background:var(--surface2);border-radius:8px;height:8px;overflow:hidden;margin:4px 0;}
.progress-bar-fill{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2));border-radius:8px;}
.stButton>button{background:linear-gradient(135deg,var(--accent),var(--accent2))!important;color:white!important;border:none!important;border-radius:8px!important;font-weight:600!important;padding:10px 24px!important;}
.stSelectbox>div>div{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:8px!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--surface2)!important;border-radius:10px;padding:4px;border:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{color:var(--muted)!important;border-radius:6px!important;font-weight:500;}
.stTabs [aria-selected="true"]{background:var(--accent)!important;color:white!important;}
.info-box{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px;box-shadow:var(--shadow);}
.info-box .ib-label{font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;}
.info-box .ib-val{font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:700;color:var(--text);}
div.stAlert{border-radius:10px!important;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────
defaults = {
    "df": None,
    "original_df": None,
    "processed_df": None,
    "file_name": "",
    "page": "Upload & Inspect",
    "encoded_columns": [],
    "encoders": {},
    "target_col": "— None —",
    "target_encoded": False,
    "transformed_columns": [],  
    "export_done": False
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 24px;'>
        <div style='font-family:"Inter",sans-serif;font-size:1.3rem;font-weight:700;color:#2563eb;'>📊 DataPrep Pro</div>
        <div style='font-size:0.75rem;color:#6b7280;margin-top:4px;'>Smart Preprocessing Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    pages = ["📁 Upload & Inspect","📈 Statistics & EDA","💡 Recommendations","🧹 Cleaning & Validation","🔠 Encoding & Outliers","📊 Visualizations","📦 Export"]
    page_map = {p: p.split(" ", 1)[1] for p in pages}
    page_keys = list(page_map.values())
    selected = st.radio("Navigation", pages, label_visibility="collapsed",
                        index=page_keys.index(st.session_state.page) if st.session_state.page in page_keys else 0)
    st.session_state.page = page_map[selected]
    st.markdown("---")

    if st.session_state.df is not None:
        df = st.session_state.df
        qs = calculate_data_quality_score(df)
        qc = "#16a34a" if qs >= 80 else "#d97706" if qs >= 60 else "#dc2626"
        st.markdown(f"""
        <div class='info-box'>
            <div class='ib-label' style='margin-bottom:12px;'>Live Dataset Stats</div>
            <div style='display:flex;justify-content:space-between;margin:6px 0;'>
                <span style='color:#6b7280;font-size:0.85rem;'>Rows</span>
                <span class='ib-val'>{len(df):,}</span>
            </div>
            <div style='display:flex;justify-content:space-between;margin:6px 0;'>
                <span style='color:#6b7280;font-size:0.85rem;'>Columns</span>
                <span class='ib-val'>{len(df.columns)}</span>
            </div>
            <div style='display:flex;justify-content:space-between;margin:6px 0;'>
                <span style='color:#6b7280;font-size:0.85rem;'>Missing</span>
                <span style='font-family:"JetBrains Mono",monospace;font-size:0.85rem;color:#d97706;font-weight:700;'>{df.isnull().sum().sum():,}</span>
            </div>
            <div style='display:flex;justify-content:space-between;margin:6px 0;'>
                <span style='color:#6b7280;font-size:0.85rem;'>Duplicates</span>
                <span style='font-family:"JetBrains Mono",monospace;font-size:0.85rem;color:#dc2626;font-weight:700;'>{df.duplicated().sum()}</span>
            </div>
            <div style='margin-top:14px;'>
                <div style='font-size:0.7rem;color:#6b7280;margin-bottom:6px;'>Quality Score</div>
                <div style='font-family:"JetBrains Mono",monospace;font-size:1.6rem;font-weight:700;color:{qc};'>{qs}<span style='font-size:0.9rem;font-weight:400;color:#6b7280;'>/100</span></div>
                <div style='background:#f1f3f9;border-radius:6px;height:7px;margin-top:6px;'>
                    <div style='width:{qs}%;height:100%;background:{qc};border-radius:6px;'></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='padding:16px;background:#f8f9fc;border:2px dashed #e2e6f0;border-radius:12px;text-align:center;color:#6b7280;font-size:0.85rem;'>
            No dataset loaded.<br>Upload a file to begin.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='margin-top:16px;padding:10px;font-size:0.72rem;color:#9ca3af;text-align:center;'>
        File: <b style='color:#6b7280;'>{st.session_state.file_name or "None"}</b>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# PAGE 1 — UPLOAD & INSPECT
# ═══════════════════════════════════════════════
if st.session_state.page == "Upload & Inspect":
    st.markdown("""
    <div class='main-header'>
        <h1>📁 Upload & Inspect</h1>
        <p>Upload your dataset to begin intelligent preprocessing analysis</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "xls"],
        help="Supports CSV, Excel (.xlsx/.xls)"
    )

    if uploaded:
        try:
    
            # Only load when a NEW file is selected
            if uploaded.name != st.session_state.get("file_name", ""):
                df = load_file(uploaded, uploaded.name)
                st.session_state.original_df = df.copy()      # RAW DATA
                st.session_state.processed_df = df.copy()     # FOR ENCODING / ML
                # Compatibility with existing code
                st.session_state.df = st.session_state.processed_df
                st.session_state.file_name = uploaded.name
                st.session_state.encoded_columns = []
                if "cleaning_history" in st.session_state:
                    st.session_state.cleaning_history = []
                if "operations" in st.session_state:
                    st.session_state.operations = []
                st.rerun()
                st.session_state.encoded_columns = []
    
                if "cleaning_history" in st.session_state:
                    st.session_state.cleaning_history = []
    
                if "operations" in st.session_state:
                    st.session_state.operations = []
    
                size_kb = uploaded.size / 1024
    
                conn = sqlite3.connect(DB_NAME)
                conn.execute(
                    "INSERT INTO file_metadata VALUES (NULL,?,?,?,?,?)",
                    (
                        uploaded.name,
                        datetime.now().isoformat(),
                        round(size_kb, 2),
                        len(df),
                        len(df.columns)
                    )
                )
                conn.commit()
                conn.close()
    
                st.success(
                    f"✅ Loaded **{uploaded.name}** — "
                    f"{len(df):,} rows × {len(df.columns)} columns"
            )
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
    if st.session_state.df is not None:
        raw_df = st.session_state.original_df
        processed_df = st.session_state.processed_df
        # Use RAW data for UI
        df = raw_df        
        summary = get_dataset_summary(raw_df)
        ct = identify_column_types(raw_df)
        cols_m = st.columns(5)
        for col_w, label, val in zip(cols_m,
            ["Rows","Columns","Missing %","Duplicates","Memory MB"],
            [f"{summary['rows']:,}", str(summary['columns']),
             f"{summary['missing_pct']}%", str(summary['duplicate_rows']),
             f"{summary['memory_mb']}"]):
            with col_w:
                st.markdown(f"""<div class='metric-card'><span class='val'>{val}</span><span class='label'>{label}</span></div>""", unsafe_allow_html=True)
        st.markdown("&nbsp;")
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["👁️ Preview","📋 Schema","🏷️ Column Types","❓ Missing","➕ Add Row","🔄 Encoded data"])
        with tab1:
            st.markdown(f"<div style='font-size:0.85rem;color:#6b7280;margin-bottom:8px;'>Dataset: <b style='color:#111827;'>{summary['rows']:,} rows × {summary['columns']} columns</b></div>", unsafe_allow_html=True)
            preview_opt = st.selectbox("Show", ["First 5 rows","First 10 rows","First 20 rows","Entire dataset"], key="preview_sel")
            n_map = {"First 5 rows":5,"First 10 rows":10,"First 20 rows":20}
            show_df = df if preview_opt == "Entire dataset" else df.head(n_map[preview_opt])
            st.dataframe(show_df, use_container_width=True, height=380)

        with tab2:
            mem_per_col = df.memory_usage(deep=True)
            schema_rows = []
            for col in df.columns:
                null_c = int(df[col].isnull().sum())
                schema_rows.append({
                    "Column": col, "Data Type": str(df[col].dtype),
                    "Null Count": null_c, "Null %": round(null_c/len(df)*100, 2),
                    "Unique Values": int(df[col].nunique()),
                    "Memory (KB)": round(mem_per_col.get(col,0)/1024, 3)
                })
            st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, height=420)

        with tab3:
            cc1, cc2 = st.columns(2)
            with cc1:
                for typ, emoji in [("numerical","🔢"),("categorical","🏷️")]:
                    st.markdown(f"**{emoji} {typ.title()}** ({len(ct[typ])})")
                    st.write(", ".join(ct[typ]) if ct[typ] else "_None detected_")
            with cc2:
                for typ, emoji in [("datetime","📅"),("boolean","✅"),("id","🔑")]:
                    st.markdown(f"**{emoji} {typ.title()}** ({len(ct[typ])})")
                    st.write(", ".join(ct[typ]) if ct[typ] else "_None detected_")

        with tab4:
            miss_report = missing_value_report(df)
            if miss_report:
                miss_df = pd.DataFrame([{
                    "Column": r["column"], "Missing Count": r["missing"],
                    "Missing %": r["pct"], "Type": r["type"]
                } for r in miss_report])
                st.dataframe(miss_df, use_container_width=True)
                try:
                    fig = go.Figure(go.Bar(
                        x=[r["column"] for r in miss_report],
                        y=[r["pct"] for r in miss_report],
                        marker_color="#2563eb",
                        text=[f"{r['pct']}%" for r in miss_report],
                        textposition="outside"
                    ))
                    fig.update_layout(title="Missing Value % by Column",
                                      template="plotly_white", height=320,
                                      yaxis_title="Missing %", xaxis_title="Column",
                                      paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {e}")
            else:
                st.markdown("<span class='badge badge-success'>✅ No missing values detected</span>", unsafe_allow_html=True)

        # ── Add Row  ── FIX 1: row count updates correctly ─────────────────────
        with tab5:
            st.markdown("**Manually add a new row to the dataset:**")

            # Always read the CURRENT df from session state so count is accurate
            current_df = st.session_state.get("original_df")
            if current_df is None or current_df.empty:
                st.warning("Please upload a dataset first.")
                st.stop()
            ct_live = identify_column_types(current_df)            
            input_data = {}
            _target_col = st.session_state.get("target_col", "— None —")
            if _target_col == "— None —":
                _target_col = None
            if _target_col:
                st.caption(f"🎯 Target column: **{_target_col}** — only existing values allowed.")
            form_cols = st.columns(min(3, len(current_df.columns)))
            for i, col in enumerate(current_df.columns):
                with form_cols[i % 3]:
                    dtype = str(current_df[col].dtype)
                    if _target_col and col == _target_col:
                        unique_vals = list(current_df[col].dropna().unique())
                        if len(unique_vals) > 2:
                            unique_vals = unique_vals[:2]
                        input_data[col] = st.selectbox(f"{col} 🎯", unique_vals, key=f"inp_{col}")
                    elif "int" in dtype or "float" in dtype:
                        input_data[col] = st.number_input(col, value=0.0, key=f"inp_{col}")
                    elif col in ct_live["boolean"]:
                        input_data[col] = st.selectbox(col, [True, False], key=f"inp_{col}")
                    elif col in ct_live["categorical"] and current_df[col].nunique() < 50:
                        opts = list(current_df[col].dropna().unique())
                        input_data[col] = st.selectbox(col, opts, key=f"inp_{col}")
                    else:
                        input_data[col] = st.text_input(col, key=f"inp_{col}")

            if st.button("➕ Add Row"):
                new_row = {col: input_data.get(col, np.nan) for col in current_df.columns}
                val_results = []
                for col, val in new_row.items():
                    status, reason = validate_single_value(col, val, current_df, target_col=_target_col)
                    val_results.append({"Column": col, "Value": val, "Status": status.title(), "Reason": reason or "—"})

                n_invalid = sum(1 for r in val_results if r["Status"] == "Invalid")

                if n_invalid == 0:
                    # FIX 1: build updated_df from current snapshot, assign, rerun
                    updated_original = pd.concat([st.session_state.original_df,pd.DataFrame([new_row])],ignore_index=True)
                    st.session_state.original_df = updated_original
                    st.session_state.df = updated_original.copy()
                    save_operation(st.session_state.file_name, "Add Row", new_row)
                    st.session_state.pop("_pending_row", None)
                    st.session_state.pop("_pending_val_results", None)
                    st.success(f"✅ Row added! Dataset: {len(current_df):,} → {len(updated_original):,} rows.")
                    st.rerun()
                else:
                    st.session_state["_pending_row"] = new_row
                    st.session_state["_pending_val_results"] = val_results

            # Pending row review
            if st.session_state.get("_pending_row") and st.session_state.get("_pending_val_results"):
                pending_row   = st.session_state["_pending_row"]
                val_results   = st.session_state["_pending_val_results"]
                n_valid   = sum(1 for r in val_results if r["Status"] == "Valid")
                n_invalid = sum(1 for r in val_results if r["Status"] == "Invalid")

                st.markdown("---")
                st.markdown("<div class='section-header'><h3>Row Validation Results</h3></div>", unsafe_allow_html=True)

                def _row_style(row):
                    if row["Status"] == "Invalid":
                        return ["background-color:#fef2f2;color:#dc2626"]*len(row)
                    return ["background-color:#f0fdf4;color:#16a34a"]*len(row)

                val_df = pd.DataFrame(val_results)
                val_df["Value"] = val_df["Value"].astype(str)
                try:
                    st.dataframe(val_df.style.apply(_row_style, axis=1), use_container_width=True, hide_index=True)
                except:
                    st.dataframe(val_df, use_container_width=True, hide_index=True)

                st.markdown(f"""
                <div style='display:flex;gap:16px;margin:12px 0;flex-wrap:wrap;'>
                    <span class='badge badge-success'>✅ {n_valid} Valid Fields</span>
                    <span class='badge badge-danger'>❌ {n_invalid} Invalid Fields</span>
                </div>
                """, unsafe_allow_html=True)
                st.warning("⚠️ This row contains invalid values.")

                c_keep, c_edit, c_del = st.columns(3)
                with c_keep:
                    if st.button("✅ Keep Row Anyway", key="keep_invalid_row"):
                        cur = st.session_state.df
                        upd = pd.concat([cur, pd.DataFrame([pending_row])], ignore_index=True)
                        st.session_state.df = upd
                        save_operation(st.session_state.file_name, "Add Row (with issues)", pending_row)
                        st.session_state.pop("_pending_row", None)
                        st.session_state.pop("_pending_val_results", None)
                        st.success(f"Row added. Dataset: {len(cur):,} → {len(upd):,} rows.")
                        st.rerun()
                with c_edit:
                    st.info("💡 Edit the values above and click **➕ Add Row** again.")
                with c_del:
                    if st.button("🗑️ Discard Row", key="discard_row"):
                        st.session_state.pop("_pending_row", None)
                        st.session_state.pop("_pending_val_results", None)
                        st.info("Row discarded.")
                        st.rerun()
        with tab6:
            st.subheader("🔄 Encoded Dataset")
            if st.session_state.get("processed_df") is not None:
                st.info(
                    "This dataset is used for ML and analysis. "
                    "The original dataset remains unchanged."
                )
                st.dataframe(
                    st.session_state.processed_df,
                    use_container_width=True,
                    height=500
                )
                st.write(
                    f"Rows: {len(st.session_state.processed_df):,} | "
                    f"Columns: {len(st.session_state.processed_df.columns):,}"
                )
            else:
                st.warning("No processed dataset available.")
    nav_buttons("Upload & Inspect")  
# ═══════════════════════════════════════════════
# PAGE 2 — STATISTICS & EDA
# ═══════════════════════════════════════════════
elif st.session_state.page == "Statistics & EDA":
    st.markdown("""
    <div class='main-header'>
        <h1>📈 Statistics & EDA</h1>
        <p>Exploratory Data Analysis — understand your dataset before cleaning</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = st.session_state.df
    ct = identify_column_types(df)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    tab1, tab2, tab3, tab4= st.tabs([
        "📊 Overview","🔢 Numerical Stats","🏷️ Categorical Stats","🔗 Correlation"
    ])

    with tab1:
        summary = get_dataset_summary(df)
        st.markdown("<div class='section-header'><h3>Dataset Overview</h3></div>", unsafe_allow_html=True)
        c1,c2,c3,c4,c5 = st.columns(5)
        for w, label, val in zip(
            [c1,c2,c3,c4,c5],
            ["Rows","Columns","Missing %","Duplicates","Memory MB"],
            [f"{summary['rows']:,}", str(summary['columns']),
             f"{summary['missing_pct']}%", str(summary['duplicate_rows']),
             f"{summary['memory_mb']}"]
        ):
            with w:
                st.markdown(f"""<div class='metric-card'><span class='val'>{val}</span>
                <span class='label'>{label}</span></div>""", unsafe_allow_html=True)

        st.markdown("&nbsp;")
        st.markdown("<div class='section-header'><h3>Column Type Breakdown</h3></div>", unsafe_allow_html=True)
        type_data = {
            "Numerical": len(ct["numerical"]),
            "Categorical": len(ct["categorical"]),
            "Boolean": len(ct["boolean"]),
            "Datetime": len(ct["datetime"]),
            "ID": len(ct["id"])
        }
        tc1, tc2 = st.columns([1,2])
        with tc1:
            for typ, count in type_data.items():
                if count > 0:
                    st.markdown(f"""
                    <div style='display:flex;justify-content:space-between;padding:8px 12px;
                         background:#f8f9fc;border-radius:8px;margin-bottom:6px;border:1px solid #e2e6f0;'>
                        <span style='font-size:0.88rem;color:#374151;'>{typ}</span>
                        <span style='font-family:"JetBrains Mono",monospace;font-weight:700;
                              color:#2563eb;'>{count}</span>
                    </div>
                    """, unsafe_allow_html=True)
        with tc2:
            fig_type = go.Figure(go.Pie(
                labels=list(type_data.keys()),
                values=list(type_data.values()),
                hole=0.5,
                marker_colors=["#2563eb","#f59e0b","#10b981","#8b5cf6","#6b7280"]
            ))
            fig_type.update_layout(
                template="plotly_white", height=280,
                margin=dict(t=20,b=20,l=20,r=20),
                paper_bgcolor="#ffffff",
                showlegend=True
            )
            st.plotly_chart(fig_type, use_container_width=True)

        st.markdown("<div class='section-header'><h3>Missing Value Heatmap</h3></div>", unsafe_allow_html=True)
        miss_matrix = df.isnull().astype(int)
        if miss_matrix.sum().sum() > 0:
            fig_miss = go.Figure(go.Heatmap(
                z=miss_matrix.T.values,
                x=list(range(len(df))),
                y=df.columns.tolist(),
                colorscale=[[0,"#f0fdf4"],[1,"#dc2626"]],
                showscale=False
            ))
            fig_miss.update_layout(
                title="Red = Missing Value",
                template="plotly_white", height=max(200, len(df.columns)*22),
                paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc",
                margin=dict(t=40,b=20,l=20,r=20)
            )
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.markdown("<span class='badge badge-success'>✅ No missing values — heatmap not needed</span>", unsafe_allow_html=True)

    with tab2:
        if not num_cols:
            st.info("No numerical columns found.")
        else:
            st.markdown("<div class='section-header'><h3>Extended Numerical Statistics</h3></div>", unsafe_allow_html=True)
            stats_df = descriptive_statistics(df)
            st.dataframe(stats_df, use_container_width=True, height=350)

            st.markdown("<div class='section-header'><h3>Skewness Summary</h3></div>", unsafe_allow_html=True)
            skew_df = calculate_skewness(df)
            def color_skew(val):
                if isinstance(val, str):
                    if "Highly" in val: return "color:#dc2626;font-weight:600"
                    if "Moderately" in val: return "color:#d97706;font-weight:600"
                    if "Normal" in val: return "color:#16a34a;font-weight:600"
                return ""
            st.dataframe(skew_df, use_container_width=True, height=300)

            st.markdown("<div class='section-header'><h3>Box Plots</h3></div>", unsafe_allow_html=True)
            n_cols_plot = len(num_cols)
            fig_box = make_subplots(rows=1, cols=n_cols_plot, subplot_titles=num_cols,
                horizontal_spacing=max(0.02, min(0.1, 0.8/max(n_cols_plot,1))))
            for i, col in enumerate(num_cols, start=1):
                fig_box.add_trace(go.Box(
                    y=df[col].dropna(), name=col,
                    marker_color="rgba(37,99,235,0.6)",
                    line_color="#2563eb",
                    boxpoints="outliers",
                    marker=dict(color="rgba(220,38,38,0.8)", size=5),
                    showlegend=False
                ), row=1, col=i)
            fig_box.update_layout(
                title="Box Plots — Red dots = Outliers",
                template="plotly_white", height=420,
                paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc"
            )
            st.plotly_chart(fig_box, use_container_width=True)

    with tab3:
        if not cat_cols:
            st.info("No categorical columns found.")
        else:
            st.markdown("<div class='section-header'><h3>Categorical Column Summary</h3></div>", unsafe_allow_html=True)
            cat_summary = []
            for col in cat_cols:
                top_val = df[col].value_counts().index[0] if df[col].nunique() > 0 else "—"
                top_pct = round(df[col].value_counts().iloc[0]/len(df)*100, 1) if df[col].nunique() > 0 else 0
                cat_summary.append({
                    "Column": col,
                    "Unique Values": df[col].nunique(),
                    "Most Frequent": str(top_val),
                    "Frequency %": top_pct,
                    "Missing %": round(df[col].isnull().sum()/len(df)*100, 2)
                })
            st.dataframe(pd.DataFrame(cat_summary), use_container_width=True, height=300)

            st.markdown("<div class='section-header'><h3>Value Counts</h3></div>", unsafe_allow_html=True)
            sel_cat = st.selectbox("Select column", cat_cols, key="eda_cat_col")
            vc = df[sel_cat].value_counts().head(20)
            fig_vc = go.Figure(go.Bar(
                x=vc.index.astype(str), y=vc.values,
                marker_color="#2563eb",
                text=vc.values, textposition="outside"
            ))
            fig_vc.update_layout(
                title=f"Value Counts: {sel_cat}",
                template="plotly_white", height=350,
                paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc"
            )
            st.plotly_chart(fig_vc, use_container_width=True)

    with tab4:
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            st.info("Need at least 2 numerical columns for correlation.")
        else:
            st.markdown("<div class='section-header'><h3>Correlation Heatmap</h3></div>", unsafe_allow_html=True)
            corr = num_df.corr().round(2)
            cols_list = corr.columns.tolist()

            corr_display = corr.copy().astype(float)
            for i in range(len(cols_list)):
                corr_display.iloc[i, i] = float('nan')

            annotations = []
            for i, row in enumerate(cols_list):
                for j, col in enumerate(cols_list):
                    v = corr.loc[row, col]
                    is_diag = (row == col)
                    annotations.append(dict(
                        x=col, y=row,
                        text="1" if is_diag else f"{v:.2f}",
                        showarrow=False,
                        font=dict(
                            size=11,
                            color="#1e293b" if is_diag else (
                                "#ffffff" if v < -0.3 else "#1e293b"
                            )
                        )
                    ))

            fig_corr = go.Figure(go.Heatmap(
                z=corr_display.values,
                x=cols_list,
                y=cols_list,
                zmin=-1, zmax=1,
                colorscale=[
                    [0.0,  "#081d58"],
                    [0.2,  "#253494"],
                    [0.4,  "#225ea8"],
                    [0.6,  "#41b6c4"],
                    [0.8,  "#a1dab4"],
                    [1.0,  "#ffffcc"],
                ],
                colorbar=dict(
                    tickvals=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    ticktext=["−1.0", "−0.6", "−0.2", "0.2", "0.6", "1.0"],
                    tickfont=dict(color="#475569", size=10),
                    thickness=10,
                    len=0.8,
                    outlinewidth=0,
                    bgcolor="#f8fafc",
                ),
                hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
                xgap=4,
                ygap=4,
            ))

            for i, col in enumerate(cols_list):
                fig_corr.add_shape(
                    type="rect",
                    x0=i - 0.5, x1=i + 0.5,
                    y0=i - 0.5, y1=i + 0.5,
                    fillcolor="#ffffcc",
                    line=dict(color="#ffffcc", width=0),
                    layer="above"
                )

            fig_corr.update_layout(
                annotations=annotations,
                template=None,
                height=max(420, len(cols_list) * 58 + 140),
                paper_bgcolor="#f8fafc",
                plot_bgcolor="#f8fafc",
                font=dict(color="#475569", family="DM Sans, sans-serif"),
                title=dict(
                    text="Correlation Heatmap — All vs All",
                    font=dict(color="#1e293b", size=13)
                ),
                xaxis=dict(
                    tickangle=-40,
                    tickfont=dict(size=11, color="#64748b"),
                    showgrid=False,
                    linecolor="#e2e8f0",
                    side="bottom",
                    fixedrange=True,
                ),
                yaxis=dict(
                    tickfont=dict(size=11, color="#64748b"),
                    showgrid=False,
                    linecolor="#e2e8f0",
                    autorange="reversed",
                    fixedrange=True,
                ),
                hoverlabel=dict(
                    bgcolor="#ffffff",
                    bordercolor="#e2e8f0",
                    font=dict(color="#1e293b", size=12)
                ),
                margin=dict(t=48, b=110, l=140, r=60),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            # Strong correlations summary
            strong = []
            for i in range(len(cols_list)):
                for j in range(i+1, len(cols_list)):
                    v = corr.iloc[i, j]
                    if abs(v) >= 0.7:
                        strong.append({
                            "Column A": cols_list[i],
                            "Column B": cols_list[j],
                            "r": round(float(v), 4),
                            "Strength": "Strong positive" if v >= 0.7 else "Strong negative"
                        })
            if strong:
                st.markdown("<div class='section-header'><h3>Strong Correlations ( |r| ≥ 0.7 )</h3></div>", unsafe_allow_html=True)
                strong_df = pd.DataFrame(strong).sort_values("r", key=abs, ascending=False)
                st.dataframe(strong_df, use_container_width=True, hide_index=True)
            else:
                st.markdown("<span class='badge badge-success'>✅ No strong correlations found ( |r| < 0.7 )</span>", unsafe_allow_html=True)
    nav_buttons("Statistics & EDA")
# ═══════════════════════════════════════════════
# PAGE 3 — RECOMMENDATIONS
# ═══════════════════════════════════════════════
elif st.session_state.page == "Recommendations":
    st.markdown("""
    <div class='main-header'>
        <h1>💡 Recommendations</h1>
        <p>Smart suggestions for every column based on data analysis</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = st.session_state.df
    ct = identify_column_types(df)
    outlier_data = detect_outliers_iqr(df)
    miss_report  = missing_value_report(df)
    miss_dict    = {r["column"]: r for r in miss_report}
    already_transformed = set(st.session_state.get("transformed_columns", []))
    transformed_suffixes = ("_log", "_sqrt", "_boxcox")
    skew_df      = calculate_skewness(df)
    skew_df      = skew_df[
        ~skew_df["Column"].isin(already_transformed) &
        ~skew_df["Column"].str.endswith(transformed_suffixes)
    ]
    skew_dict    = dict(zip(skew_df["Column"], skew_df["Skewness"]))
    dup_count    = df.duplicated().sum()
    inv_vals     = detect_invalid_values(df)
    neg_vals     = detect_negative_values(df)
    encoded_set  = set(st.session_state.get("encoded_columns", []))
    target_col_r = st.session_state.get("target_col", "— None —")
    
    def _is_ohe_dummy(col, encoded_set):
        """Return True if col looks like a one-hot dummy of an already-encoded column."""
        for enc_col in encoded_set:
            if col.startswith(enc_col + "_"):
                return True
        return False
    
    enc_cols = [
        c for c in ct["categorical"] + ct["boolean"]
        if c not in encoded_set
        and c != target_col_r
        and not _is_ohe_dummy(c, encoded_set)
    ]
    cols_with_out = {c: v for c, v in outlier_data.items() if v["count"] > 0}
    skewed       = [(c, s) for c, s in skew_dict.items() if abs(s) > 0.5]
    num_df       = df.select_dtypes(include=[np.number])

    # ── Summary strip ──
    has_missing  = len(miss_dict) > 0
    has_dupes    = dup_count > 0
    has_outliers = len(cols_with_out) > 0
    has_skew     = len(skewed) > 0
    has_enc      = len(enc_cols) > 0
    has_invalid  = len(inv_vals) > 0 or len(neg_vals) > 0

    s1,s2,s3,s4,s5,s6 = st.columns(6)
    for widget, label, val, ok in [
        (s1, "Duplicates",   str(dup_count),          not has_dupes),
        (s2, "Missing Cols", str(len(miss_dict)),      not has_missing),
        (s3, "Outlier Cols", str(len(cols_with_out)),  not has_outliers),
        (s4, "Skewed Cols",  str(len(skewed)),         not has_skew),
        (s5, "To Encode",    str(len(enc_cols)),       not has_enc),
        (s6, "Invalid",      str(len(inv_vals)+len(neg_vals)), not has_invalid),
    ]:
        color = "#16a34a" if ok else "#dc2626"
        with widget:
            st.markdown(f"""
            <div style='background:#fff;border:1px solid #e2e6f0;border-radius:10px;
                 padding:14px;text-align:center;'>
                <div style='font-family:"JetBrains Mono",monospace;font-size:1.6rem;
                     font-weight:700;color:{color};'>{val}</div>
                <div style='font-size:0.7rem;color:#6b7280;margin-top:4px;
                     text-transform:uppercase;letter-spacing:1px;'>{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("&nbsp;")

    # ── Tabs per category ──
    rt1,rt2,rt3,rt4,rt5,rt6,rt7 = st.tabs([
        "🗑️ Duplicates","❓ Missing","📦 Outliers",
        "〰️ Skewness","🔠 Encoding","✔️ Invalid","📋 Checklist"
    ])

    def rec_row(icon, title, detail, fix, status):
        color = {"ok":"#16a34a","warn":"#d97706","bad":"#dc2626"}.get(status,"#d97706")
        border = {"ok":"#bbf7d0","warn":"#fde68a","bad":"#fecaca"}.get(status,"#fde68a")
        bg     = {"ok":"#f0fdf4","warn":"#fffbeb","bad":"#fef2f2"}.get(status,"#fffbeb")
        st.markdown(f"""
        <div style='background:{bg};border-left:4px solid {color};border:1px solid {border};
             border-radius:10px;padding:16px 20px;margin-bottom:10px;'>
            <div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>
                <span style='font-size:1.1rem;'>{icon}</span>
                <span style='font-weight:600;font-size:0.95rem;color:#111827;'>{title}</span>
                <span style='margin-left:auto;background:{"#dcfce7" if status=="ok" else "#fef9c3" if status=="warn" else "#fee2e2"};
                      color:{color};border-radius:20px;padding:2px 10px;
                      font-size:0.72rem;font-weight:600;'>
                      {"✅ OK" if status=="ok" else "⚠️ Review" if status=="warn" else "❌ Action Needed"}
                </span>
            </div>
            <div style='font-size:0.85rem;color:#374151;margin-bottom:6px;'>{detail}</div>
            <div style='font-size:0.82rem;color:#6b7280;'><b style='color:#374151;'>Fix →</b> {fix}</div>
        </div>
        """, unsafe_allow_html=True)

    with rt1:
        if has_dupes:
            rec_row("🗑️", f"{dup_count} duplicate rows ({round(dup_count/len(df)*100,2)}%)",
                "Duplicates inflate model performance and skew statistics.",
                "Go to 🧹 Cleaning → Duplicates tab → Remove All", "warn")
        else:
            rec_row("✅", "No duplicate rows", "All rows are unique.", "No action needed.", "ok")

    with rt2:
        if not has_missing:
            rec_row("✅", "No missing values", "All columns are complete.", "No action needed.", "ok")
        else:
            for col, m in miss_dict.items():
                if m["pct"] > 30:
                    fix = "Consider dropping this column — too much data missing"
                    status = "bad"
                elif col in ct["numerical"]:
                    fix = "Fill with Median — robust to outliers for numerical data"
                    status = "warn"
                else:
                    fix = "Fill with Mode — most frequent value for categorical data"
                    status = "warn"
                rec_row("❓", f"'{col}' — {m['missing']} missing ({m['pct']}%)",
                    f"Column type: {m['type']}", fix, status)

    with rt3:
        if not has_outliers:
            rec_row("✅", "No outliers detected", "All numerical columns within IQR bounds.", "No action needed.", "ok")
        else:
            for col, o in cols_with_out.items():
                fix = "Cap (Winsorise) — preserves rows" if o["pct"] > 10 else "Remove outlier rows"
                status = "bad" if o["pct"] > 10 else "warn"
                rec_row("📦", f"'{col}' — {o['count']} outliers ({o['pct']}%)",
                    f"IQR bounds: [{round(o['lower'],2)}, {round(o['upper'],2)}]  |  Mean: {round(o['mean'],2)}  |  Std: {round(o['std'],2)}",
                    f"{fix} → Go to 🔠 Encoding & Outliers → Outliers tab", status)

    with rt4:
        if not skewed:
            rec_row("✅", "No significant skewness", "All columns approximately normal.", "No action needed.", "ok")
        else:
            for col, sk in skewed:
                direction = "right skewed (+)" if sk > 0 else "left skewed (−)"
                label = "Highly skewed" if abs(sk) > 1 else "Moderately skewed"
                status = "bad" if abs(sk) > 1 else "warn"
                fix = "Log or Box-Cox transform → Go to 🔠 Encoding & Outliers → Skewness tab" if abs(sk) > 1 else "Sqrt transform → Go to 🔠 Encoding & Outliers → Skewness tab"
                rec_row("〰️", f"'{col}' — {label} (skewness = {sk:.4f})",
                    f"{direction}. Affects regression and mean-based models.", fix, status)

    with rt5:
            all_cat = [c for c in ct["categorical"] + ct["boolean"]]
            target_col_r = st.session_state.get("target_col", "— None —")
            target_encoded = st.session_state.get("target_encoded", False)
    
            # Show target status
            if target_col_r != "— None —":
                if target_encoded:
                    rec_row("✅", f"Target '{target_col_r}' encoded",
                        "Target variable has been encoded this session.",
                        "No action needed.", "ok")
                else:
                    rec_row("🎯", f"Target '{target_col_r}' — not yet encoded",
                        "Target variable should be encoded before training.",
                        "Go to 🔠 Encoding & Outliers → Encoding tab → Select Target", "warn")
    
            if not enc_cols:
                if not all_cat:
                    rec_row("✅", "No categorical columns", "All columns are numerical.", "No action needed.", "ok")
                else:
                    rec_row("✅", "All categorical feature columns encoded",
                        f"{len(all_cat)} column(s) fully processed.",
                        "No action needed.", "ok")
            else:
                for col in enc_cols:
                    enc, exp = recommend_encoding(df, col)
                    n = df[col].nunique()
                    rec_row("🔠", f"'{col}' — {n} unique values",
                        f"Column type: {'Boolean' if col in ct['boolean'] else 'Categorical'}",
                        f"Use {enc.upper()} encoding — {exp} → Go to 🔠 Encoding tab", "warn")

    with rt6:
        if not has_invalid:
            rec_row("✅", "No invalid values", "All values pass domain validation.", "No action needed.", "ok")
        else:
            for col, info in inv_vals.items():
                rec_row("⚠️", f"'{col}' — {info['issue']}",
                    f"{info['count']} rows with invalid values",
                    "Use Custom Range Validation → Go to 🧹 Cleaning → Validation tab", "bad")
            for col, info in neg_vals.items():
                rec_row("➖", f"'{col}' — {info['count']} negative values",
                    info["reason"],
                    "Remove or cap rows → Go to 🧹 Cleaning → Validation tab", "warn")

    with rt7:
        checklist = [
            ("Check shape & dtypes",                    True),
            ("Handle missing values",                   not has_missing),
            ("Remove duplicate rows",                   not has_dupes),
            ("Detect & treat outliers",                 not has_outliers),
            ("Check skewness & transform",              not has_skew),
            ("Encode categorical columns",              not has_enc),
            ("Check correlation / multicollinearity",   num_df.shape[1] >= 2),
            ("Fix invalid / negative values",           not has_invalid),
            ("Export cleaned dataset",                st.session_state.get("export_done", False))
        ]
        done_count  = sum(1 for _, d in checklist if d)
        total_count = len(checklist)
        pct_done    = round(done_count/total_count*100)
        st.markdown(f"""
        <div style='margin-bottom:16px;'>
            <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
                <span style='font-size:0.85rem;color:#374151;font-weight:600;'>Progress</span>
                <span style='font-family:"JetBrains Mono",monospace;font-size:0.85rem;
                      color:#2563eb;font-weight:700;'>{done_count}/{total_count}</span>
            </div>
            <div style='background:#e2e6f0;border-radius:8px;height:8px;'>
                <div style='width:{pct_done}%;height:100%;background:#2563eb;border-radius:8px;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        for task, done in checklist:
            color  = "#16a34a" if done else "#6b7280"
            badge  = "Done" if done else "Pending"
            bg     = "#f0fdf4" if done else "#f8f9fc"
            border = "#bbf7d0" if done else "#e2e6f0"
            bcolor = "#16a34a" if done else "#9ca3af"
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                 padding:12px 16px;background:{bg};border:1px solid {border};
                 border-radius:8px;margin-bottom:6px;'>
                <span style='font-size:0.88rem;color:{color};font-weight:{"600" if done else "400"};'>
                    {"✅" if done else "○"} {task}
                </span>
                <span style='font-size:0.72rem;color:{bcolor};font-weight:600;
                      background:{"#dcfce7" if done else "#f1f3f9"};
                      border-radius:20px;padding:2px 10px;border:1px solid {border};'>
                    {badge}
                </span>
            </div>
            """, unsafe_allow_html=True)

    nav_buttons("Recommendations")
# ═══════════════════════════════════════════════
# PAGE 4 — CLEANING & VALIDATION
# ═══════════════════════════════════════════════
elif st.session_state.page == "Cleaning & Validation":
    st.markdown("""
    <div class='main-header'>
        <h1>🧹 Cleaning & Validation</h1>
        <p>Remove duplicates, fix missing values, and detect data anomalies</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first on the Upload & Inspect page.")
        st.stop()

    df = st.session_state.df
    tab1, tab2, tab3 = st.tabs(["🗑️ Duplicates","🔧 Missing Values","✔️ Validation"])

    with tab1:
        st.markdown("<div class='section-header'><h3>Duplicate Row Detection</h3></div>", unsafe_allow_html=True)
        dupe_rows = df[df.duplicated(keep="first")]
        n_dupes = len(dupe_rows)
        if n_dupes > 0:
            pct_dupes = round(n_dupes/len(df)*100, 2)
            c1d, c2d, c3d = st.columns(3)
            with c1d: st.markdown(f"""<div class='metric-card'><span class='val'>{n_dupes:,}</span><span class='label'>Duplicate Rows</span></div>""", unsafe_allow_html=True)
            with c2d: st.markdown(f"""<div class='metric-card'><span class='val'>{pct_dupes}%</span><span class='label'>Of Dataset</span></div>""", unsafe_allow_html=True)
            with c3d: st.markdown(f"""<div class='metric-card'><span class='val'>{len(df)-n_dupes:,}</span><span class='label'>Unique Rows</span></div>""", unsafe_allow_html=True)
            st.markdown("&nbsp;")
            st.dataframe(dupe_rows, use_container_width=True, height=300)
            if st.button("🗑️ Remove All Duplicates"):
                before = len(df)
                st.session_state.df = df.drop_duplicates(keep="first").reset_index(drop=True)
                removed = before - len(st.session_state.df)
                save_operation(st.session_state.file_name, "Remove Duplicates", f"Removed {removed} rows")
                st.success(f"✅ Removed {removed} duplicate rows.")
                st.rerun()
        else:
            st.markdown("<span class='badge badge-success'>✅ No duplicates found</span>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='section-header'><h3>Missing Value Treatment</h3></div>", unsafe_allow_html=True)
        report = missing_value_report(df)
        if not report:
            st.markdown("<span class='badge badge-success'>✅ No missing values</span>", unsafe_allow_html=True)
        else:
            for item in report:
                with st.expander(f"**{item['column']}** — {item['missing']} missing ({item['pct']}%) [{item['type']}]"):
                    chosen = st.selectbox(f"Strategy for {item['column']}", item["strategies"], key=f"strat_{item['column']}")
                    custom_val = None
                    if chosen == "custom":
                        custom_val = st.text_input("Custom fill value", key=f"custom_{item['column']}")
                    if st.button(f"Apply to {item['column']}", key=f"apply_{item['column']}"):
                        try:
                            new_df, stats_r = fill_missing_values(df, item["column"], chosen, custom_val)
                            st.session_state.df = new_df
                            save_operation(st.session_state.file_name, f"Fill Missing: {item['column']}", f"{chosen} – filled {stats_r['filled']}")
                            st.success(f"Filled {stats_r['filled']} values using '{chosen}'.")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

    with tab3:
        st.markdown("<div class='section-header'><h3>Data Validation & Anomaly Detection</h3></div>", unsafe_allow_html=True)
    
        # ── User-defined range validation ──
        st.markdown("**🎯 Custom Range Validation (Numerical Columns)**")
        num_cols_v = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols_v:
            range_col = st.selectbox("Select column to validate", ["— Select —"] + num_cols_v, key="range_val_col")
            if range_col != "— Select —":
                col_min = float(df[range_col].min())
                col_max = float(df[range_col].max())
                st.caption(f"Actual data range: `{col_min}` → `{col_max}`")
                # reset defaults only when column changes
                if st.session_state.get("_range_col_prev") != range_col:
                    st.session_state["range_min"] = col_min
                    st.session_state["range_max"] = col_max
                    st.session_state["_range_col_prev"] = range_col
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        user_min = st.number_input("Minimum allowed value", key="range_min")
                    with rc2:
                        user_max = st.number_input("Maximum allowed value", key="range_max")
                        if user_min >= user_max:
                            st.error("❌ Min must be less than Max.")
                        else:
                            out_of_range = df[(df[range_col] < user_min) | (df[range_col] > user_max)]
                            if len(out_of_range) > 0:
                                st.markdown(f"<span class='badge badge-danger'>❌ {len(out_of_range)} rows out of range [{user_min}, {user_max}]</span>", unsafe_allow_html=True)
                                with st.expander(f"Show {len(out_of_range)} invalid rows"):
                                    st.dataframe(out_of_range, use_container_width=True)
                                if st.button("🗑️ Remove out-of-range rows", key="remove_range"):
                                    before = len(df)
                                    st.session_state.df = df[
                                        (df[range_col] >= user_min) & (df[range_col] <= user_max)
                                    ].reset_index(drop=True)
                                    save_operation(st.session_state.file_name, f"Range Filter: {range_col}", f"Removed {before - len(st.session_state.df)} rows outside [{user_min},{user_max}]")
                                    st.success(f"✅ Removed {before - len(st.session_state.df)} rows.")
                                    st.rerun()
                            else:
                                st.markdown(f"<span class='badge badge-success'>✅ All values within [{user_min}, {user_max}]</span>", unsafe_allow_html=True)
    
        st.markdown("---")
        v1, v2 = st.columns(2)
        with v1:
            st.markdown("**⚠️ Invalid Domain Values**")
            inv = detect_invalid_values(df)
            for col, info in inv.items():
                st.markdown(f"<span class='badge badge-danger'>{col}</span> {info['issue']} — {info['count']} rows", unsafe_allow_html=True)
            if not inv: st.markdown("<span class='badge badge-success'>✅ None detected</span>", unsafe_allow_html=True)
            st.markdown("&nbsp;")
            st.markdown("**📧 Invalid Emails**")
            emails = detect_invalid_email(df)
            for col, info in emails.items():
                st.markdown(f"<span class='badge badge-warning'>{col}</span> {info['count']} invalid emails", unsafe_allow_html=True)
            if not emails: st.markdown("<span class='badge badge-success'>✅ None detected</span>", unsafe_allow_html=True)
        with v2:
            st.markdown("**➖ Negative Values (domain-aware)**")
            negs = detect_negative_values(df)
            for col, info in negs.items():
                st.markdown(f"<span class='badge badge-warning'>{col}</span> {info['count']} negative rows — {info['reason']}", unsafe_allow_html=True)
            if not negs: st.markdown("<span class='badge badge-success'>✅ None detected</span>", unsafe_allow_html=True)
            st.markdown("&nbsp;")
            st.markdown("**📞 Invalid Phone Numbers**")
            phones = detect_invalid_phone(df)
            for col, info in phones.items():
                st.markdown(f"<span class='badge badge-warning'>{col}</span> {info['count']} invalid", unsafe_allow_html=True)
            if not phones: st.markdown("<span class='badge badge-success'>✅ None detected</span>", unsafe_allow_html=True)

        st.markdown("&nbsp;")
        st.markdown("**📅 Future Dates**")
        future = detect_future_dates(df)
        for col, info in future.items():
            st.markdown(f"<span class='badge badge-danger'>{col}</span> {info['count']} future dates", unsafe_allow_html=True)
        if not future: st.markdown("<span class='badge badge-success'>✅ No future dates detected</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<div class='section-header'><h3>Valid / Invalid Row Segregation</h3></div>", unsafe_allow_html=True)

        def _build_invalid_mask(df):
            mask = pd.Series(False, index=df.index)
            for col in df.columns:
                cl = col.lower()
                if pd.api.types.is_numeric_dtype(df[col]):
                    if any(k in cl for k in ["age"]):
                        mask |= (df[col] < 0) | (df[col] > 150)
                    if any(k in cl for k in ["pct","percent"]) and "growth" not in cl:
                        mask |= (df[col] < 0) | (df[col] > 100)
                    if is_non_negative_column(col):
                        mask |= df[col] < 0
            pat_email = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$")
            for col in df.select_dtypes(include="object").columns:
                if any(k in col.lower() for k in ["email","mail"]):
                    bad = df[col].dropna().apply(lambda x: not bool(pat_email.match(str(x))))
                    mask.loc[bad[bad].index] = True
            now = pd.Timestamp.now()
            for col in df.columns:
                if any(k in col.lower() for k in ["birth","dob","born","date","created","joined"]):
                    try:
                        parsed = pd.to_datetime(df[col], errors="coerce")
                        mask |= (parsed > now).fillna(False)
                    except: pass
            return mask

        invalid_mask = _build_invalid_mask(df)
        valid_df = df[~invalid_mask]; invalid_df = df[invalid_mask]
        ci1, ci2 = st.columns(2)
        with ci1: st.markdown(f"""<div class='metric-card'><span class='val' style='color:#16a34a;'>{len(valid_df):,}</span><span class='label'>Valid Rows</span></div>""", unsafe_allow_html=True)
        with ci2: st.markdown(f"""<div class='metric-card'><span class='val' style='color:#dc2626;'>{len(invalid_df):,}</span><span class='label'>Invalid Rows</span></div>""", unsafe_allow_html=True)
        st.markdown("&nbsp;")
        with st.expander(f"✅ Valid Rows ({len(valid_df):,})", expanded=False):
            st.dataframe(valid_df.head(200), use_container_width=True, height=320)
        if len(invalid_df) > 0:
            with st.expander(f"❌ Invalid Rows ({len(invalid_df):,})", expanded=True):
                st.dataframe(invalid_df, use_container_width=True, height=320)
                c_rm, c_exp, _ = st.columns(3)
                with c_rm:
                    if st.button("🗑️ Remove Invalid Rows"):
                        before = len(df)
                        st.session_state.df = valid_df.reset_index(drop=True)
                        save_operation(st.session_state.file_name, "Remove Invalid Rows", f"Removed {before-len(valid_df)} rows")
                        st.success(f"✅ Removed {before-len(valid_df)} invalid rows.")
                        st.rerun()
                with c_exp:
                    st.download_button("⬇️ Export Invalid Rows",
                                       data=invalid_df.to_csv(index=False).encode("utf-8"),
                                       file_name="invalid_rows.csv", mime="text/csv")
    nav_buttons("Cleaning & Validation")
# ═══════════════════════════════════════════════
# PAGE 5 — ENCODING & OUTLIERS
# ═══════════════════════════════════════════════
elif st.session_state.page == "Encoding & Outliers":
    st.markdown("""
    <div class='main-header'>
        <h1>🔠 Encoding & Outliers</h1>
        <p>Encode categorical features, detect outliers, and analyse distributions</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = st.session_state.df
    ct = identify_column_types(df)
    tab1, tab2, tab3 = st.tabs(["🔡 Encoding","📦 Outliers","〰️ Skewness"])

    # ── Encoding  ── FIX 2: exclude already-encoded columns ──────────────────
    with tab1:
        st.markdown("<div class='section-header'><h3>Target-First Categorical Encoding</h3></div>", unsafe_allow_html=True)

        encoding_df = st.session_state.original_df
        all_cols = list(st.session_state.original_df.columns)
        ct = identify_column_types(encoding_df)
        if "target_col" not in st.session_state:
            st.session_state.target_col = "— None —"
        saved_target = st.session_state.get("target_col", "— None —")
        already_encoded = st.session_state.get("target_encoded", False)
        
        if already_encoded and saved_target != "— None —":
            target_col = saved_target
        else:
            target_options = ["— None —"] + all_cols
            target_index = target_options.index(saved_target) if saved_target in target_options else 0
            target_col = st.selectbox("Select Target Variable", target_options, index=target_index, key="target_col_widget")
            if target_col != st.session_state.get("target_col"):
                st.session_state.target_col = target_col
                st.session_state.target_encoded = False
        if target_col != "— None —":
            already_encoded = st.session_state.get("target_encoded", False)
        
            if already_encoded:
                st.markdown(f"""
                <div style='display:flex;align-items:center;gap:10px;padding:12px 16px;
                     background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;margin:12px 0;'>
                    <span style='font-size:1.1rem;'>🎯</span>
                    <span style='font-size:0.9rem;color:#374151;'>
                        Target variable: <code style='background:#dcfce7;padding:2px 6px;border-radius:4px;
                        color:#16a34a;font-weight:700;'>{target_col}</code>
                        &nbsp;— locked for training. Encoding applied.
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                rec_enc, rec_exp = recommend_encoding(st.session_state.original_df, target_col, is_target=True)
                st.markdown(f"""
                <div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:14px;margin:12px 0;'>
                    <b style='color:#2563eb;'>Target:</b> <code>{target_col}</code> &nbsp;|&nbsp;
                    <b style='color:#2563eb;'>Recommended:</b> <code>{rec_enc}</code><br>
                    <span style='color:#374151;font-size:0.9rem;'>{rec_exp}</span>
                </div>
                """, unsafe_allow_html=True)
                chosen_enc_t = st.selectbox("Encoding method for target",
                                             ["label","onehot","ordinal","frequency"],
                                             index=["label","onehot","ordinal","frequency"].index(rec_enc),
                                             key="enc_target_method")
                ordinal_order_t = None
                if chosen_enc_t == "ordinal":
                    ord_str = st.text_input("Ordinal order (comma-separated, low→high)", key="ord_target")
                    if ord_str: ordinal_order_t = [x.strip() for x in ord_str.split(",")]
                if st.button(f"Apply Encoding to Target: {target_col}"):
                    try:
                        new_df, mapping = apply_encoding(df, target_col, chosen_enc_t, ordinal_order_t)
                        st.session_state.processed_df = new_df
                        st.session_state.df = new_df
                        if target_col not in st.session_state.encoded_columns:
                            st.session_state.encoded_columns.append(target_col)
                        st.session_state.target_encoded = True
                        save_operation(st.session_state.file_name, f"Encoding: {target_col}", chosen_enc_t)
                        st.success(f"✅ Applied {chosen_enc_t} encoding to '{target_col}'.")
                        if mapping is not None:
                            st.dataframe(mapping.head(20), use_container_width=True)
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
        st.markdown("---")
        st.markdown("**Feature Column Encoding**")

        # FIX 2: exclude columns that have already been encoded
        encoded_set = set(st.session_state.encoded_columns)
        def _is_ohe_dummy(col, encoded_set):
            for enc_col in encoded_set:
                if col.startswith(enc_col + "_"):
                    return True
            return False
        
        enc_candidates = [
            c for c in ct["categorical"] + ct["boolean"]
            if c != target_col
            and c not in encoded_set
            and not c.endswith("_encoded")
            and not _is_ohe_dummy(c, encoded_set)
        ]
        ordinal_candidates = [
            c for c in enc_candidates
            if encoding_df[c].dtype == "object"    
        ]
        total_feature_cats = len([c for c in ct["categorical"] + ct["boolean"] if c != target_col])
        done_count = len([c for c in encoded_set if c != target_col])

        if done_count > 0 and done_count < total_feature_cats:
            st.markdown(f"<span class='badge badge-info'>ℹ️ {done_count} of {total_feature_cats} feature columns already encoded — hidden below</span>", unsafe_allow_html=True)

         # Recommendation buckets
        onehot_cols = []
        label_cols = []
        frequency_cols = []
        if not enc_candidates:
            if encoded_set:
                st.success("✅ All categorical feature columns have been encoded.")
            else:
                st.info("No categorical feature columns found.")
        else:
            for col in enc_candidates:
                rec, _ = recommend_encoding(st.session_state.original_df, col)
                if rec == "onehot":
                    onehot_cols.append(col)
                elif rec == "label":
                    label_cols.append(col)
                elif rec == "frequency":
                    frequency_cols.append(col)
            
            # ─────────────────────────────────────
            # Ordinal Encoding
            # ─────────────────────────────────────
            if True:
                st.subheader("📈 Ordinal Encoding")
                ord_col = st.selectbox(
                    "Select column for Ordinal Encoding",
                    ["— Select —"] + enc_candidates,
                    key="ordinal_select"
                )
                if ord_col != "— Select —":
                    existing_vals = sorted(encoding_df[ord_col].dropna().unique().tolist())
                    st.markdown(f"<b>Existing values in {ord_col}:</b> <code>{existing_vals}</code>", unsafe_allow_html=True)
                    ord_str = st.text_input(
                        "Enter order (comma-separated, low→high)",
                        placeholder="low,medium,high",
                        key="ordinal_order"
                    )
                    if st.button("Apply Ordinal Encoding"):
                        if not ord_str:
                            st.warning("Please enter the ordinal order.")
                        else:
                            # normalize both sides — strip + lowercase for comparison
                            ordinal_order = [x.strip() for x in ord_str.split(",")]
                            existing_stripped = [str(v).strip() for v in existing_vals]
                            invalid_vals = [v for v in ordinal_order if v.strip() not in existing_stripped]
                            missing_vals = [v for v in existing_stripped if v not in [x.strip() for x in ordinal_order]]
                            # remap ordinal_order to match exact original values
                            val_map = {str(v).strip(): v for v in existing_vals}
                            ordinal_order = [val_map.get(x.strip(), x.strip()) for x in ord_str.split(",")]                            
                            if invalid_vals:
                                st.error(f"❌ These values don't exist in column: `{invalid_vals}`")
                            elif missing_vals:
                                st.warning(f"⚠️ These column values are not in your order: `{missing_vals}`. They will become NaN.")
                            else:
                                new_df, mapping = apply_encoding(
                                    st.session_state.processed_df,
                                    ord_col, "ordinal", ordinal_order
                                )
                                st.session_state.processed_df = new_df
                                st.session_state.df = new_df
                                if ord_col not in st.session_state.encoded_columns:
                                    st.session_state.encoded_columns.append(ord_col)
                                st.success(f"✅ Ordinal encoding applied to '{ord_col}'")
                                st.rerun()
            # ─────────────────────────────────────
            # One-Hot Encoding
            # ─────────────────────────────────────
            if onehot_cols:
                st.subheader("🔵 One-Hot Encoding")
                selected_cols = st.multiselect(
                    "Select columns for One-Hot Encoding",
                    onehot_cols,
                    key="onehot_select"
                )
                if st.button("Apply One-Hot Encoding"):
                    for col in selected_cols:
                        new_df, mapping = apply_encoding(
                            st.session_state.processed_df,
                            col,
                            "onehot"
                        )
                        st.session_state.processed_df = new_df
                        st.session_state.df = new_df   # temporary compatibility
                        if col not in st.session_state.encoded_columns:
                            st.session_state.encoded_columns.append(col)
                    st.success(f"✅ Encoded {len(selected_cols)} column(s)")
                    st.rerun()
            # ─────────────────────────────────────
            # Label Encoding
            # ─────────────────────────────────────
            if label_cols:
                st.subheader("🏷️ Label Encoding")
                selected_cols = st.multiselect(
                    "Select columns for Label Encoding",
                    label_cols,
                    key="label_select"
                )
                if st.button("Apply Label Encoding"):
                    for col in selected_cols:
                        new_df, mapping = apply_encoding(
                            st.session_state.processed_df,
                            col,
                            "label"
                        )
                        st.session_state.processed_df = new_df
                        st.session_state.df = new_df   # temporary compatibility
                        if col not in st.session_state.encoded_columns:
                            st.session_state.encoded_columns.append(col)        
                    st.success(f"✅ Encoded {len(selected_cols)} column(s)")
                    st.rerun()
            # ─────────────────────────────────────
            # Frequency Encoding
            # ─────────────────────────────────────
            if frequency_cols:
                st.subheader("📊 Frequency Encoding")
                selected_cols = st.multiselect(
                    "Select columns for Frequency Encoding",
                    frequency_cols,
                    key="freq_select"
                )
                if st.button("Apply Frequency Encoding"):
                    for col in selected_cols:
                        new_df, mapping = apply_encoding(
                            st.session_state.processed_df,
                            col,
                            "frequency"
                        )
                        st.session_state.processed_df = new_df
                        st.session_state.df = new_df   # temporary compatibility
                        if col not in st.session_state.encoded_columns:
                            st.session_state.encoded_columns.append(col)
                    st.success(f"✅ Encoded {len(selected_cols)} column(s)")
                    st.rerun()
    # ── Outliers  ── FIX 3: go.Strip → go.Box + go.Scatter overlay ──────────
    with tab2:
        st.markdown("<div class='section-header'><h3>Outlier Detection & Treatment</h3></div>", unsafe_allow_html=True)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.info("No numerical columns found.")
        else:
            method_choice = st.radio("Detection Method", ["IQR (Interquartile Range)","Z-Score"], horizontal=True)
            use_iqr = "IQR" in method_choice

            if use_iqr:
                outlier_data = detect_outliers_iqr(df)
                st.markdown("""
                <div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:16px;margin:12px 0;'>
                    <b style='color:#2563eb;'>IQR Method</b><br><br>
                    Q1 = 25th pct &nbsp;|&nbsp; Q3 = 75th pct &nbsp;|&nbsp; IQR = Q3 − Q1<br>
                    Lower = Q1 − 1.5×IQR &nbsp;|&nbsp; Upper = Q3 + 1.5×IQR
                </div>
                """, unsafe_allow_html=True)
                stats_rows = [{"Column":c,"Q1":round(i["Q1"],3),"Q3":round(i["Q3"],3),
                                "IQR":round(i["IQR"],3),"Lower":round(i["lower"],3),
                                "Upper":round(i["upper"],3),"Outliers":i["count"],"Outlier %":i["pct"]}
                               for c,i in outlier_data.items()]
            else:
                outlier_data = detect_outliers_zscore(df)
                st.markdown("""
                <div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:16px;margin:12px 0;'>
                    <b style='color:#2563eb;'>Z-Score Method</b><br><br>
                    Z = (x − mean) / std &nbsp;|&nbsp; Values with |Z| &gt; 3 are outliers.
                </div>
                """, unsafe_allow_html=True)
                stats_rows = [{"Column":c,"Mean":round(i["mean"],3),"Std":round(i["std"],3),
                                "Threshold":f"|Z|>{i['threshold']}","Outliers":i["count"],"Outlier %":i["pct"]}
                               for c,i in outlier_data.items()]
           
            st.dataframe(pd.DataFrame(stats_rows),use_container_width=True)
            # ── FIX 3: Box plots (go.Box) + outlier scatter overlay ──
            st.markdown("<div class='section-header'><h3>Box Plot — Outliers Highlighted</h3></div>", unsafe_allow_html=True)
            st.caption("Blue boxes = normal distribution. Red dots = outlier values beyond whiskers.")
            try:
                n_cols_plot = len(num_cols)
                h_space = max(0.02, min(0.1, 0.8/max(n_cols_plot,1)))
                fig_box = make_subplots(
                    rows=1, cols=n_cols_plot,
                    subplot_titles=num_cols,
                    horizontal_spacing=h_space
                )
                legend_added = False
                for i, col in enumerate(num_cols, start=1):
                    info = outlier_data.get(col, {})
                    outlier_idx = set(info.get("rows", []))
                    series = df[col].dropna()
                    out_series = series[series.index.isin(outlier_idx)]

                    # Box trace — all data (whiskers auto-clip to IQR fence)
                    fig_box.add_trace(go.Box(
                        y=series,
                        name=col,
                        marker_color="rgba(37,99,235,0.55)",
                        line_color="#2563eb",
                        fillcolor="rgba(37,99,235,0.15)",
                        boxpoints=False,          # hide built-in point overlay
                        showlegend=not legend_added,
                        legendgroup="normal",
                        legendgrouptitle_text="" if legend_added else "Normal",
                    ), row=1, col=i)

                    # Scatter overlay for outlier points in red
                    if len(out_series) > 0:
                        fig_box.add_trace(go.Scatter(
                            y=out_series,
                            x=[col]*len(out_series),
                            mode="markers",
                            marker=dict(color="rgba(220,38,38,0.85)", size=7, symbol="circle-open"),
                            name="Outlier" if not legend_added else "",
                            legendgroup="outlier",
                            showlegend=not legend_added,
                        ), row=1, col=i)

                    legend_added = True

                fig_box.update_layout(
                    title="Box Plots — Outliers (Red ◯) vs Normal Range",
                    template="plotly_white",
                    height=max(420, 380),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#f8f9fc",
                    showlegend=True,
                )
                st.plotly_chart(fig_box, use_container_width=True)
            except Exception as e:
                st.error(f"Box plot error: {e}")

            # Density histogram overlay
            try:
                n_r = (len(num_cols)+2)//3
                fig_dens = make_subplots(rows=n_r, cols=3,
                    subplot_titles=num_cols, horizontal_spacing=0.08, vertical_spacing=0.12)
                for idx, col in enumerate(num_cols):
                    r, c = divmod(idx, 3)
                    info = outlier_data.get(col, {})
                    lo = info.get("lower", -np.inf); hi = info.get("upper", np.inf)
                    series = df[col].dropna()
                    normal = series[(series>=lo) & (series<=hi)]
                    outs   = series[(series<lo) | (series>hi)]
                    fig_dens.add_trace(go.Histogram(x=normal, nbinsx=25,
                        marker_color="rgba(37,99,235,0.5)", showlegend=False), row=r+1, col=c+1)
                    if len(outs)>0:
                        fig_dens.add_trace(go.Histogram(x=outs, nbinsx=10,
                            marker_color="rgba(220,38,38,0.7)", showlegend=False), row=r+1, col=c+1)
                fig_dens.update_layout(title="Density — Normal (Blue) vs Outliers (Red)",
                    template="plotly_white", barmode="overlay",
                    height=max(350, n_r*300), paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc")
                st.plotly_chart(fig_dens, use_container_width=True)
            except Exception as e:
                st.error(f"Density plot error: {e}")

            st.markdown("<div class='section-header'><h3>Treat Outliers</h3></div>", unsafe_allow_html=True)
            cols_with_outliers = [c for c in num_cols if outlier_data.get(c, {}).get("count", 0) > 0]
            if not cols_with_outliers:
                st.markdown("<span class='badge badge-success'>✅ No outliers found in any column</span>", unsafe_allow_html=True)
                selected_col = None
            else:
                selected_col = st.selectbox(
                    f"Select column to treat ({len(cols_with_outliers)} columns with outliers)",
                    cols_with_outliers,
                    key="out_treat_col")
            if selected_col:
                ca, cb, cc = st.columns(3)
                method_key = "iqr" if use_iqr else "zscore"
                with ca:
                    if st.button("🗑️ Remove Outliers"):
                        try:
                            new_df, r = remove_outliers(df, selected_col, method=method_key)
                            st.session_state.df = new_df
                            save_operation(st.session_state.file_name, f"Remove Outliers: {selected_col}", r)
                            st.success(f"Removed {r['removed']} outlier rows."); st.rerun()
                        except Exception as e: st.error(str(e))
                with cb:
                    if st.button("📌 Cap Outliers (Winsorise)"):
                        try:
                            new_df, r = cap_outliers(df, selected_col)
                            st.session_state.df = new_df
                            save_operation(st.session_state.file_name, f"Cap Outliers: {selected_col}", r)
                            st.success(f"Capped {r['capped']} outliers."); st.rerun()
                        except Exception as e: st.error(str(e))
                with cc:
                    st.info("Select Remove or Cap above.")

    with tab3:
        st.markdown("<div class='section-header'><h3>Skewness Analysis</h3></div>", unsafe_allow_html=True)
        skew_df = calculate_skewness(df)
        if skew_df.empty:
            st.info("No numerical columns.")
        else:
            st.dataframe(skew_df,use_container_width=True)
            transformed_suffixes = ("_log", "_sqrt", "_boxcox")
            already_transformed = set(st.session_state.get("transformed_columns", []))
            num_cols_sk = [
                c for c in df.select_dtypes(include=[np.number]).columns.tolist()
                if not c.endswith(transformed_suffixes)
                and c not in already_transformed
            ]
            n_r = (len(num_cols_sk)+1)//2
            try:
                fig_sk = make_subplots(rows=n_r, cols=2,
                    subplot_titles=[f"{r['Column']} | sk={r['Skewness']} ({r['Classification']})" for _,r in skew_df.iterrows()],
                    horizontal_spacing=0.08, vertical_spacing=0.14)
                cls_color = {"Highly Left Skewed":"#dc2626","Moderately Left Skewed":"#f97316",
                             "Approximately Normal":"#16a34a","Moderately Right Skewed":"#f59e0b",
                             "Highly Right Skewed":"#dc2626"}
                for idx, col in enumerate(num_cols_sk):
                    r, c = divmod(idx, 2)
                    data = df[col].dropna()
                    cls = skew_df.loc[skew_df["Column"]==col,"Classification"].values
                    color = cls_color.get(cls[0] if len(cls) else "", "#2563eb")
                    fig_sk.add_trace(go.Histogram(x=data, nbinsx=30,
                        marker_color="rgba(37,99,235,0.5)", showlegend=False), row=r+1, col=c+1)
                    try:
                        kde = stats.gaussian_kde(data)
                        x_r = np.linspace(data.min(), data.max(), 200)
                        kde_y = kde(x_r)*len(data)*(data.max()-data.min())/30
                        fig_sk.add_trace(go.Scatter(x=x_r, y=kde_y, mode="lines",
                            line=dict(color=color,width=2.5), showlegend=False), row=r+1, col=c+1)
                    except: pass
                    fig_sk.add_vline(x=float(data.mean()), line_dash="dash",
                        line_color="#dc2626", line_width=1.5, row=r+1, col=c+1)
                fig_sk.update_layout(title="Distribution + KDE (Red dashed = Mean)",
                    template="plotly_white", height=max(400,n_r*320),
                    paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc")
                st.plotly_chart(fig_sk, use_container_width=True)
            except Exception as e:
                st.error(f"Skewness chart error: {e}")

            st.markdown("**Apply Transformation:**")
            
            # Exclude already-transformed columns and approximately normal columns
            transformed_suffixes = ("_log", "_sqrt", "_boxcox")
            skew_dict_current = dict(zip(skew_df["Column"], skew_df["Skewness"]))
            
            skew_candidates = [
                c for c in num_cols_sk
                if not c.endswith(transformed_suffixes)
                and c not in already_transformed
                and abs(skew_dict_current.get(c, 0)) > 0.5
            ]
            
            if not skew_candidates:
                st.markdown("<span class='badge badge-success'>✅ No skewed columns remaining — all distributions are approximately normal</span>", unsafe_allow_html=True)
            else:
                skew_col = st.selectbox(
                    f"Column ({len(skew_candidates)} skewed remaining)",
                    skew_candidates,
                    key="skew_col"
                )
            transform = st.radio("Transformation", ["Log","Sqrt","Box-Cox"], horizontal=True)
            if st.button("Apply Transform"):
                try:
                    df_t = st.session_state.df.copy()
                    
                    if transform == "Log":
                        shift = abs(df_t[skew_col].min())+1 if df_t[skew_col].min()<=0 else 0
                        df_t[skew_col] = np.log(df_t[skew_col]+shift)
                    elif transform == "Sqrt":
                        shift = abs(df_t[skew_col].min()) if df_t[skew_col].min()<0 else 0
                        df_t[skew_col] = np.sqrt(df_t[skew_col]+shift)
                    elif transform == "Box-Cox":
                        s = df_t[skew_col].dropna()
                        shift = abs(s.min())+1 if s.min()<=0 else 0
                        t, _ = boxcox(s+shift)
                        df_t.loc[s.index, skew_col] = t

                    # Check if transform actually reduced skewness
                    new_skew = abs(df_t[skew_col].skew())
                    old_skew = abs(df[skew_col].skew())

                    st.session_state.df = df_t

                    # Track transformed columns
                    if skew_col not in st.session_state.transformed_columns:
                        st.session_state.transformed_columns.append(skew_col)

                    save_operation(
                        st.session_state.file_name,
                        f"{transform} Transform: {skew_col}",
                        f"skew {round(old_skew,4)} → {round(new_skew,4)}"
                    )

                    if new_skew <= 0.5:
                        st.success(f"✅ '{skew_col}' is now approximately normal (skew={round(new_skew,4)})")
                    elif new_skew < old_skew:
                        st.warning(f"⚠️ Skewness reduced but still present: {round(old_skew,4)} → {round(new_skew,4)}. Try a different transform.")
                    else:
                        st.error(f"❌ Transform did not help: skew={round(new_skew,4)}. Try Box-Cox instead.")

                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        nav_buttons("Encoding & Outliers")

# ═══════════════════════════════════════════════
# PAGE — VISUALIZATIONS 
# ═══════════════════════════════════════════════
elif st.session_state.page == "Visualizations":
    st.markdown("""
    <div class='main-header'>
        <h1>📊 Visualizations</h1>
        <p>Interactive charts</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = st.session_state.df
    ct = identify_column_types(df)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Custom Plot","🔵 Scatter","📦 Box & Violin",
        "📊 Category Breakdown"
    ])

    with tab1:
        st.markdown("<div class='section-header'><h3>Custom Chart Builder</h3></div>", unsafe_allow_html=True)
        cp1, cp2, cp3 = st.columns(3)
        with cp1:
            chart_type = st.selectbox("Chart Type", ["Line","Bar","Histogram","Area"], key="custom_chart_type")
        with cp2:
            x_col = st.selectbox("X Axis", df.columns.tolist(), key="custom_x")
        with cp3:
            y_col = st.selectbox("Y Axis", ["— None —"] + num_cols, key="custom_y")
        color_col = st.selectbox("Color by (optional)", ["— None —"] + cat_cols, key="custom_color")
        color_val = None if color_col == "— None —" else color_col
        y_val = None if y_col == "— None —" else y_col
        try:
            if chart_type == "Line":
                fig = px.line(df, x=x_col, y=y_val, color=color_val, template="plotly_white", height=420)
            elif chart_type == "Bar":
                fig = px.bar(df, x=x_col, y=y_val, color=color_val, template="plotly_white", height=420, barmode="group")
            elif chart_type == "Histogram":
                fig = px.histogram(df, x=x_col, color=color_val, template="plotly_white", height=420, nbins=30)
            elif chart_type == "Area":
                fig = px.area(df, x=x_col, y=y_val, color=color_val, template="plotly_white", height=420)
            fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {e}")

    with tab2:
        st.markdown("<div class='section-header'><h3>Scatter Plot</h3></div>", unsafe_allow_html=True)
        if len(num_cols) < 2:
            st.info("Need at least 2 numerical columns.")
        else:
            sc1, sc2, sc3 = st.columns(3)
            with sc1: sx = st.selectbox("X Axis", num_cols, key="scatter_x")
            with sc2: sy = st.selectbox("Y Axis", num_cols[::-1], key="scatter_y")
            with sc3: sc = st.selectbox("Color by", ["— None —"] + cat_cols + num_cols, key="scatter_color")
            sc_size = st.selectbox("Size by (optional)", ["— None —"] + num_cols, key="scatter_size")
            sc_color = None if sc == "— None —" else sc
            sc_size_val = None if sc_size == "— None —" else sc_size
            try:
                fig_sc = px.scatter(df, x=sx, y=sy, color=sc_color, size=sc_size_val,
                    template="plotly_white", height=450,
                    trendline="ols" if sc_color is None else None, opacity=0.7)
                fig_sc.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc")
                st.plotly_chart(fig_sc, use_container_width=True)
                r = df[[sx,sy]].dropna().corr().iloc[0,1]
                direction = "positive" if r > 0 else "negative"
                strength  = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"
                st.markdown(f"""
                <div style='background:#f8f9fc;border:1px solid #e2e6f0;border-radius:8px;
                     padding:12px 16px;font-size:0.88rem;color:#374151;'>
                    <b>Pearson r = {r:.4f}</b> — {strength} {direction} correlation
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Scatter error: {e}")

    with tab3:
        st.markdown("<div class='section-header'><h3>Box & Violin Plots</h3></div>", unsafe_allow_html=True)
        if not num_cols:
            st.info("No numerical columns.")
        else:
            bv1, bv2, bv3 = st.columns(3)
            with bv1: bv_col   = st.selectbox("Numerical column", num_cols, key="bv_col")
            with bv2: bv_group = st.selectbox("Group by (optional)", ["— None —"] + cat_cols, key="bv_group")
            with bv3: bv_type  = st.radio("Plot type", ["Box","Violin","Both"], key="bv_type", horizontal=True)
            bv_group_val = None if bv_group == "— None —" else bv_group
            try:
                if bv_type == "Box":
                    fig_bv = px.box(df, y=bv_col, x=bv_group_val, color=bv_group_val,
                        template="plotly_white", height=420, points="outliers")
                elif bv_type == "Violin":
                    fig_bv = px.violin(df, y=bv_col, x=bv_group_val, color=bv_group_val,
                        template="plotly_white", height=420, box=True, points="outliers")
                else:
                    fig_bv = make_subplots(rows=1, cols=2,
                        subplot_titles=[f"Box — {bv_col}", f"Violin — {bv_col}"])
                    for val in (df[bv_group_val].unique() if bv_group_val else [None]):
                        subset = df[df[bv_group_val]==val][bv_col].dropna() if bv_group_val else df[bv_col].dropna()
                        name   = str(val) if val is not None else bv_col
                        fig_bv.add_trace(go.Box(y=subset, name=name, boxpoints="outliers"), row=1, col=1)
                        fig_bv.add_trace(go.Violin(y=subset, name=name, box_visible=True), row=1, col=2)
                    fig_bv.update_layout(template="plotly_white", height=420, showlegend=False)
                fig_bv.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc")
                st.plotly_chart(fig_bv, use_container_width=True)
            except Exception as e:
                st.error(f"Plot error: {e}")

    with tab4:
        st.markdown("<div class='section-header'><h3>Category Breakdown</h3></div>", unsafe_allow_html=True)
        if not cat_cols:
            st.info("No categorical columns.")
        else:
            cb1, cb2, cb3 = st.columns(3)
            with cb1: cat_sel  = st.selectbox("Categorical column", cat_cols, key="cat_break_col")
            with cb2: num_sel  = st.selectbox("Numerical column", ["— Count —"] + num_cols, key="cat_break_num")
            with cb3: cat_type = st.radio("Chart", ["Bar","Pie","Donut"], key="cat_break_type", horizontal=True)
            try:
                if num_sel == "— Count —":
                    data = df[cat_sel].value_counts().reset_index()
                    data.columns = [cat_sel, "Count"]
                else:
                    data = df.groupby(cat_sel)[num_sel].mean().reset_index()
                    data.columns = [cat_sel, num_sel]
                if cat_type == "Bar":
                    fig_cb = px.bar(data, x=cat_sel, y=data.columns[1],
                        template="plotly_white", height=400,
                        color=data.columns[1],
                        color_continuous_scale=["#bfdbfe","#2563eb"],
                        text=data.columns[1])
                    fig_cb.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                elif cat_type == "Pie":
                    fig_cb = px.pie(data, names=cat_sel, values=data.columns[1],
                        template="plotly_white", height=400)
                else:
                    fig_cb = px.pie(data, names=cat_sel, values=data.columns[1],
                        hole=0.45, template="plotly_white", height=400)
                fig_cb.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc")
                st.plotly_chart(fig_cb, use_container_width=True)
            except Exception as e:
                st.error(f"Chart error: {e}")
    nav_buttons("Visualizations & Insights")

# ═══════════════════════════════════════════════
# PAGE — EXPORT
# ═══════════════════════════════════════════════
elif st.session_state.page == "Export":
    st.markdown("""
    <div class='main-header'>
        <h1>📦 Export</h1>
        <p>Download your cleaned and processed dataset</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = st.session_state.df
    st.markdown(f"""
    <div style='padding:20px;background:#eff6ff;border:1px solid #bfdbfe;
         border-radius:12px;margin-bottom:24px;'>
        <div style='font-size:0.8rem;color:#6b7280;margin-bottom:6px;'>READY TO EXPORT</div>
        <div style='font-family:"JetBrains Mono",monospace;font-size:1.4rem;
             color:#2563eb;font-weight:700;'>{len(df):,} rows × {len(df.columns)} columns</div>
        <div style='font-size:0.85rem;color:#6b7280;margin-top:4px;'>File: {st.session_state.file_name}</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["💾 Download","👁️ Preview"])

    with tab1:
        st.markdown("**📄 Processed / Encoded Dataset**")
        c1e, c2e = st.columns(2)
        with c1e:
            if st.session_state.get("processed_df") is not None:
                if st.download_button("⬇️ Download Encoded CSV",
                    data=export_csv(st.session_state.processed_df),
                    file_name=f"encoded_{st.session_state.file_name.rsplit('.',1)[0]}.csv",
                    mime="text/csv", key="export_encoded_csv"):
                    st.session_state.export_done = True
                    st.rerun()
        with c2e:
            if st.session_state.get("processed_df") is not None:
                try:
                    excel_data = export_excel(st.session_state.processed_df)
                    clicked_xlsx = st.download_button(
                        "⬇️ Download Encoded Excel",
                        data=excel_data,
                        file_name=f"encoded_{st.session_state.file_name.rsplit('.',1)[0]}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",key="export_encoded_xlsx")
                except Exception as e:
                    st.error(f"Excel export error: {e}")
                    clicked_xlsx = False
                    if clicked_xlsx:
                        st.session_state.export_done = True
                        st.rerun()

    with tab2:
        st.markdown("**Original / Cleaned Dataset (first 20 rows)**")
        st.dataframe(df.head(20), use_container_width=True, height=320)
        if st.session_state.get("processed_df") is not None:
            st.markdown("**Encoded / Processed Dataset (first 20 rows)**")
            st.dataframe(st.session_state.processed_df.head(20), use_container_width=True, height=320)

    
    nav_buttons("Export")
