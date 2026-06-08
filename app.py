"""
Smart Data Preprocessing & Data Quality Dashboard
Streamlit UI — app.py
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import io
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="DataPrep Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# INLINE BACKEND (mirrors notebook logic)
# ──────────────────────────────────────────────
import re
from difflib import SequenceMatcher
from scipy import stats
from scipy.stats import boxcox
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

DB_NAME = "dashboard.db"

def init_database():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS file_metadata
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, file_name TEXT,
                  upload_date TEXT, file_size_kb REAL, row_count INTEGER, column_count INTEGER)""")
    c.execute("""CREATE TABLE IF NOT EXISTS processing_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, file_name TEXT,
                  operation TEXT, details TEXT, timestamp TEXT)""")
    conn.commit(); conn.close()

init_database()

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

def load_file(buf, name):
    n = name.lower()
    if n.endswith(".csv"):      return pd.read_csv(buf, low_memory=False)
    elif n.endswith(".xlsx"):   return pd.read_excel(buf, engine="openpyxl")
    elif n.endswith(".xls"):    return pd.read_excel(buf)
    else: raise ValueError("Unsupported format")

def identify_column_types(df):
    ct = {"numerical": [], "categorical": [], "datetime": [], "boolean": [], "id": []}
    for col in df.columns:
        cl = col.lower(); s = df[col].dropna()
        if s.empty: ct["categorical"].append(col); continue
        if any(k in cl for k in ["_id","id","index","key","uuid","guid"]) and s.nunique()==len(s):
            ct["id"].append(col); continue
        if s.dtype==bool or set(map(str,s.unique())).issubset({"True","False","true","false","1","0","yes","no","Yes","No"}):
            ct["boolean"].append(col); continue
        if pd.api.types.is_datetime64_any_dtype(s): ct["datetime"].append(col); continue
        if s.dtype==object:
            try: pd.to_datetime(s.head(30),infer_datetime_format=True); ct["datetime"].append(col); continue
            except: pass
        if pd.api.types.is_numeric_dtype(s): ct["numerical"].append(col)
        else: ct["categorical"].append(col)
    return ct

def get_dataset_summary(df):
    return {
        "rows": len(df), "columns": len(df.columns),
        "missing_cells": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().sum().sum()/df.size*100,2),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum()/1024**2,3)
    }

def missing_value_report(df):
    ct = identify_column_types(df); report=[]
    for col in df.columns:
        m = df[col].isnull().sum()
        if m>0:
            if col in ct["numerical"]: dtype,strats="numerical",["mean","median","mode","drop"]
            elif col in ct["datetime"]: dtype,strats="datetime",["ffill","bfill","drop"]
            else: dtype,strats="categorical",["mode","ffill","bfill","custom","drop"]
            report.append({"column":col,"missing":m,"pct":round(m/len(df)*100,2),"type":dtype,"strategies":strats})
    return report

def fill_missing_values(df, column, strategy, custom_value=None):
    before=df[column].isnull().sum(); df=df.copy()
    if strategy=="mean":   df[column].fillna(df[column].mean(),inplace=True)
    elif strategy=="median": df[column].fillna(df[column].median(),inplace=True)
    elif strategy=="mode":
        m=df[column].mode()
        if not m.empty: df[column].fillna(m[0],inplace=True)
    elif strategy=="ffill": df[column].fillna(method="ffill",inplace=True)
    elif strategy=="bfill": df[column].fillna(method="bfill",inplace=True)
    elif strategy=="custom" and custom_value is not None: df[column].fillna(custom_value,inplace=True)
    elif strategy=="drop": df=df.dropna(subset=[column]).reset_index(drop=True)
    after=df[column].isnull().sum()
    return df,{"before":before,"after":after,"filled":before-after}

def detect_outliers_iqr(df):
    result={}
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1,Q3=df[col].quantile(0.25),df[col].quantile(0.75); IQR=Q3-Q1
        lo,hi=Q1-1.5*IQR, Q3+1.5*IQR
        out=df[(df[col]<lo)|(df[col]>hi)]
        result[col]={"count":len(out),"pct":round(len(out)/len(df)*100,2),"lower":lo,"upper":hi,"rows":out.index.tolist()}
    return result

def remove_outliers(df, column, method="iqr"):
    before=len(df)
    if method=="iqr":
        Q1,Q3=df[column].quantile(0.25),df[column].quantile(0.75); IQR=Q3-Q1
        df=df[(df[column]>=Q1-1.5*IQR)&(df[column]<=Q3+1.5*IQR)]
    df=df.reset_index(drop=True)
    return df,{"removed":before-len(df),"new_count":len(df)}

def cap_outliers(df, column):
    df=df.copy()
    Q1,Q3=df[column].quantile(0.25),df[column].quantile(0.75); IQR=Q3-Q1
    lo,hi=Q1-1.5*IQR,Q3+1.5*IQR
    n=((df[column]<lo)|(df[column]>hi)).sum()
    df[column]=df[column].clip(lower=lo,upper=hi)
    return df,{"capped":int(n)}

def calculate_skewness(df):
    rows=[]
    for col in df.select_dtypes(include=[np.number]).columns:
        sk=df[col].skew()
        cls="Normal" if abs(sk)<0.5 else "Moderately Skewed" if abs(sk)<1 else "Highly Skewed"
        rows.append({"Column":col,"Skewness":round(sk,4),"Classification":cls})
    return pd.DataFrame(rows)

def descriptive_statistics(df):
    rows=[]
    for col in df.select_dtypes(include=[np.number]).columns:
        s=df[col].dropna()
        mode_v=s.mode().iloc[0] if not s.mode().empty else np.nan
        rows.append({"Column":col,"Count":len(s),"Mean":round(s.mean(),4),"Median":round(s.median(),4),
                     "Mode":round(mode_v,4),"Std":round(s.std(),4),"Variance":round(s.var(),4),
                     "Min":round(s.min(),4),"Max":round(s.max(),4),
                     "Q1":round(s.quantile(0.25),4),"Q3":round(s.quantile(0.75),4),
                     "Skewness":round(s.skew(),4),"Kurtosis":round(s.kurtosis(),4)})
    return pd.DataFrame(rows)

def detect_invalid_values(df):
    issues={}
    for col in df.select_dtypes(include=[np.number]).columns:
        cl=col.lower()
        if any(k in cl for k in ["age"]):
            b=df[(df[col]<0)|(df[col]>150)]
            if len(b): issues[col]={"issue":"Age out of range (0–150)","count":len(b)}
        elif any(k in cl for k in ["pct","percent","rate"]):
            b=df[(df[col]<0)|(df[col]>100)]
            if len(b): issues[col]={"issue":"Percentage out of range","count":len(b)}
        elif any(k in cl for k in ["salary","income","revenue","price","cost","amount"]):
            b=df[df[col]<0]
            if len(b): issues[col]={"issue":"Negative monetary value","count":len(b)}
    return issues

def detect_negative_values(df):
    r={}
    for col in df.select_dtypes(include=[np.number]).columns:
        n=df[df[col]<0]
        if len(n): r[col]={"count":len(n)}
    return r

def detect_invalid_email(df):
    pat=re.compile(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$"); r={}
    for col in df.select_dtypes(include="object").columns:
        if any(k in col.lower() for k in ["email","mail","e-mail"]):
            bad=df[col].dropna().apply(lambda x: not bool(pat.match(str(x))))
            if bad.sum(): r[col]={"count":int(bad.sum())}
    return r

def detect_invalid_phone(df):
    pat=re.compile(r"^[\+]?[\d\s\-\(\)]{7,15}$"); r={}
    for col in df.select_dtypes(include="object").columns:
        if any(k in col.lower() for k in ["phone","mobile","tel","contact"]):
            bad=df[col].dropna().apply(lambda x: not bool(pat.match(str(x))))
            if bad.sum(): r[col]={"count":int(bad.sum())}
    return r

def detect_future_dates(df):
    now=pd.Timestamp.now(); r={}
    for col in df.columns:
        if any(k in col.lower() for k in ["birth","dob","born","date","created","joined"]):
            try:
                parsed=pd.to_datetime(df[col],errors="coerce")
                n=(parsed>now).sum()
                if n: r[col]={"count":int(n)}
            except: pass
    return r

def detect_duplicate_information_columns(df):
    suggestions=[]; cols=df.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1,len(cols)):
            ratio=SequenceMatcher(None,cols[i].lower(),cols[j].lower()).ratio()
            if ratio>0.7:
                suggestions.append({"col1":cols[i],"col2":cols[j],"reason":f"Name similarity: {ratio:.2f}"})
    num_df=df.select_dtypes(include=[np.number])
    if num_df.shape[1]>1:
        corr=num_df.corr().abs()
        for i in range(len(corr.columns)):
            for j in range(i+1,len(corr.columns)):
                v=corr.iloc[i,j]
                if v>0.98:
                    suggestions.append({"col1":corr.columns[i],"col2":corr.columns[j],"reason":f"Near-perfect correlation: {v:.3f}"})
    return suggestions

def recommend_encoding(df, col):
    n=df[col].nunique()
    if n==2: return "label"
    elif n<=10: return "onehot"
    else: return "frequency"

def apply_encoding(df, col, enc_type, ordinal_order=None):
    df=df.copy(); mapping=None
    if enc_type=="label":
        le=LabelEncoder(); df[col+"_encoded"]=le.fit_transform(df[col].astype(str))
        mapping=pd.DataFrame({"Original":le.classes_,"Encoded":range(len(le.classes_))})
    elif enc_type=="onehot":
        dummies=pd.get_dummies(df[col],prefix=col); df=pd.concat([df,dummies],axis=1)
        mapping=pd.DataFrame({"Original":dummies.columns,"Encoded":dummies.columns})
    elif enc_type=="ordinal" and ordinal_order:
        om={v:i for i,v in enumerate(ordinal_order)}
        df[col+"_ordinal"]=df[col].map(om)
        mapping=pd.DataFrame({"Original":ordinal_order,"Encoded":range(len(ordinal_order))})
    elif enc_type=="frequency":
        freq=df[col].value_counts(normalize=True)
        df[col+"_freq"]=df[col].map(freq)
        mapping=freq.reset_index(); mapping.columns=["Original","Frequency"]
    return df, mapping

def calculate_data_quality_score(df):
    miss_pct=df.isnull().sum().sum()/df.size*100
    dup_pct=df.duplicated().sum()/len(df)*100
    outliers=detect_outliers_iqr(df)
    avg_out=np.mean([v["pct"] for v in outliers.values()]) if outliers else 0
    inv=detect_invalid_values(df)
    inv_pct=sum(v["count"] for v in inv.values())/len(df)*100 if inv else 0
    score=max(0,30-miss_pct*0.6)+max(0,20-dup_pct*0.4)+max(0,20-avg_out*0.4)+max(0,15-inv_pct*0.3)+15
    return round(min(100,score),1)

def export_csv(df):
    return df.to_csv(index=False).encode("utf-8")

def export_excel(df):
    out=io.BytesIO()
    with pd.ExcelWriter(out,engine="openpyxl") as w: df.to_excel(w,index=False,sheet_name="Processed")
    return out.getvalue()

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface2: #18181f;
    --border: #2a2a38;
    --accent: #6366f1;
    --accent2: #818cf8;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
    --text: #e2e2f0;
    --muted: #6b6b80;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background: var(--bg) !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* Main header */
.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(99,102,241,0.08) 0%, transparent 60%);
}
.main-header h1 {
    font-family: 'Space Mono', monospace !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #fff !important;
    margin: 0 !important;
    letter-spacing: -1px;
}
.main-header p { color: var(--muted) !important; margin: 6px 0 0 0 !important; }

/* Metric cards */
.metric-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-card .val {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent2);
    display: block;
}
.metric-card .label { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }

/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 20px;
    background: var(--surface2);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    margin: 20px 0 16px 0;
}
.section-header h3 { margin: 0 !important; font-size: 1rem; font-weight: 600; color: var(--text); }

/* Status badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
}
.badge-success { background: rgba(34,197,94,0.15); color: var(--success); border: 1px solid rgba(34,197,94,0.3); }
.badge-warning { background: rgba(245,158,11,0.15); color: var(--warning); border: 1px solid rgba(245,158,11,0.3); }
.badge-danger  { background: rgba(239,68,68,0.15);  color: var(--danger);  border: 1px solid rgba(239,68,68,0.3); }
.badge-info    { background: rgba(99,102,241,0.15); color: var(--accent2); border: 1px solid rgba(99,102,241,0.3); }

/* Progress bar */
.progress-bar-wrap { background: var(--surface2); border-radius: 8px; height: 8px; overflow: hidden; margin: 4px 0; }
.progress-bar-fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); border-radius: 8px; transition: width 0.5s ease; }

/* Tables */
.stDataFrame { border-radius: 10px !important; overflow: hidden; }
[data-testid="stDataFrame"] div { background: var(--surface2) !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Select boxes, inputs */
.stSelectbox > div, .stMultiSelect > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: var(--surface2) !important; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: var(--muted) !important; border-radius: 6px !important; }
.stTabs [aria-selected="true"] { background: var(--accent) !important; color: white !important; }

/* Expander */
.streamlit-expanderHeader { background: var(--surface2) !important; border-radius: 8px !important; }

/* Score chip */
.score-chip {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 16px; border-radius: 20px; font-family: 'Space Mono', monospace;
    font-weight: 700; font-size: 0.9rem;
}

/* Upload area */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    background: var(--surface2) !important;
}

div.stAlert { border-radius: 10px !important; }

/* Nav pills in sidebar */
.nav-pill {
    display: block; padding: 10px 16px; margin: 4px 0;
    border-radius: 8px; cursor: pointer; font-weight: 500;
    transition: background 0.15s;
    color: var(--muted);
    text-decoration: none;
}
.nav-pill.active { background: rgba(99,102,241,0.2); color: var(--accent2); border-left: 2px solid var(--accent); }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────
defaults = {
    "df": None, "original_df": None, "file_name": "",
    "page": "Upload & Inspect", "quality_score": 0
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 24px 0;'>
        <div style='font-family:"Space Mono",monospace; font-size:1.3rem; font-weight:700; color:#818cf8;'>🧬 DataPrep Pro</div>
        <div style='font-size:0.75rem; color:#6b6b80; margin-top:4px;'>Smart Preprocessing Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    pages = ["📁 Upload & Inspect", "🧹 Cleaning & Validation", "🔢 Encoding & Outliers", "📊 Statistics & Export"]
    page_map = {p: p.split(" ", 1)[1] for p in pages}

    selected = st.radio("Navigation", pages, label_visibility="collapsed",
                        index=["Upload & Inspect","Cleaning & Validation","Encoding & Outliers","Statistics & Export"].index(st.session_state.page)
                        if st.session_state.page in ["Upload & Inspect","Cleaning & Validation","Encoding & Outliers","Statistics & Export"] else 0)
    st.session_state.page = page_map[selected]

    st.markdown("---")

    # Live stats
    if st.session_state.df is not None:
        df = st.session_state.df
        qs = calculate_data_quality_score(df)
        st.session_state.quality_score = qs
        color = "#22c55e" if qs >= 80 else "#f59e0b" if qs >= 60 else "#ef4444"
        st.markdown(f"""
        <div style='padding:16px; background:#111118; border:1px solid #2a2a38; border-radius:12px;'>
            <div style='font-size:0.7rem; color:#6b6b80; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;'>Live Dataset Stats</div>
            <div style='display:flex; justify-content:space-between; margin:6px 0;'>
                <span style='color:#6b6b80; font-size:0.85rem;'>Rows</span>
                <span style='font-family:"Space Mono",monospace; color:#e2e2f0; font-size:0.85rem;'>{len(df):,}</span>
            </div>
            <div style='display:flex; justify-content:space-between; margin:6px 0;'>
                <span style='color:#6b6b80; font-size:0.85rem;'>Columns</span>
                <span style='font-family:"Space Mono",monospace; color:#e2e2f0; font-size:0.85rem;'>{len(df.columns)}</span>
            </div>
            <div style='display:flex; justify-content:space-between; margin:6px 0;'>
                <span style='color:#6b6b80; font-size:0.85rem;'>Missing</span>
                <span style='font-family:"Space Mono",monospace; color:#f59e0b; font-size:0.85rem;'>{df.isnull().sum().sum():,}</span>
            </div>
            <div style='display:flex; justify-content:space-between; margin:6px 0;'>
                <span style='color:#6b6b80; font-size:0.85rem;'>Duplicates</span>
                <span style='font-family:"Space Mono",monospace; color:#ef4444; font-size:0.85rem;'>{df.duplicated().sum()}</span>
            </div>
            <div style='margin-top:14px;'>
                <div style='font-size:0.7rem; color:#6b6b80; margin-bottom:6px;'>Quality Score</div>
                <div style='font-family:"Space Mono",monospace; font-size:1.6rem; font-weight:700; color:{color};'>{qs}<span style='font-size:0.9rem;'>/100</span></div>
                <div style='background:#1a1a28; border-radius:6px; height:6px; margin-top:6px;'>
                    <div style='width:{qs}%; height:100%; background:{color}; border-radius:6px;'></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='padding:16px; background:#111118; border:1px dashed #2a2a38; border-radius:12px; text-align:center; color:#6b6b80; font-size:0.85rem;'>
            No dataset loaded.<br>Upload a file to begin.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='margin-top:20px; padding:10px; font-size:0.72rem; color:#444; text-align:center;'>
        File: <b style='color:#6b6b80;'>{st.session_state.file_name or "None"}</b>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# PAGE 1: UPLOAD & INSPECT
# ──────────────────────────────────────────────
if st.session_state.page == "Upload & Inspect":

    st.markdown("""
    <div class='main-header'>
        <h1>📁 Upload & Inspect</h1>
        <p>Upload your dataset to begin intelligent preprocessing analysis</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Choose a file", type=["csv","xlsx","xls"],
                                 help="Supports CSV, Excel (.xlsx/.xls)")

    if uploaded:
        try:
            df = load_file(uploaded, uploaded.name)
            st.session_state.df = df
            st.session_state.original_df = df.copy()
            st.session_state.file_name = uploaded.name
            size_kb = uploaded.size / 1024
            conn = sqlite3.connect(DB_NAME)
            conn.execute("INSERT INTO file_metadata VALUES (NULL,?,?,?,?,?)",
                         (uploaded.name, datetime.now().isoformat(), round(size_kb,2), len(df), len(df.columns)))
            conn.commit(); conn.close()
            st.success(f"✅ Loaded **{uploaded.name}** — {len(df):,} rows × {len(df.columns)} columns")
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

    if st.session_state.df is not None:
        df = st.session_state.df
        summary = get_dataset_summary(df)
        ct = identify_column_types(df)

        # Quick metrics
        c1,c2,c3,c4,c5 = st.columns(5)
        for col_widget, label, val in zip(
            [c1,c2,c3,c4,c5],
            ["Rows","Columns","Missing %","Duplicates","Memory MB"],
            [f"{summary['rows']:,}", str(summary['columns']),
             f"{summary['missing_pct']}%", str(summary['duplicate_rows']),
             f"{summary['memory_mb']}"]
        ):
            with col_widget:
                st.markdown(f"""<div class='metric-card'><span class='val'>{val}</span><span class='label'>{label}</span></div>""", unsafe_allow_html=True)

        st.markdown("&nbsp;")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Preview","📋 Schema","🔲 Column Types","⚠️ Missing","➕ Add Row"])

        with tab1:
            st.dataframe(df.head(50), use_container_width=True, height=380)

        with tab2:
            schema = pd.DataFrame({
                "Column": df.columns,
                "DType": df.dtypes.astype(str).values,
                "Non-Null": df.count().values,
                "Null": df.isnull().sum().values,
                "Unique": [df[c].nunique() for c in df.columns],
                "Sample": [str(df[c].dropna().iloc[0]) if not df[c].dropna().empty else "—" for c in df.columns]
            })
            st.dataframe(schema, use_container_width=True, height=400)

        with tab3:
            cc1, cc2 = st.columns(2)
            with cc1:
                for typ, emoji in [("numerical","🔢"),("categorical","🏷️")]:
                    st.markdown(f"**{emoji} {typ.title()}**")
                    if ct[typ]: st.write(", ".join(ct[typ]))
                    else: st.write("_None detected_")
            with cc2:
                for typ, emoji in [("datetime","📅"),("boolean","☑️"),("id","🔑")]:
                    st.markdown(f"**{emoji} {typ.title()}**")
                    if ct[typ]: st.write(", ".join(ct[typ]))
                    else: st.write("_None detected_")

        with tab4:
            miss_report = missing_value_report(df)
            if miss_report:
                miss_df = pd.DataFrame([{"Column":r["column"],"Missing":r["missing"],"% Missing":r["pct"],"Type":r["type"]} for r in miss_report])
                st.dataframe(miss_df, use_container_width=True)
                fig = go.Figure(go.Bar(
                    x=[r["column"] for r in miss_report],
                    y=[r["pct"] for r in miss_report],
                    marker_color="#6366f1", text=[f"{r['pct']}%" for r in miss_report], textposition="outside"
                ))
                fig.update_layout(title="Missing Value % by Column", template="plotly_dark", height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("<span class='badge badge-success'>✓ No missing values detected</span>", unsafe_allow_html=True)

        with tab5:
            st.markdown("**Manually add a new row to the dataset:**")
            input_data = {}
            form_cols = st.columns(min(3, len(df.columns)))
            for i, col in enumerate(df.columns):
                with form_cols[i % 3]:
                    dtype = str(df[col].dtype)
                    if "int" in dtype or "float" in dtype:
                        input_data[col] = st.number_input(col, value=0.0, key=f"inp_{col}")
                    elif col in ct["boolean"]:
                        input_data[col] = st.selectbox(col, [True, False], key=f"inp_{col}")
                    elif col in ct["categorical"] and df[col].nunique() < 50:
                        opts = list(df[col].dropna().unique())
                        input_data[col] = st.selectbox(col, opts, key=f"inp_{col}")
                    else:
                        input_data[col] = st.text_input(col, key=f"inp_{col}")

            if st.button("➕ Add Row"):
                try:
                    new_row = {col: input_data.get(col, np.nan) for col in df.columns}
                    st.session_state.df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    save_operation(st.session_state.file_name, "Add Row", new_row)
                    st.success("Row added successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

# ──────────────────────────────────────────────
# PAGE 2: CLEANING & VALIDATION
# ──────────────────────────────────────────────
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
    tab1, tab2, tab3, tab4 = st.tabs(["🔁 Duplicates", "🕳️ Missing Values", "🚨 Validation", "🔗 Similar Columns"])

    # ── Duplicates ──
    with tab1:
        st.markdown("<div class='section-header'><h3>Duplicate Row Detection</h3></div>", unsafe_allow_html=True)
        n_dupes = df.duplicated().sum()
        if n_dupes > 0:
            st.markdown(f"Found <span class='badge badge-danger'>{n_dupes} duplicates</span> ({round(n_dupes/len(df)*100,2)}% of rows)", unsafe_allow_html=True)
            with st.expander("Show duplicate rows"):
                st.dataframe(df[df.duplicated(keep="first")], use_container_width=True)
            if st.button("🗑️ Remove All Duplicates"):
                before = len(df)
                st.session_state.df = df.drop_duplicates(keep="first").reset_index(drop=True)
                removed = before - len(st.session_state.df)
                save_operation(st.session_state.file_name, "Remove Duplicates", f"Removed {removed} rows")
                st.success(f"Removed {removed} duplicate rows.")
                st.rerun()
        else:
            st.markdown("<span class='badge badge-success'>✓ No duplicates found</span>", unsafe_allow_html=True)

    # ── Missing Values ──
    with tab2:
        st.markdown("<div class='section-header'><h3>Missing Value Treatment</h3></div>", unsafe_allow_html=True)
        report = missing_value_report(df)
        if not report:
            st.markdown("<span class='badge badge-success'>✓ No missing values</span>", unsafe_allow_html=True)
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
                            save_operation(st.session_state.file_name, f"Fill Missing: {item['column']}", f"{chosen} — filled {stats_r['filled']}")
                            st.success(f"Filled {stats_r['filled']} values using '{chosen}'.")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

    # ── Validation ──
    with tab3:
        st.markdown("<div class='section-header'><h3>Data Validation & Anomaly Detection</h3></div>", unsafe_allow_html=True)

        v1, v2 = st.columns(2)
        with v1:
            st.markdown("**🔴 Invalid Domain Values**")
            inv = detect_invalid_values(df)
            if inv:
                for col, info in inv.items():
                    st.markdown(f"<span class='badge badge-danger'>{col}</span> {info['issue']} — {info['count']} rows", unsafe_allow_html=True)
            else:
                st.markdown("<span class='badge badge-success'>✓ None detected</span>", unsafe_allow_html=True)

            st.markdown("&nbsp;")
            st.markdown("**📧 Invalid Emails**")
            emails = detect_invalid_email(df)
            if emails:
                for col, info in emails.items():
                    st.markdown(f"<span class='badge badge-warning'>{col}</span> {info['count']} invalid emails", unsafe_allow_html=True)
            else:
                st.markdown("<span class='badge badge-success'>✓ None detected</span>", unsafe_allow_html=True)

        with v2:
            st.markdown("**➖ Negative Values**")
            negs = detect_negative_values(df)
            if negs:
                for col, info in negs.items():
                    st.markdown(f"<span class='badge badge-warning'>{col}</span> {info['count']} negative rows", unsafe_allow_html=True)
            else:
                st.markdown("<span class='badge badge-success'>✓ None detected</span>", unsafe_allow_html=True)

            st.markdown("&nbsp;")
            st.markdown("**📱 Invalid Phone Numbers**")
            phones = detect_invalid_phone(df)
            if phones:
                for col, info in phones.items():
                    st.markdown(f"<span class='badge badge-warning'>{col}</span> {info['count']} invalid", unsafe_allow_html=True)
            else:
                st.markdown("<span class='badge badge-success'>✓ None detected</span>", unsafe_allow_html=True)

        st.markdown("&nbsp;")
        st.markdown("**📅 Future Dates**")
        future = detect_future_dates(df)
        if future:
            for col, info in future.items():
                st.markdown(f"<span class='badge badge-danger'>{col}</span> {info['count']} future dates", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge badge-success'>✓ No future dates detected</span>", unsafe_allow_html=True)

    # ── Similar Columns ──
    with tab4:
        st.markdown("<div class='section-header'><h3>Similar / Redundant Column Detection</h3></div>", unsafe_allow_html=True)
        suggestions = detect_duplicate_information_columns(df)
        if suggestions:
            st.warning(f"Found {len(suggestions)} potential redundant column pairs. Review and decide which to drop.")
            for s in suggestions:
                c1c, c2c, c3c = st.columns([2,2,1])
                with c1c: st.markdown(f"<span class='badge badge-info'>{s['col1']}</span>", unsafe_allow_html=True)
                with c2c: st.markdown(f"<span class='badge badge-info'>{s['col2']}</span> — {s['reason']}", unsafe_allow_html=True)
                with c3c:
                    if st.button(f"Drop {s['col2']}", key=f"drop_{s['col1']}_{s['col2']}"):
                        if s["col2"] in st.session_state.df.columns:
                            st.session_state.df = st.session_state.df.drop(columns=[s["col2"]])
                            save_operation(st.session_state.file_name, "Drop Column", s["col2"])
                            st.success(f"Dropped column: {s['col2']}")
                            st.rerun()
        else:
            st.markdown("<span class='badge badge-success'>✓ No redundant columns detected</span>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# PAGE 3: ENCODING & OUTLIERS
# ──────────────────────────────────────────────
elif st.session_state.page == "Encoding & Outliers":

    st.markdown("""
    <div class='main-header'>
        <h1>🔢 Encoding & Outliers</h1>
        <p>Encode categorical features, detect outliers, and analyze distributions</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = st.session_state.df
    ct = identify_column_types(df)
    tab1, tab2, tab3, tab4 = st.tabs(["🏷️ Encoding", "📦 Outliers", "📐 Skewness", "📈 Distributions"])

    # ── Encoding ──
    with tab1:
        st.markdown("<div class='section-header'><h3>Categorical Encoding</h3></div>", unsafe_allow_html=True)
        enc_candidates = ct["categorical"] + ct["boolean"]
        if not enc_candidates:
            st.info("No categorical columns found.")
        else:
            for col in enc_candidates:
                n_unique = df[col].nunique()
                rec = recommend_encoding(df, col)
                with st.expander(f"**{col}** — {n_unique} unique values · Recommended: `{rec}`"):
                    left, right = st.columns([1,1])
                    with left:
                        st.markdown("**Before Encoding:**")
                        st.dataframe(df[col].value_counts().reset_index().head(10), use_container_width=True)
                    chosen_enc = st.selectbox("Encoding method", ["label","onehot","ordinal","frequency"],
                                              index=["label","onehot","ordinal","frequency"].index(rec),
                                              key=f"enc_{col}")
                    ordinal_order = None
                    if chosen_enc == "ordinal":
                        ordinal_order_str = st.text_input("Ordinal order (comma-separated, low→high)", key=f"ord_{col}")
                        if ordinal_order_str:
                            ordinal_order = [x.strip() for x in ordinal_order_str.split(",")]
                    if st.button(f"Apply Encoding to {col}", key=f"apply_enc_{col}"):
                        try:
                            new_df, mapping = apply_encoding(df, col, chosen_enc, ordinal_order)
                            st.session_state.df = new_df
                            df = new_df
                            save_operation(st.session_state.file_name, f"Encoding: {col}", chosen_enc)
                            with right:
                                st.markdown("**Mapping Table:**")
                                if mapping is not None:
                                    st.dataframe(mapping.head(20), use_container_width=True)
                            st.success(f"Applied {chosen_enc} encoding to '{col}'.")
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

    # ── Outliers ──
    with tab2:
        st.markdown("<div class='section-header'><h3>Outlier Detection & Treatment</h3></div>", unsafe_allow_html=True)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.info("No numerical columns found.")
        else:
            outliers_iqr = detect_outliers_iqr(df)
            method_choice = st.radio("Detection method", ["IQR","Z-Score"], horizontal=True)

            if method_choice == "Z-Score":
                from scipy.stats import zscore as zsc
                outliers_display = {}
                for col in num_cols:
                    z = np.abs(zsc(df[col].dropna()))
                    n = (z > 3).sum()
                    outliers_display[col] = {"count": int(n), "pct": round(int(n)/len(df)*100,2)}
            else:
                outliers_display = {k: {"count":v["count"],"pct":v["pct"]} for k,v in outliers_iqr.items()}

            # Summary table
            out_table = pd.DataFrame([{"Column":k,"Outliers":v["count"],"Outlier %":v["pct"]} for k,v in outliers_display.items()])
            st.dataframe(out_table, use_container_width=True)

            # Per-column treatment + boxplot
            selected_col = st.selectbox("Select column to treat", num_cols)
            if selected_col:
                info = outliers_iqr.get(selected_col, {})

                fig = go.Figure()
                fig.add_trace(go.Box(y=df[selected_col].dropna(), name=selected_col,
                                     boxmean=True, marker_color="#6366f1", line_color="#4f46e5"))
                fig.update_layout(title=f"Boxplot: {selected_col}", template="plotly_dark", height=350)
                st.plotly_chart(fig, use_container_width=True)

                ca, cb, cc = st.columns(3)
                with ca:
                    if st.button("🗑️ Remove Outliers"):
                        try:
                            new_df, r = remove_outliers(df, selected_col, method=method_choice.lower().replace("-",""))
                            st.session_state.df = new_df
                            save_operation(st.session_state.file_name, f"Remove Outliers: {selected_col}", r)
                            st.success(f"Removed {r['removed']} outlier rows."); st.rerun()
                        except Exception as e: st.error(str(e))
                with cb:
                    if st.button("📎 Cap Outliers"):
                        try:
                            new_df, r = cap_outliers(df, selected_col)
                            st.session_state.df = new_df
                            save_operation(st.session_state.file_name, f"Cap Outliers: {selected_col}", r)
                            st.success(f"Capped {r['capped']} outliers."); st.rerun()
                        except Exception as e: st.error(str(e))
                with cc:
                    st.info("Keep: no action taken.")

    # ── Skewness ──
    with tab3:
        st.markdown("<div class='section-header'><h3>Skewness Analysis</h3></div>", unsafe_allow_html=True)
        skew_df = calculate_skewness(df)
        if skew_df.empty:
            st.info("No numerical columns.")
        else:
            color_map = {"Normal":"#22c55e","Moderately Skewed":"#f59e0b","Highly Skewed":"#ef4444"}
            skew_df["color"] = skew_df["Classification"].map(color_map)
            fig = go.Figure(go.Bar(
                x=skew_df["Column"], y=skew_df["Skewness"],
                marker_color=skew_df["color"],
                text=skew_df["Classification"], textposition="outside"
            ))
            fig.update_layout(title="Skewness by Column", template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(skew_df[["Column","Skewness","Classification"]], use_container_width=True)

            st.markdown("**Apply Transformation:**")
            skew_col = st.selectbox("Column", df.select_dtypes(include=[np.number]).columns)
            transform = st.radio("Transformation", ["Log","Sqrt","Box-Cox"], horizontal=True)
            if st.button("Apply Transform"):
                try:
                    if transform == "Log":
                        shift = abs(df[skew_col].min())+1 if df[skew_col].min()<=0 else 0
                        st.session_state.df[skew_col+"_log"] = np.log(df[skew_col]+shift)
                    elif transform == "Sqrt":
                        shift = abs(df[skew_col].min()) if df[skew_col].min()<0 else 0
                        st.session_state.df[skew_col+"_sqrt"] = np.sqrt(df[skew_col]+shift)
                    elif transform == "Box-Cox":
                        s = df[skew_col].dropna()
                        shift = abs(s.min())+1 if s.min()<=0 else 0
                        t,_ = boxcox(s+shift)
                        st.session_state.df.loc[s.index, skew_col+"_boxcox"] = t
                    save_operation(st.session_state.file_name, f"{transform} Transform: {skew_col}", "applied")
                    st.success(f"{transform} transform applied. New column added."); st.rerun()
                except Exception as e: st.error(str(e))

    # ── Distributions ──
    with tab4:
        st.markdown("<div class='section-header'><h3>Distribution Analysis</h3></div>", unsafe_allow_html=True)
        num_cols2 = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols2:
            st.info("No numerical columns.")
        else:
            dist_col = st.selectbox("Select column", num_cols2, key="dist_col")
            data = df[dist_col].dropna()
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=data, name="Histogram", nbinsx=30,
                                       marker_color="rgba(99,102,241,0.6)", opacity=0.8))
            try:
                kde = stats.gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 200)
                kde_y = kde(x_range) * len(data) * (data.max()-data.min()) / 30
                fig.add_trace(go.Scatter(x=x_range, y=kde_y, mode="lines", name="KDE",
                                         line=dict(color="#f59e0b", width=2)))
            except: pass
            fig.update_layout(title=f"Distribution: {dist_col}", template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────
# PAGE 4: STATISTICS & EXPORT
# ──────────────────────────────────────────────
elif st.session_state.page == "Statistics & Export":

    st.markdown("""
    <div class='main-header'>
        <h1>📊 Statistics & Export</h1>
        <p>Explore statistical summaries, correlations, quality scores, and export your data</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = st.session_state.df
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📐 Statistics","🔗 Correlation","🏆 Quality Score","🕐 History","💾 Export"])

    # ── Statistics ──
    with tab1:
        st.markdown("<div class='section-header'><h3>Descriptive Statistics</h3></div>", unsafe_allow_html=True)
        stats_df = descriptive_statistics(df)
        if stats_df.empty:
            st.info("No numerical columns to summarize.")
        else:
            st.dataframe(stats_df, use_container_width=True, height=400)

        st.markdown("<div class='section-header'><h3>Categorical Summary</h3></div>", unsafe_allow_html=True)
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        if cat_cols:
            cat_col = st.selectbox("Select categorical column", cat_cols)
            vc = df[cat_col].value_counts().head(20)
            fig = go.Figure(go.Bar(x=vc.index.astype(str), y=vc.values,
                                   marker_color="#6366f1", text=vc.values, textposition="outside"))
            fig.update_layout(title=f"Value Counts: {cat_col}", template="plotly_dark", height=350)
            st.plotly_chart(fig, use_container_width=True)

    # ── Correlation ──
    with tab2:
        st.markdown("<div class='section-header'><h3>Correlation Analysis (No Heatmap)</h3></div>", unsafe_allow_html=True)
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            st.info("Need at least 2 numerical columns for correlation.")
        else:
            method = st.radio("Correlation method", ["pearson","spearman"], horizontal=True)
            corr_matrix = num_df.corr(method=method)
            focus_col = st.selectbox("Focus column", corr_matrix.columns)
            corr_vals = corr_matrix[focus_col].drop(focus_col).dropna().sort_values(key=abs, ascending=True)
            colors = ["#22c55e" if v>0 else "#ef4444" for v in corr_vals.values]
            fig = go.Figure(go.Bar(
                x=corr_vals.values, y=corr_vals.index, orientation="h",
                marker_color=colors, text=[f"{v:.3f}" for v in corr_vals.values], textposition="outside"
            ))
            fig.update_layout(
                title=f"{method.capitalize()} Correlation with '{focus_col}'",
                xaxis_title="Correlation Coefficient", template="plotly_dark",
                height=max(300, len(corr_vals)*40+100), xaxis=dict(range=[-1,1])
            )
            st.plotly_chart(fig, use_container_width=True)

            # All correlations table
            with st.expander("Full correlation matrix (table)"):
                st.dataframe(corr_matrix.round(3), use_container_width=True)

    # ── Quality Score ──
    with tab3:
        st.markdown("<div class='section-header'><h3>Data Quality Score</h3></div>", unsafe_allow_html=True)
        score = calculate_data_quality_score(df)
        color = "#22c55e" if score>=80 else "#f59e0b" if score>=60 else "#ef4444"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x":[0,1],"y":[0,1]},
            title={"text":"Data Quality Score","font":{"size":18}},
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":color},
                "steps":[
                    {"range":[0,40],"color":"#2d1414"},
                    {"range":[40,70],"color":"#2d2014"},
                    {"range":[70,100],"color":"#142d14"}
                ],
                "threshold":{"line":{"color":"white","width":3},"thickness":0.75,"value":score}
            }
        ))
        fig.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig, use_container_width=True)

        # Score breakdown
        miss_pct = df.isnull().sum().sum()/df.size*100
        dup_pct  = df.duplicated().sum()/len(df)*100
        out_info = detect_outliers_iqr(df)
        avg_out  = np.mean([v["pct"] for v in out_info.values()]) if out_info else 0
        inv      = detect_invalid_values(df)
        inv_pct  = sum(v["count"] for v in inv.values())/len(df)*100 if inv else 0

        breakdown = {
            "Missing Values (30 pts)": max(0,30-miss_pct*0.6),
            "Duplicates (20 pts)":     max(0,20-dup_pct*0.4),
            "Outliers (20 pts)":       max(0,20-avg_out*0.4),
            "Invalid Values (15 pts)": max(0,15-inv_pct*0.3),
            "Consistency (15 pts)":    15.0
        }
        bd_df = pd.DataFrame({"Factor":list(breakdown.keys()),"Score":[round(v,2) for v in breakdown.values()]})
        bd_df["Max"] = [30,20,20,15,15]
        bd_df["Pct"] = (bd_df["Score"]/bd_df["Max"]*100).round(1)

        for _, row in bd_df.iterrows():
            pct = row["Pct"]
            bar_color = "#22c55e" if pct>=80 else "#f59e0b" if pct>=60 else "#ef4444"
            st.markdown(f"""
            <div style='margin:8px 0;'>
                <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
                    <span style='font-size:0.85rem;color:#e2e2f0;'>{row['Factor']}</span>
                    <span style='font-family:"Space Mono",monospace;font-size:0.85rem;color:{bar_color};'>{row['Score']}/{row['Max']}</span>
                </div>
                <div class='progress-bar-wrap'>
                    <div class='progress-bar-fill' style='width:{pct}%;background:{bar_color};'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── History ──
    with tab4:
        st.markdown("<div class='section-header'><h3>Processing History</h3></div>", unsafe_allow_html=True)
        hist = get_processing_history(st.session_state.file_name)
        if hist.empty:
            st.info("No operations recorded yet.")
        else:
            st.dataframe(hist[["timestamp","operation","details"]].rename(columns={
                "timestamp":"Timestamp","operation":"Operation","details":"Details"
            }), use_container_width=True, height=400)

    # ── Export ──
    with tab5:
        st.markdown("<div class='section-header'><h3>Export Processed Dataset</h3></div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style='padding:16px;background:#111118;border:1px solid #2a2a38;border-radius:12px;margin-bottom:20px;'>
            <div style='font-size:0.8rem;color:#6b6b80;margin-bottom:8px;'>READY TO EXPORT</div>
            <div style='font-family:"Space Mono",monospace;font-size:1.4rem;color:#818cf8;'>{len(df):,} rows × {len(df.columns)} columns</div>
            <div style='font-size:0.85rem;color:#6b6b80;margin-top:4px;'>File: {st.session_state.file_name}</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📄 CSV Export**")
            csv_bytes = export_csv(df)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_bytes,
                file_name=f"processed_{st.session_state.file_name.rsplit('.',1)[0]}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            st.markdown("**📊 Excel Export**")
            try:
                excel_bytes = export_excel(df)
                st.download_button(
                    label="⬇️ Download Excel",
                    data=excel_bytes,
                    file_name=f"processed_{st.session_state.file_name.rsplit('.',1)[0]}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Excel export error: {e}")

        st.markdown("&nbsp;")
        with st.expander("Preview export (first 20 rows)"):
            st.dataframe(df.head(20), use_container_width=True)
