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
    page_order = ["Upload & Inspect", "Statistics & EDA", "Recommendations", "Cleaning", "Encoding & Outliers", "Visualizations", "Export"]
    idx = page_order.index(current_page)
    slug = current_page.replace(" ", "_").replace("&", "and")

    st.markdown("---")
    
    col_prev, col_space, col_next = st.columns([2, 6, 2])

    with col_prev:
        if idx > 0:
            if st.button(f"⬅️ {page_order[idx-1]}", key=f"nav_prev_{slug}", type="secondary"):
                st.session_state.page = page_order[idx-1]
                st.rerun()

    with col_next:
        if idx < len(page_order) - 1:
            if st.button(f"{page_order[idx+1]} ➡️", key=f"nav_next_{slug}", type="secondary"):
                st.session_state.page = page_order[idx+1]
                st.rerun()    



# ── Database ──────────────────────────────────
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
    conn.commit()
    conn.close()  

def save_operation(file_name, operation, details):
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.execute(
            "INSERT INTO processing_history (file_name, operation, details, timestamp) VALUES (?,?,?,?)",
            (file_name, operation, str(details), datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
    except:
        pass

def get_processing_history(file_name=None):
    try:
        conn = sqlite3.connect(DB_NAME)
        if file_name:
            df = pd.read_sql(
                "SELECT * FROM processing_history WHERE file_name=? ORDER BY timestamp DESC",
                conn, params=(file_name,)
            )
        else:
            df = pd.read_sql(
                "SELECT * FROM processing_history ORDER BY timestamp DESC", conn
            )
        conn.close()
        return df
    except:
        return pd.DataFrame()

init_database()
# ── File loading ──────────────────────────────
def load_file(buf, name):
    n = name.lower()

    if n.endswith(".csv"):
        for enc in ["utf-8", "latin1", "cp1252", "ISO-8859-1"]:
            try:
                buf.seek(0)
                return pd.read_csv(buf, encoding=enc, low_memory=False)
            except UnicodeDecodeError:
                continue

        raise ValueError("Unable to determine CSV encoding.")

    elif n.endswith(".xlsx"):
        return pd.read_excel(buf, engine="openpyxl")

    elif n.endswith(".xls"):
        return pd.read_excel(buf)

    else:
        raise ValueError("Unsupported format")

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
    before = df[column].isnull().sum()
    df = df.copy()
    
    if strategy == "mean":
        df[column] = df[column].fillna(df[column].mean())
    elif strategy == "median":
        df[column] = df[column].fillna(df[column].median())
    elif strategy == "mode":
        m = df[column].mode()
        if not m.empty:
            df[column] = df[column].fillna(m[0])
    elif strategy == "ffill":
        df[column] = df[column].ffill().bfill()   # ← bfill catches leading nulls
    elif strategy == "bfill":
        df[column] = df[column].bfill().ffill()   # ← ffill catches trailing nulls
    elif strategy == "custom" and custom_value is not None:
        try:
            # try to cast to column dtype
            typed_val = df[column].dtype.type(custom_value)
        except (ValueError, TypeError):
            typed_val = custom_value
        df[column] = df[column].fillna(typed_val)
    elif strategy == "drop":
        df = df[df[column].notna()].reset_index(drop=True)
    
    after = df[column].isnull().sum()
    return df, {"before": int(before), "after": int(after), "filled": int(before - after)}
    
def detect_outliers_iqr(df):
    result = {}
    if df is None or len(df) == 0:
        return result
    for col in df.select_dtypes(include=[np.number]).columns:
        try:
            s = df[col].dropna()
            if len(s) < 4:
                continue
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            out = df[(df[col] < lo) | (df[col] > hi)]
            result[col] = {
                "count": len(out),
                "pct": round(len(out)/len(df)*100, 2),
                "lower": lo, "upper": hi,
                "Q1": Q1, "Q3": Q3, "IQR": IQR,
                "mean": s.mean(), "std": s.std(),
                "rows": out.index.tolist()
            }
        except Exception:
            continue
    return result


def detect_outliers_zscore(df, threshold=3):
    result = {}
    if df is None or len(df) == 0:
        return result
    for col in df.select_dtypes(include=[np.number]).columns:
        try:
            s = df[col].dropna()
            if len(s) < 4:
                continue
            z = np.abs(stats.zscore(s))
            outlier_idx = s.index[z > threshold].tolist()
            result[col] = {
                "count": len(outlier_idx),
                "pct": round(len(outlier_idx)/len(df)*100, 2),
                "mean": round(s.mean(), 4),
                "std": round(s.std(), 4),
                "threshold": threshold,
                "rows": outlier_idx
            }
        except Exception:
            continue
    return result
def prepare_line_data(df, x_col, y_col, color_col=None, agg_fn="mean"):
    df = df.copy()
    try:
        df[x_col] = pd.to_datetime(df[x_col])
        is_date = True
    except:
        is_date = False
    df = df.dropna(subset=[x_col] + ([y_col] if y_col else []))
    if y_col:
        group_cols = [x_col] + ([color_col] if color_col else [])
        agg_map = {"mean":"mean","sum":"sum","count":"count","median":"median","max":"max","min":"min"}
        df = df.groupby(group_cols)[y_col].agg(agg_map.get(agg_fn,"mean")).reset_index()
    df = df.sort_values(x_col)
    return df, is_date    
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

def _is_phone_column(col_name):
    c1 = col_name.lower().replace(" ", "_").replace("-", "_")
 
    # Hard name keywords that are definitively NOT phone numbers
    NOT_PHONE = [
        "name", "person", "first", "last", "full", "display",
        "label", "title", "description", "address", "city", "country",
        "suit", "aviv", "type", "category", "status"
    ]
    if any(nk in cl for nk in NOT_PHONE):
        return False
 
    # Must contain one of these root keywords (as a word boundary, not substring)
    PHONE_ROOTS = ["phone", "mobile", "fax", "tel", "cellphone", "cell_phone",
                   "whatsapp", "landline", "pager"]
    for root in PHONE_ROOTS:
        # whole-word match: root must be preceded/followed by _ or start/end of string
        pattern = rf"(^|_){re.escape(root)}($|_)"
        if re.search(pattern, cl):
            return True
 
    # 'contact' alone → phone; 'contact_name', 'contactname' → NOT phone
    if re.search(r"(^|_)contact($|_)", cl):
        return True
 
    return False
 
 
def detect_invalid_phone(df):
    """Improved phone validator — handles dots, 8-digit, extensions, country codes."""
    PHONE_PAT = re.compile(
        r"^\+?[\d]{1,4}?"           # optional country code
        r"[\s.\-()]?"
        r"[\d]{2,4}"                # area code
        r"[\s.\-()]?"
        r"[\d]{2,4}"
        r"[\s.\-()]?"
        r"[\d]{0,4}"
        r"(\s*(x|ext|#)\s*\d{1,6})?$",  # optional extension
        re.IGNORECASE
    )
    r = {}
    for col in df.select_dtypes(include="object").columns:
        if not _is_phone_column(col):
            continue
        series = df[col].dropna().astype(str)
        def _bad(v):
            digits = re.sub(r"[^\d]", "", v)
            if len(digits) == 0:
                return True
            if len(digits) < 7 or len(digits) > 15:
                return True
            return False
        bad = series.apply(_bad)
        if bad.sum():
            r[col] = {"count": int(bad.sum())}
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
    if df is None or len(df) == 0 or df.size == 0:
        return 0.0
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
/* Nav buttons */
div[data-testid="column"]:first-child .stButton > button {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: var(--text2) !important;
    width: 100% !important;
    text-align: left !important;
    font-size: 0.82rem !important;
}
div[data-testid="column"]:last-child .stButton > button {
    background: var(--accent) !important;
    border: none !important;
    color: #fff !important;
    width: 100% !important;
    text-align: right !important;
    font-size: 0.82rem !important;
}
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

    pages = ["📁 Upload & Inspect","📈 Statistics & EDA","💡 Recommendations","🧹 Cleaning","🔠 Encoding & Outliers","📊 Visualizations","📦 Export"]
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
            if uploaded.name != st.session_state.get("file_name", ""):
                df = load_file(uploaded, uploaded.name)

                # Save everything to session state FIRST before any rerun
                st.session_state.original_df       = df.copy()
                st.session_state.processed_df      = df.copy()
                st.session_state.df                = df.copy()
                st.session_state.file_name         = uploaded.name
                st.session_state.encoded_columns   = []
                st.session_state.encoders          = {}
                st.session_state.target_col        = "— None —"
                st.session_state.target_encoded    = False
                st.session_state.transformed_columns = []
                st.session_state.export_done       = False
                st.session_state.cleaning_history  = []
                st.session_state.operations        = []

                # DB logging — safe to do before rerun
                try:
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
                except Exception:
                    pass

                st.success(
                    f"✅ Loaded **{uploaded.name}** — "
                    f"{len(df):,} rows × {len(df.columns)} columns"
                )
                # rerun LAST — after everything is saved
                st.rerun()

        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
    if st.session_state.df is not None:
        raw_df = st.session_state.original_df
        processed_df = st.session_state.processed_df
        # Use RAW data for UI
        df = raw_df        
        summary = get_dataset_summary(raw_df)
        ct = identify_column_types(raw_df)
        cols_m = st.columns(4)
        for col_w, label, val in zip(cols_m,
            ["Rows","Columns","Missing %","Duplicates"],
            [f"{summary['rows']:,}", str(summary['columns']),
             f"{summary['missing_pct']}%", str(summary['duplicate_rows'])]):
                 with col_w:
                    st.markdown(f"""<div class='metric-card'><span class='val'>{val}</span><span class='label'>{label}</span></div>""", unsafe_allow_html=True)
                    st.markdown("&nbsp;")
        tab1, tab2, tab3, tab4, tab5  = st.tabs(["👁️ Preview","📋 Schema","🏷️ Column Types","❓ Missing","🔄 Encoded data"])
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

        with tab5:
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
        st.markdown("<div class='section-header'><h3>Outlier Detection & Treatment</h3></div>", unsafe_allow_html=True)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.info("No numerical columns found.")
        else:
            method_choice = st.radio("Detection Method", ["IQR (Interquartile Range)","Z-Score"], horizontal=True)
            use_iqr = "IQR" in method_choice
    
            if use_iqr:
                outlier_data = detect_outliers_iqr(df)   # ← always reassign here
                stats_rows = [{"Column":c,"Q1":round(i["Q1"],3),"Q3":round(i["Q3"],3),
                                "IQR":round(i["IQR"],3),"Lower":round(i["lower"],3),
                                "Upper":round(i["upper"],3),"Outliers":i["count"],"Outlier %":i["pct"]}
                               for c,i in outlier_data.items()]
            else:
                outlier_data = detect_outliers_zscore(df)  # ← always reassign here
                stats_rows = [{"Column":c,"Mean":round(i["mean"],3),"Std":round(i["std"],3),
                                "Threshold":f"|Z|>{i['threshold']}","Outliers":i["count"],"Outlier %":i["pct"]}
                               for c,i in outlier_data.items()]
    
            if stats_rows:
                st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)
            else:
                st.markdown("<span class='badge badge-success'>✅ No outliers detected</span>", unsafe_allow_html=True)
    
            # Box plot
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
                    info = outlier_data.get(col, {})          # ← now always in scope
                    outlier_idx = set(info.get("rows", []))
                    series = df[col].dropna()
                    out_series = series.loc[series.index.intersection(list(outlier_idx))]
    
                    fig_box.add_trace(go.Box(
                        y=series,
                        name=col,
                        marker_color="rgba(37,99,235,0.55)",
                        line_color="#2563eb",
                        fillcolor="rgba(37,99,235,0.15)",
                        boxpoints=False,
                        showlegend=not legend_added,
                        legendgroup="normal",
                    ), row=1, col=i)
    
                    if len(out_series) > 0:
                        fig_box.add_trace(go.Scatter(
                            y=out_series.values,
                            x=[col] * len(out_series),
                            mode="markers",
                            marker=dict(color="rgba(220,38,38,0.85)", size=9, symbol="circle"),
                            name="Outlier" if not legend_added else "",
                            legendgroup="outlier",
                            showlegend=not legend_added,
                        ), row=1, col=i)
    
                    legend_added = True
    
                fig_box.update_layout(
                    title="Box Plots — Outliers (Red ●) vs Normal Range",
                    template="plotly_white",
                    height=max(420, 380),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#f8f9fc",
                    showlegend=True,
                )
                st.plotly_chart(fig_box, use_container_width=True)
            except Exception as e:
                st.error(f"Box plot error: {e}")
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
            st.markdown(
                "<div class='section-header'><h3>Correlation Heatmap</h3></div>",
                unsafe_allow_html=True
            )
            
            corr = num_df.corr().round(2)
            
            fig_corr = go.Figure(
                data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.columns,
                    zmin=-1,
                    zmax=1,
            
                    # Professional Red-White-Blue palette
                    colorscale=[
                        [0.0, "#b91c1c"],   # Strong negative
                        [0.25, "#ef4444"],
                        [0.50, "#ffffff"],  # Zero correlation
                        [0.75, "#3b82f6"],
                        [1.0, "#1e3a8a"]    # Strong positive
                    ],
            
                    text=corr.values,
                    texttemplate="%{text:.2f}",
                    textfont={"size": 11},
                    hovertemplate=
                    "<b>%{y}</b> vs <b>%{x}</b><br>" +
                    "Correlation: %{z:.3f}<extra></extra>",
            
                    xgap=2,
                    ygap=2,
            
                    colorbar=dict(
                        title="Correlation",
                        thickness=14,
                        len=0.8
                    )
                )
            )
            
            fig_corr.update_layout(
                height=max(500, len(corr.columns) * 45),
                template="plotly_white",
            
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
            
                margin=dict(l=80, r=30, t=50, b=80),
            
                title=dict(
                    text="Correlation Matrix",
                    x=0.5,
                    font=dict(size=18)
                ),
            
                xaxis=dict(
                    tickangle=-45,
                    tickfont=dict(size=11),
                    side="bottom"
                ),
            
                yaxis=dict(
                    tickfont=dict(size=11),
                    autorange="reversed"
                )
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
            # Strong correlations summary
            strong = []
            cols_list = corr.columns.tolist()
            for i in range(len(cols_list)):
                for j in range(i + 1, len(cols_list)):
                    v = corr.iloc[i, j]
                    if abs(v) >= 0.7:
                        strong.append({
                            "Column A": cols_list[i],
                            "Column B": cols_list[j],
                            "r": round(float(v), 4),
                            "Strength": "Strong positive" if v > 0 else "Strong negative"
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
# PAGE 4 — CLEANING 
# ═══════════════════════════════════════════════
elif st.session_state.page == "Cleaning":
    st.markdown("""
    <div class='main-header'>
        <h1>🧹 Cleaning </h1>
        <p>Remove duplicates, fix missing values, and detect data anomalies</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first on the Upload & Inspect page.")
        st.stop()

    df = st.session_state.df
    tab1, tab2, tab3 = st.tabs(["🗑️ Duplicates","🔧 Missing Values","🛠️ Feature Engineering"])

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
                            st.session_state.df = new_df          # ← must assign new_df, not df
                            st.session_state.original_df = new_df # ← add this line too
                            save_operation(st.session_state.file_name, f"Fill Missing: {item['column']}", f"{chosen} – filled {stats_r['filled']}")
                            st.success(...)
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
    with tab3:
        st.markdown("<div class='section-header'><h3>🛠️ Feature Engineering</h3></div>", unsafe_allow_html=True)
     
        # Initialise cache keys
        if "fe_created_cols" not in st.session_state:
            st.session_state.fe_created_cols = []
     
        df_fe = st.session_state.df
     
        # ── Previously created features strip ────────────────────────────────────
        existing_fe_cols = [c for c in st.session_state.fe_created_cols if c in df_fe.columns]
        if existing_fe_cols:
            fe_chips = "".join([
                f"<span style='display:inline-flex;background:#eff6ff;border:1px solid #bfdbfe;"
                f"color:#2563eb;border-radius:999px;padding:2px 10px;font-size:0.72rem;"
                f"font-weight:600;margin:2px;'>✦ {c}</span>"
                for c in existing_fe_cols
            ])
            st.markdown(f"""
            <div style='background:#f8faff;border:1px solid #e0eaff;border-radius:10px;
                 padding:12px 16px;margin-bottom:16px;'>
                <div style='font-size:0.72rem;color:#6b7280;text-transform:uppercase;
                     letter-spacing:1px;margin-bottom:8px;'>
                     Engineered columns in dataset ({len(existing_fe_cols)})</div>
                <div style='line-height:2.2;'>{fe_chips}</div>
            </div>
            """, unsafe_allow_html=True)
     
        # ═══════════════════════════════
        # SECTION 1 — Date Feature Extraction
        # ═══════════════════════════════
        st.markdown("**📅 Extract Features from Date Column**")
     
        def _try_parse_as_date(series):
            """
            Try multiple common date formats on a sample.
            Returns (parsed_series_or_None, success_rate).
            Mixed formats handled by iterating format list then generic fallback.
            """
            sample = series.dropna().head(30)
            if len(sample) == 0:
                return None, 0.0
            result = pd.Series([pd.NaT] * len(sample), index=sample.index)
            for fmt in [
                "%m/%d/%Y %H:%M", "%m/%d/%Y",
                "%m-%d-%Y %H:%M", "%m-%d-%Y",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
                "%d/%m/%Y %H:%M", "%d/%m/%Y",
                "%d-%m-%Y %H:%M", "%d-%m-%Y",
                "%Y/%m/%d", "%b %d %Y", "%B %d %Y",
            ]:
                still_null = result.isna() & sample.notna()
                if not still_null.any():
                    break
                try:
                    attempt = pd.to_datetime(sample[still_null], format=fmt, errors="coerce")
                    result[still_null] = attempt
                except Exception:
                    continue
            # final generic pass
            still_null = result.isna() & sample.notna()
            if still_null.any():
                result[still_null] = pd.to_datetime(sample[still_null], errors="coerce")
            success_rate = result.notna().sum() / max(len(sample), 1)
            return result if success_rate >= 0.6 else None, success_rate
     
        # Detect datetime-like columns — proper datetime OR string columns that parse
        datetime_candidates = []
        for col in df_fe.columns:
            if pd.api.types.is_datetime64_any_dtype(df_fe[col]):
                datetime_candidates.append(col)
            elif df_fe[col].dtype == object:
                parsed, rate = _try_parse_as_date(df_fe[col])
                if parsed is not None:
                    datetime_candidates.append(col)
     
        if not datetime_candidates:
            st.info("No date columns detected. If your date column is stored as text, convert it first using Data Type Conversion above.")
        else:
            fe_date_col = st.selectbox(
                "Date column",
                datetime_candidates,
                key="fe_date_col"
            )
     
            # Show sample values with parsed preview
            raw_samples = df_fe[fe_date_col].dropna().head(4).tolist()
            try:
                parsed_samples = pd.to_datetime(
                    df_fe[fe_date_col].dropna().head(4), errors="coerce"
                ).tolist()
                preview_pairs = [
                    f"`{r}` → `{p.strftime('%Y-%m-%d') if pd.notna(p) else 'NaT'}`"
                    for r, p in zip(raw_samples, parsed_samples)
                ]
                st.caption("Parse preview: " + "  |  ".join(preview_pairs))
            except Exception:
                st.caption(f"Sample values: {', '.join(str(v) for v in raw_samples)}")
     
            fe_parts = st.multiselect(
                "Features to extract",
                ["Year", "Month", "Day", "DayOfWeek", "DayName",
                 "Quarter", "WeekOfYear", "IsWeekend", "MonthName"],
                default=["Year", "Month", "Day", "DayOfWeek", "Quarter"],
                key="fe_date_parts"
            )
     
            if st.button("Extract Date Features", key="fe_extract_btn", type="primary",
                         disabled=len(fe_parts) == 0):
                try:
                    df_t  = st.session_state.df.copy()
                    pdf_t = st.session_state.processed_df.copy()
     
                    # Parse column regardless of current dtype
                    parsed_col = pd.Series([pd.NaT] * len(df_t), index=df_t.index)
                    raw = df_t[fe_date_col]
                    for fmt in [
                        "%m/%d/%Y %H:%M", "%m/%d/%Y",
                        "%m-%d-%Y %H:%M", "%m-%d-%Y",
                        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
                        "%d/%m/%Y %H:%M", "%d/%m/%Y",
                        "%d-%m-%Y %H:%M", "%d-%m-%Y",
                    ]:
                        still_null = parsed_col.isna() & raw.notna()
                        if not still_null.any():
                            break
                        try:
                            attempt = pd.to_datetime(raw[still_null], format=fmt, errors="coerce")
                            parsed_col[still_null] = attempt
                        except Exception:
                            continue
                    still_null = parsed_col.isna() & raw.notna()
                    if still_null.any():
                        parsed_col[still_null] = pd.to_datetime(raw[still_null], errors="coerce")
     
                    extract_map = {
                        "Year":       lambda s: s.dt.year,
                        "Month":      lambda s: s.dt.month,
                        "Day":        lambda s: s.dt.day,
                        "DayOfWeek":  lambda s: s.dt.dayofweek,
                        "DayName":    lambda s: s.dt.day_name(),
                        "Quarter":    lambda s: s.dt.quarter,
                        "WeekOfYear": lambda s: s.dt.isocalendar().week.astype(int),
                        "IsWeekend":  lambda s: s.dt.dayofweek.isin([5, 6]).astype(int),
                        "MonthName":  lambda s: s.dt.month_name(),
                    }
     
                    created_cols = []
                    for part in fe_parts:
                        new_col = f"{fe_date_col}_{part.lower()}"
                        vals = extract_map[part](parsed_col)
                        df_t[new_col]  = vals
                        pdf_t[new_col] = vals
                        created_cols.append(new_col)
                        if new_col not in st.session_state.fe_created_cols:
                            st.session_state.fe_created_cols.append(new_col)
     
                    st.session_state.df = df_t
                    st.session_state.processed_df = pdf_t
                    save_operation(
                        st.session_state.file_name,
                        f"Feature Engineering: {fe_date_col}",
                        f"Extracted: {', '.join(fe_parts)}"
                    )
     
                    # Show the extracted data immediately
                    st.success(f"✅ Created {len(created_cols)} new columns: {', '.join(created_cols)}")
                    preview_cols = [fe_date_col] + created_cols
                    st.markdown("**Preview — extracted columns (first 10 rows):**")
                    st.dataframe(
                        df_t[preview_cols].head(10),
                        use_container_width=True,
                        height=300
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Extraction error: {e}")
     
        st.markdown("---")
     
        # ═══════════════════════════════
        # SECTION 2 — Redundant Column Detection
        # ═══════════════════════════════
        st.markdown("**🗑️ Detect & Drop Redundant Columns**")
        st.caption("Auto-detects standalone year/month/day columns, near-duplicate numeric columns, and zero-variance columns.")
     
        df_fe = st.session_state.df  # refresh after extraction
     
        # Keywords that indicate a standalone temporal column
        TEMPORAL_PATTERNS = {
            "year":    ["year", "yr", "anno"],
            "month":   ["month", "mon", "mth"],
            "day":     ["dayofweek", "day_of_week", "weekday"],
            "quarter": ["quarter", "qtr", "q1", "q2", "q3", "q4"],
            "week":    ["week", "weekofyear", "week_no"],
        }
     
        def _is_standalone_temporal(col_name):
            """
            True if the column name IS a temporal unit name (year, month, etc.)
            but is NOT itself a date column (e.g. 'OrderDate' is not standalone temporal).
            """
            cl = col_name.lower().replace(" ", "_").replace("-", "_")
            # Skip columns that contain 'date' or 'time' — those are date columns, not standalones
            if any(k in cl for k in ["date", "time", "datetime", "timestamp"]):
                return False
            for unit, patterns in TEMPORAL_PATTERNS.items():
                for p in patterns:
                    # whole-word boundary match
                    if re.fullmatch(p, cl) or re.search(rf"(^|_){re.escape(p)}($|_)", cl):
                        return unit
            return False
     
        redundancy_suggestions = []
     
        # Rule 1: High correlation between numeric columns
        num_df_r = df_fe.select_dtypes(include=[np.number])
        if num_df_r.shape[1] > 1:
            try:
                corr_r = num_df_r.corr().abs()
                for i in range(len(corr_r.columns)):
                    for j in range(i + 1, len(corr_r.columns)):
                        v = corr_r.iloc[i, j]
                        if v > 0.98:
                            c1, c2 = corr_r.columns[i], corr_r.columns[j]
                            keep    = c1 if len(c1) >= len(c2) else c2
                            suggest = c2 if keep == c1 else c1
                            redundancy_suggestions.append({
                                "Column to drop": suggest,
                                "Reason": f"Near-duplicate of '{keep}' (r = {v:.3f})",
                                "Keep": keep
                            })
            except Exception:
                pass
     
        # Rule 2: Standalone temporal columns whose info is already in an extracted column
        extracted_fe = set(st.session_state.fe_created_cols)
        for col in df_fe.columns:
            unit = _is_standalone_temporal(col)
            if not unit:
                continue
            # Check if we have an extracted column that covers this unit
            matching_extracted = [
                c for c in extracted_fe
                if unit in c.lower() and c in df_fe.columns
            ]
            # Also check if another column literally contains the same values
            matching_by_value = []
            col_vals = df_fe[col].dropna()
            for other in df_fe.columns:
                if other == col or other not in df_fe.select_dtypes(include=[np.number]).columns:
                    continue
                other_vals = df_fe[other].dropna()
                if len(col_vals) == len(other_vals):
                    try:
                        if np.corrcoef(col_vals.astype(float), other_vals.astype(float))[0, 1] > 0.98:
                            matching_by_value.append(other)
                    except Exception:
                        pass
     
            if matching_extracted:
                redundancy_suggestions.append({
                    "Column to drop": col,
                    "Reason": f"Standalone '{unit}' — already extracted as '{matching_extracted[0]}'",
                    "Keep": matching_extracted[0]
                })
            elif matching_by_value:
                redundancy_suggestions.append({
                    "Column to drop": col,
                    "Reason": f"Identical values as '{matching_by_value[0]}'",
                    "Keep": matching_by_value[0]
                })
            elif unit in ["year", "month"]:
                # Flag even without a match — standalone year/month are often redundant
                redundancy_suggestions.append({
                    "Column to drop": col,
                    "Reason": f"Standalone '{unit}' column — likely derivable from a date column",
                    "Keep": "— (consider extracting from date first)"
                })
     
        # Rule 3: Zero / near-zero variance
        for col in df_fe.columns:
            if df_fe[col].nunique(dropna=True) <= 1:
                redundancy_suggestions.append({
                    "Column to drop": col,
                    "Reason": "Single unique value — zero variance",
                    "Keep": "—"
                })
     
        # Deduplicate
        seen_drops = set()
        unique_suggestions = []
        for s in redundancy_suggestions:
            if s["Column to drop"] not in seen_drops:
                seen_drops.add(s["Column to drop"])
                unique_suggestions.append(s)
     
        if not unique_suggestions:
            st.markdown("<span class='badge badge-success'>✅ No redundant columns detected</span>", unsafe_allow_html=True)
        else:
            sugg_df = pd.DataFrame(unique_suggestions)
            st.dataframe(sugg_df, use_container_width=True, hide_index=True)
     
            cols_to_drop = st.multiselect(
                "Select columns to drop",
                options=[s["Column to drop"] for s in unique_suggestions],
                default=[s["Column to drop"] for s in unique_suggestions],
                key="fe_drop_cols"
            )
            if st.button("Drop Selected Columns", key="fe_drop_btn", type="primary",
                         disabled=len(cols_to_drop) == 0):
                df_t  = st.session_state.df.drop(columns=cols_to_drop, errors="ignore")
                pdf_t = st.session_state.processed_df.drop(columns=cols_to_drop, errors="ignore")
                # Remove dropped cols from fe_created_cols cache too
                st.session_state.fe_created_cols = [
                    c for c in st.session_state.fe_created_cols if c not in cols_to_drop
                ]
                st.session_state.df = df_t
                st.session_state.processed_df = pdf_t
                save_operation(
                    st.session_state.file_name,
                    "Drop Redundant Columns",
                    f"Dropped: {', '.join(cols_to_drop)}"
                )
                st.success(f"✅ Dropped {len(cols_to_drop)} column(s): {', '.join(cols_to_drop)}")
                st.rerun()
     
        st.markdown("---")
     
        # ═══════════════════════════════
        # SECTION 3 — Phone Number Cleaning
        # ═══════════════════════════════
        st.markdown("**📞 Phone Number Cleaning**")
        st.caption("Detects phone/mobile/fax columns only — excludes ContactName, TelAviv, MobileSuit, etc.")
     
        df_fe = st.session_state.df  # refresh
     
        phone_cols = [
            c for c in df_fe.select_dtypes(include="object").columns
            if _is_phone_column(c)
        ]
     
        if not phone_cols:
            st.info(
                "No phone number columns detected. "
                "Detected columns must contain: phone, mobile, fax, tel, cell, whatsapp, landline, "
                "or 'contact' as a standalone word — but NOT contactname, MobileSuit, TelAviv, etc."
            )
        else:
            ph_col = st.selectbox("Phone column", phone_cols, key="ph_col_select")
            ph_series = df_fe[ph_col].dropna().astype(str)
     
            def classify_phone(val):
                digits = re.sub(r"[^\d]", "", val)
                n = len(digits)
                if n == 0:   return val, "invalid", "No digits found"
                elif n < 7:  return val, "invalid", f"Too short ({n} digits)"
                elif n == 8: return digits, "warning", "8-digit (local format)"
                elif 9 <= n <= 15: return digits, "valid", f"{n}-digit number"
                else:        return val, "invalid", f"Too long ({n} digits)"
     
            results = ph_series.apply(classify_phone)
            valid_ct   = sum(1 for _, s, _ in results if s == "valid")
            warning_ct = sum(1 for _, s, _ in results if s == "warning")
            invalid_ct = sum(1 for _, s, _ in results if s == "invalid")
     
            ph1, ph2, ph3 = st.columns(3)
            for w, lbl, cnt, color in [
                (ph1, "Valid",    valid_ct,   "#16a34a"),
                (ph2, "8-digit",  warning_ct, "#d97706"),
                (ph3, "Invalid",  invalid_ct, "#dc2626"),
            ]:
                with w:
                    st.markdown(f"""<div class='metric-card'>
                        <span class='val' style='color:{color};'>{cnt}</span>
                        <span class='label'>{lbl}</span></div>""", unsafe_allow_html=True)
     
            st.markdown("&nbsp;")
     
            invalid_samples = [
                {"Original value": val, "Issue": reason}
                for val, (_, status, reason) in zip(ph_series, results)
                if status == "invalid"
            ][:10]
            if invalid_samples:
                with st.expander(f"👁️ Preview invalid entries ({min(10, invalid_ct)} shown)"):
                    st.dataframe(pd.DataFrame(invalid_samples), use_container_width=True, hide_index=True)
     
            ph_action = st.radio(
                "Action",
                [
                    "Standardise — strip to digits only, keep all rows",
                    "Flag — add new column '{col}_valid' (1 = valid, 0 = invalid)",
                    "Drop rows with invalid numbers",
                    "Replace invalid with NaN",
                ],
                key="ph_action_radio"
            )
     
            if st.button("Apply Phone Cleaning", key="ph_clean_btn", type="primary"):
                try:
                    df_t  = st.session_state.df.copy()
                    pdf_t = st.session_state.processed_df.copy()
     
                    def clean_val(val):
                        if pd.isna(val): return val
                        d = re.sub(r"[^\d]", "", str(val))
                        return d if d else np.nan
     
                    def is_valid(val):
                        if pd.isna(val): return 0
                        d = re.sub(r"[^\d]", "", str(val))
                        return 1 if 7 <= len(d) <= 15 else 0
     
                    if "Standardise" in ph_action:
                        df_t[ph_col] = df_t[ph_col].apply(clean_val)
                        pdf_t[ph_col] = pdf_t[ph_col].apply(clean_val)
                        msg = f"✅ Standardised '{ph_col}' to digits only."
     
                    elif "Flag" in ph_action:
                        flag_col = f"{ph_col}_valid"
                        df_t[flag_col] = df_t[ph_col].apply(is_valid)
                        pdf_t[flag_col] = pdf_t[ph_col].apply(is_valid)
                        if flag_col not in st.session_state.fe_created_cols:
                            st.session_state.fe_created_cols.append(flag_col)
                        msg = f"✅ Added '{flag_col}' column (1 = valid, 0 = invalid)."
     
                    elif "Drop rows" in ph_action:
                        before = len(df_t)
                        mask = df_t[ph_col].apply(is_valid).astype(bool)
                        df_t  = df_t[mask].reset_index(drop=True)
                        pdf_t = pdf_t[mask].reset_index(drop=True)
                        msg = f"✅ Removed {before - len(df_t)} rows with invalid phone numbers."
     
                    else:
                        def nullify(val):
                            if pd.isna(val): return val
                            d = re.sub(r"[^\d]", "", str(val))
                            return val if 7 <= len(d) <= 15 else np.nan
                        df_t[ph_col] = df_t[ph_col].apply(nullify)
                        pdf_t[ph_col] = pdf_t[ph_col].apply(nullify)
                        msg = f"✅ Invalid phone values replaced with NaN."
     
                    st.session_state.df = df_t
                    st.session_state.processed_df = pdf_t
                    save_operation(st.session_state.file_name, f"Phone Cleaning: {ph_col}", ph_action)
                    st.success(msg)
                    st.rerun()
                except Exception as e:
                    st.error(f"Phone cleaning error: {e}")
    nav_buttons("Cleaning")
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

    # ── Encoding Tab (tab1) — Redesigned ─────────────────────────────────────
    with tab1:

        encoding_df = st.session_state.original_df
        ct = identify_column_types(encoding_df)

        # ── Keywords that disqualify a column from encoding ───────────────────
        SKIP_KEYWORDS = [
            "name", "id", "index", "key", "uuid", "guid",
            "phone", "mobile", "tel", "contact",
            "date", "time", "datetime", "timestamp", "created", "updated",
            "dob", "born", "joined", "period", "year", "month", "day",
            "email", "mail",
        ]

        def _should_skip_for_encoding(col_name):
            cl = col_name.lower().replace(" ", "_").replace("-", "_")
            return any(kw in cl for kw in SKIP_KEYWORDS)

        # All categorical + boolean, excluding ID/name/date/phone columns
        all_cat_cols = [
            c for c in list(dict.fromkeys(ct["categorical"] + ct["boolean"]))
            if not _should_skip_for_encoding(c)
            and c not in ct["id"]
            and c not in ct["datetime"]
        ]

        # Already encoded columns
        encoded_set = set(st.session_state.get("encoded_columns", []))

        def _is_ohe_dummy(col, enc_set):
            return any(col.startswith(ec + "_") for ec in enc_set)

        available_cols = [
            c for c in all_cat_cols
            if c not in encoded_set
            and not _is_ohe_dummy(c, encoded_set)
        ]

        # ═══════════════════════════════════════════
        # SECTION A — DATA TYPE CONVERSION
        # ═══════════════════════════════════════════
        st.markdown("<div class='section-header'><h3>🔄 Data Type Conversion</h3></div>", unsafe_allow_html=True)

        st.markdown("""
        <div style='background:#fffbeb;border:1px solid #fde68a;border-radius:10px;
             padding:12px 16px;margin-bottom:16px;font-size:0.84rem;color:#92400e;'>
            💡 Use this to fix columns with wrong types — e.g. an <b>Order Date</b> stored as text,
            or a numeric ID stored as float. Changes apply to both the working and processed dataset.
        </div>
        """, unsafe_allow_html=True)

        all_cols = list(st.session_state.df.columns)

        dt1, dt2, dt3 = st.columns([3, 2, 2])
        with dt1:
            dtype_col = st.selectbox(
                "Column",
                all_cols,
                key="dtype_col_select",
                help="Select any column to change its data type"
            )
        with dt2:
            current_dtype = str(st.session_state.df[dtype_col].dtype)
            st.markdown(f"""
            <div style='padding:8px 0 4px;'>
                <div style='font-size:0.72rem;color:#6b7280;text-transform:uppercase;
                     letter-spacing:1px;margin-bottom:4px;'>Current type</div>
                <div style='font-family:"JetBrains Mono",monospace;font-size:0.9rem;
                     color:#374151;font-weight:700;background:#f1f3f9;
                     border-radius:6px;padding:6px 10px;display:inline-block;'>{current_dtype}</div>
            </div>
            """, unsafe_allow_html=True)
        with dt3:
            target_dtype = st.selectbox(
                "Convert to",
                [
                    "datetime — parse dates automatically",
                    "datetime — custom format",
                    "int — integer",
                    "float — decimal",
                    "str — text / object",
                    "bool — True / False",
                    "category — pandas category",
                ],
                key="dtype_target_select"
            )

        # Show format input only for custom datetime
        custom_fmt = None
        if "custom format" in target_dtype:
            st.markdown("""
            <div style='font-size:0.8rem;color:#374151;margin:6px 0 2px;'>
                <b>Date format string</b>
                <span style='color:#6b7280;'> — use Python strftime codes</span>
            </div>
            """, unsafe_allow_html=True)
            fmt_col1, fmt_col2 = st.columns([3, 2])
            with fmt_col1:
                custom_fmt = st.text_input(
                    "Format",
                    placeholder="%Y-%m-%d %H:%M:%S",
                    label_visibility="collapsed",
                    key="dtype_custom_fmt"
                )
            with fmt_col2:
                # Show sample values as hint
                sample_vals = st.session_state.df[dtype_col].dropna().head(3).tolist()
                st.caption(f"Sample: {', '.join(str(v) for v in sample_vals)}")

        # Preview sample after conversion
        with st.expander("👁️ Preview conversion on first 5 rows", expanded=False):
            try:
                preview_s = st.session_state.df[dtype_col].head(5).copy()
                dtype_key = target_dtype.split(" — ")[0]
                if dtype_key == "datetime":
                    if custom_fmt:
                        converted = pd.to_datetime(preview_s, format=custom_fmt, errors="coerce")
                    else:
                        converted = pd.to_datetime(preview_s, errors="coerce")
                elif dtype_key == "int":
                    converted = pd.to_numeric(preview_s, errors="coerce").astype("Int64")
                elif dtype_key == "float":
                    converted = pd.to_numeric(preview_s, errors="coerce")
                elif dtype_key == "str":
                    converted = preview_s.astype(str)
                elif dtype_key == "bool":
                    converted = preview_s.astype(bool)
                elif dtype_key == "category":
                    converted = preview_s.astype("category")
                else:
                    converted = preview_s

                prev_df = pd.DataFrame({
                    "Original": preview_s.values,
                    "Converted": converted.values
                })
                st.dataframe(prev_df, use_container_width=True, height=200)
            except Exception as e:
                st.error(f"Preview error: {e}")

        if st.button("Apply Type Conversion", key="apply_dtype_btn", type="primary"):
            try:
                df_t = st.session_state.df.copy()
                pdf_t = st.session_state.processed_df.copy()
                dtype_key = target_dtype.split(" — ")[0]

                def _convert(series):
                    if dtype_key == "datetime":
                        if custom_fmt:
                            # User supplied a specific format — try it first, fall back to auto
                            parsed = pd.to_datetime(series, format=custom_fmt, errors="coerce")
                            # For rows that failed, retry with flexible parser
                            failed_mask = parsed.isna() & series.notna()
                            if failed_mask.any():
                                parsed[failed_mask] = pd.to_datetime(
                                    series[failed_mask], errors="coerce"
                                )
                            return parsed
                        else:
                            # Mixed-format path — try slash format, dash format, then generic
                            # Handles: 2/24/2003 0:00  AND  05-07-2003 00:00
                            result = pd.Series([pd.NaT] * len(series), index=series.index)
                            remaining = series.copy()
                 
                            for fmt in ["%m/%d/%Y %H:%M", "%m/%d/%Y", "%m-%d-%Y %H:%M",
                                        "%m-%d-%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
                                        "%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M", "%d-%m-%Y"]:
                                still_null = result.isna() & remaining.notna()
                                if not still_null.any():
                                    break
                                try:
                                    attempt = pd.to_datetime(
                                        remaining[still_null], format=fmt, errors="coerce"
                                    )
                                    result[still_null] = attempt
                                except Exception:
                                    continue
                 
                            # Final fallback for anything still unparsed
                            still_null = result.isna() & remaining.notna()
                            if still_null.any():
                                result[still_null] = pd.to_datetime(
                                    remaining[still_null], errors="coerce"
                                )
                            return result
                 
                    elif dtype_key == "int":
                        return pd.to_numeric(series, errors="coerce").astype("Int64")
                    elif dtype_key == "float":
                        return pd.to_numeric(series, errors="coerce")
                    elif dtype_key == "str":
                        return series.astype(str)
                    elif dtype_key == "bool":
                        return series.astype(bool)
                    elif dtype_key == "category":
                        return series.astype("category")
                    return series
                df_t[dtype_col]  = _convert(df_t[dtype_col])
                pdf_t[dtype_col] = _convert(pdf_t[dtype_col])

                null_count = df_t[dtype_col].isna().sum()

                st.session_state.df = df_t
                st.session_state.processed_df = pdf_t
                save_operation(
                    st.session_state.file_name,
                    f"Type Conversion: {dtype_col}",
                    f"{current_dtype} → {dtype_key}"
                )
                st.success(f"✅ '{dtype_col}' converted to **{dtype_key}**.")
                if null_count > 0:
                    st.warning(f"⚠️ {null_count} value(s) could not be converted and were set to NaT/NaN.")
                st.rerun()
            except Exception as e:
                st.error(f"Conversion error: {e}")

        st.markdown("---")

        # ═══════════════════════════════════════════
        # SECTION B — CATEGORICAL ENCODING
        # ═══════════════════════════════════════════
        st.markdown("<div class='section-header'><h3>🔠 Categorical Encoding</h3></div>", unsafe_allow_html=True)

        # Show which columns were auto-skipped
        skipped_cols = [
            c for c in list(dict.fromkeys(ct["categorical"] + ct["boolean"]))
            if _should_skip_for_encoding(c) or c in ct["id"] or c in ct["datetime"]
        ]
        if skipped_cols:
            skip_chips = "".join([
                f"<span style='display:inline-flex;background:#f1f3f9;border:1px solid #e2e6f0;"
                f"color:#6b7280;border-radius:999px;padding:2px 10px;font-size:0.72rem;"
                f"font-weight:600;margin:2px;'>{c}</span>"
                for c in skipped_cols
            ])
            st.markdown(f"""
            <div style='margin-bottom:14px;'>
                <span style='font-size:0.72rem;color:#6b7280;text-transform:uppercase;
                     letter-spacing:1px;'>Auto-excluded (ID / name / date / phone)&nbsp;</span>
                {skip_chips}
            </div>
            """, unsafe_allow_html=True)

        # ── All done ──────────────────────────────────────────────────────────
        if not available_cols and all_cat_cols:
            st.success("✅ All categorical columns have been encoded.")
            
            chips_html = "".join([
                f"<span style='display:inline-flex;align-items:center;gap:5px;"
                f"background:#f0fdf4;border:1px solid #bbf7d0;color:#16a34a;"
                f"border-radius:999px;padding:3px 12px;font-size:0.75rem;"
                f"font-weight:600;margin:3px;'>✓ {c}</span>"
                for c in sorted(encoded_set) if c in all_cat_cols
            ])
            st.markdown(f"""
            <div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;
                 padding:16px 20px;margin-top:8px;'>
                <div style='font-size:0.72rem;color:#16a34a;font-weight:700;
                     text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;'>
                     Encoded columns</div>
                <div style='line-height:2.2;'>{chips_html}</div>
            </div>
            """, unsafe_allow_html=True)

        elif not all_cat_cols:
            st.info("No encodable categorical columns found (ID, name, date, and phone columns are excluded).")

        else:
            # Progress bar
            done  = len(all_cat_cols) - len(available_cols)
            total = len(all_cat_cols)
            pct   = int(done / total * 100) if total else 0

            st.markdown(f"""
            <div style='margin-bottom:18px;'>
                <div style='display:flex;justify-content:space-between;
                     align-items:center;margin-bottom:6px;'>
                    <span style='font-size:0.8rem;color:#6b7280;'>Encoding progress</span>
                    <span style='font-family:"JetBrains Mono",monospace;font-size:0.8rem;
                         color:#2563eb;font-weight:700;'>{done} / {total}</span>
                </div>
                <div style='background:#e2e6f0;border-radius:999px;height:5px;'>
                    <div style='width:{pct}%;height:100%;background:#2563eb;
                         border-radius:999px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Already encoded chips
            if encoded_set:
                chips_html = "".join([
                    f"<span style='display:inline-flex;background:#eff6ff;border:1px solid #bfdbfe;"
                    f"color:#2563eb;border-radius:999px;padding:2px 10px;font-size:0.72rem;"
                    f"font-weight:600;margin:2px;'>✓ {c}</span>"
                    for c in sorted(encoded_set) if c in all_cat_cols
                ])
                st.markdown(f"""
                <div style='margin-bottom:14px;'>
                    <span style='font-size:0.72rem;color:#6b7280;text-transform:uppercase;
                         letter-spacing:1px;'>Already encoded&nbsp;</span>{chips_html}
                </div>
                """, unsafe_allow_html=True)

            # Main form panel
            st.markdown("""
            <div style='background:#ffffff;border:1px solid #e2e6f0;border-radius:12px;
                 padding:20px 24px;'>
            """, unsafe_allow_html=True)

            enc_technique = st.selectbox(
                "Encoding Technique",
                [
                    "Label Encoding",
                    "One-Hot Encoding",
                    "Ordinal Encoding",
                    "Frequency Encoding",
                    "Binary Encoding",
                ],
                key="enc_technique_unified",
                help=(
                    "Label: integer codes 0…N-1  |  "
                    "One-Hot: new binary column per category  |  "
                    "Ordinal: user-defined rank order  |  "
                    "Frequency: replace with category frequency  |  "
                    "Binary: label → binary bit columns"
                )
            )

            hints = {
                "Label Encoding":     "Best for binary columns or tree-based models.",
                "One-Hot Encoding":   "Best for low-cardinality columns (≤ 10 unique values).",
                "Ordinal Encoding":   "Use when categories have a natural order (e.g. low → medium → high).",
                "Frequency Encoding": "Best for high-cardinality columns — replaces each category with its relative frequency.",
                "Binary Encoding":    "Label → binary bit columns. Compact alternative to one-hot for medium cardinality.",
            }
            st.caption(hints[enc_technique])

            col_labels = {c: f"{c}  ({encoding_df[c].nunique()} unique)" for c in available_cols}
            selected_cols = st.multiselect(
                "Columns to Encode",
                options=available_cols,
                format_func=lambda c: col_labels[c],
                placeholder="Select one or more columns…",
                key="enc_cols_unified"
            )

            # Ordinal order inputs
            ordinal_orders = {}
            if enc_technique == "Ordinal Encoding" and selected_cols:
                st.markdown(
                    "<div style='margin-top:10px;font-size:0.82rem;color:#374151;"
                    "font-weight:600;margin-bottom:4px;'>Define rank order "
                    "<span style='color:#6b7280;font-weight:400;'>"
                    "(comma-separated, lowest → highest)</span></div>",
                    unsafe_allow_html=True
                )
                for col in selected_cols:
                    existing = sorted(encoding_df[col].dropna().unique().tolist())
                    ord_input = st.text_input(
                        f"{col}",
                        placeholder=f"e.g. {', '.join(str(v) for v in existing[:5])}",
                        key=f"ordinal_order_{col}",
                        help=f"All values: {existing}"
                    )
                    ordinal_orders[col] = ord_input

            st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

            apply_btn = st.button(
                "Apply Encoding",
                type="primary",
                disabled=len(selected_cols) == 0,
                key="apply_encoding_unified"
            )

            st.markdown("</div>", unsafe_allow_html=True)

            # Apply logic
            if apply_btn and selected_cols:
                errors    = []
                successes = []

                for col in selected_cols:
                    try:
                        working_df = st.session_state.processed_df.copy()

                        if enc_technique == "Label Encoding":
                            new_df, _ = apply_encoding(working_df, col, "label")

                        elif enc_technique == "One-Hot Encoding":
                            new_df, _ = apply_encoding(working_df, col, "onehot")

                        elif enc_technique == "Frequency Encoding":
                            new_df, _ = apply_encoding(working_df, col, "frequency")

                        elif enc_technique == "Binary Encoding":
                            le = LabelEncoder()
                            labels   = le.fit_transform(working_df[col].astype(str))
                            max_bits = max(int(np.ceil(np.log2(len(le.classes_) + 1))), 1)
                            new_df   = working_df.copy()
                            for bit in range(max_bits):
                                new_df[f"{col}_bin{bit}"] = (labels >> bit) & 1
                            new_df = new_df.drop(columns=[col])
                            st.session_state.encoders[col] = {
                                cls: int(code) for code, cls in enumerate(le.classes_)
                            }

                        elif enc_technique == "Ordinal Encoding":
                            ord_str = ordinal_orders.get(col, "")
                            if not ord_str.strip():
                                errors.append(f"'{col}': no ordinal order provided.")
                                continue
                            ordinal_order = [x.strip() for x in ord_str.split(",")]
                            existing_vals = [str(v).strip() for v in
                                             encoding_df[col].dropna().unique()]
                            invalid = [v for v in ordinal_order if v not in existing_vals]
                            if invalid:
                                errors.append(
                                    f"'{col}': values not found in column: {invalid}"
                                )
                                continue
                            val_map      = {str(v).strip(): v for v in encoding_df[col].dropna().unique()}
                            mapped_order = [val_map.get(x, x) for x in ordinal_order]
                            new_df, _    = apply_encoding(working_df, col, "ordinal", mapped_order)

                        else:
                            errors.append(f"'{col}': unknown technique.")
                            continue

                        st.session_state.processed_df = new_df
                        st.session_state.df           = new_df
                        if col not in st.session_state.encoded_columns:
                            st.session_state.encoded_columns.append(col)
                        save_operation(
                            st.session_state.file_name,
                            f"Encoding [{enc_technique}]: {col}",
                            enc_technique
                        )
                        successes.append(col)

                    except Exception as e:
                        errors.append(f"'{col}': {e}")

                if successes:
                    st.success(f"Successfully encoded: {', '.join(successes)}")
                for err in errors:
                    st.error(f"❌ {err}")

                encoded_set_after = set(st.session_state.encoded_columns)
                remaining = [
                    c for c in all_cat_cols
                    if c not in encoded_set_after
                    and not _is_ohe_dummy(c, encoded_set_after)
                ]
                if not remaining and successes:
                    st.success("✅ All categorical columns have been encoded.")
                    

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
                            save_operation(st.session_state.file_name, f"Remove Outliers: {selected_col}", str(r))
                            st.success(f"Removed {r['removed']} outlier rows."); st.rerun()
                        except Exception as e: st.error(str(e))
                with cb:
                    if st.button("📌 Cap Outliers (Winsorise)"):
                        try:
                            new_df, r = cap_outliers(df, selected_col)
                            st.session_state.df = new_df
                            save_operation(st.session_state.file_name, f"Cap Outliers: {selected_col}", str(r))
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
        <p>Interactive charts with filtering, grouping and data controls</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please upload a dataset first.")
        st.stop()

    df = st.session_state.df
    ct = identify_column_types(df)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    
    # ── Date grouping helper ─────────────────────────────────────────────────
    def apply_date_grouping(df, col, freq):
        df = df.copy()
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            if freq == "Month":
                df[col] = df[col].dt.to_period("M").astype(str)
            elif freq == "Year":
                df[col] = df[col].dt.to_period("Y").astype(str)
            elif freq == "Quarter":
                df[col] = df[col].dt.to_period("Q").astype(str)
            elif freq == "Week":
                df[col] = df[col].dt.to_period("W").astype(str)
            elif freq == "Day":
                df[col] = df[col].dt.date.astype(str)
        except Exception:
            pass
        return df
 
   
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Custom Plot", "🔵 Scatter", "📦 Box & Violin", "📊 Category Breakdown"
    ])

    # ══════════════════════════════════════════
    # TAB 1 — CUSTOM PLOT
    # ══════════════════════════════════════════
    with tab1:
        st.markdown(
            "<div class='section-header'><h3>Custom Chart Builder</h3></div>",
            unsafe_allow_html=True
        )

        with st.container():
            st.markdown("""
            <div style='background:#ffffff;border:1px solid #e2e6f0;border-radius:12px;
                 padding:16px 20px;margin-bottom:16px;'>
            <div style='font-size:0.8rem;color:#6b7280;font-weight:600;
                 text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;'>
                 Chart Settings</div>
            """, unsafe_allow_html=True)

            cp1, cp2, cp3, cp4 = st.columns(4)
            with cp1:
                chart_type = st.selectbox(
                    "Chart Type", ["Line", "Bar", "Histogram", "Area"],
                    key="custom_chart_type"
                )
            with cp2:
                x_col = st.selectbox("X Axis", df.columns.tolist(), key="custom_x")
            with cp3:
                y_col = st.selectbox("Y Axis", ["— None —"] + num_cols, key="custom_y")
            with cp4:
                color_col = st.selectbox(
                    "Color by", ["— None —"] + cat_cols, key="custom_color"
                )

            # ── detect if X is a date column ──
            ct_live   = identify_column_types(df)
            x_is_date = (
                x_col in ct_live["datetime"]
                or (x_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[x_col]))
                or (
                    x_col in df.columns
                    and df[x_col].dtype == object
                    and any(k in x_col.lower() for k in
                            ["date", "month", "year", "time", "day", "week", "period"])
                )
            )

            if x_is_date:
                st.markdown("""
                <div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;
                     padding:10px 14px;margin:8px 0;font-size:0.83rem;color:#2563eb;'>
                    📅 <b>Date column detected</b> — group by a time period below
                </div>
                """, unsafe_allow_html=True)
                dg1, dg2, dg3, dg4 = st.columns(4)
                with dg1:
                    date_freq = st.radio(
                        "Group dates by",
                        ["None", "Day", "Week", "Month", "Quarter", "Year"],
                        horizontal=True, key="date_freq"
                    )
                with dg2:
                    date_agg = st.selectbox(
                        "Aggregate Y by",
                        ["Sum", "Mean", "Count", "Median", "Max", "Min"],
                        key="date_agg"
                    )
                with dg3:
                    fill_gaps = st.checkbox(
                        "Fill missing periods with 0",
                        value=False, key="date_fill_gaps"
                    )
                with dg4:
                    date_sort = st.checkbox(
                        "Sort by date", value=True, key="date_sort"
                    )
            else:
                date_freq = "None"
                date_agg  = "Sum"
                fill_gaps = False
                date_sort = True
                dg1, dg2, dg3, dg4 = st.columns(4)
                with dg1:
                    agg_func = st.selectbox(
                        "Aggregate Y by",
                        ["None (raw)", "Mean", "Sum", "Count", "Median", "Max", "Min"],
                        key="custom_agg"
                    )
                with dg2:
                    group_col = st.selectbox(
                        "Group X by", ["— None —"] + cat_cols, key="custom_group"
                    )
                with dg3:
                    sort_order = st.selectbox(
                        "Sort by",
                        ["None", "X ascending", "X descending",
                         "Y ascending", "Y descending"],
                        key="custom_sort"
                    )
                with dg4:
                    show_labels = st.checkbox(
                        "Show value labels", value=False, key="custom_labels"
                    )

            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
            
        
        

        color_val = None if color_col == "— None —" else color_col
        y_val     = None if y_col     == "— None —" else y_col
      

        try:
            # ── DATE GROUPING PATH ──────────────────────────────────────────
            if x_is_date and date_freq != "None" and y_val:
                plot_df  = apply_date_grouping(plot_df, x_col, date_freq)
                agg_map  = {
                    "Sum": "sum", "Mean": "mean", "Count": "count",
                    "Median": "median", "Max": "max", "Min": "min"
                }
                agg_fn = agg_map.get(date_agg, "sum")

                if color_val:
                    plot_df = (plot_df
                               .groupby([x_col, color_val])[y_val]
                               .agg(agg_fn).reset_index())
                else:
                    plot_df = (plot_df
                               .groupby(x_col)[y_val]
                               .agg(agg_fn).reset_index())

                if date_sort:
                    plot_df = plot_df.sort_values(x_col)

                if fill_gaps and not color_val:
                    freq_map = {
                        "Day": "D", "Week": "W", "Month": "M",
                        "Quarter": "Q", "Year": "Y"
                    }
                    try:
                        all_periods = pd.period_range(
                            start=plot_df[x_col].min(),
                            end=plot_df[x_col].max(),
                            freq=freq_map.get(date_freq, "M")
                        ).astype(str)
                        plot_df = (plot_df
                                   .set_index(x_col)
                                   .reindex(all_periods, fill_value=0)
                                   .reset_index())
                        plot_df.columns = [x_col, y_val]
                    except Exception:
                        pass
                x_col_plot = x_col
                text_col = y_val if st.session_state.get("custom_labels", False) else None
                if chart_type == "Line":
                    fig = px.line(
                        plot_df, x=x_col_plot, y=y_val, color=color_val,
                        template="plotly_white", height=420, text=text_col
                    )
                    if show_labels:
                        fig.update_traces(textposition="top center")
                
                elif chart_type == "Bar":
                    fig = px.bar(
                        plot_df, x=x_col_plot, y=y_val, color=color_val,
                        template="plotly_white", height=420,
                        barmode="group", text=text_col
                    )
                    if show_labels:
                        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                
                elif chart_type == "Histogram":
                    fig = px.histogram(
                        plot_df, x=x_col_plot, color=color_val,
                        template="plotly_white", height=420, nbins=30
                    )
                    # histograms don't support text labels — skip
                
                elif chart_type == "Area":
                    fig = px.area(
                        plot_df, x=x_col_plot, y=y_val, color=color_val,
                        template="plotly_white", height=420, text=text_col
                    )
                    if show_labels:
                        fig.update_traces(textposition="top center")
                
                fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc")
                st.plotly_chart(fig, use_container_width=True)
                # ── summary strip ──
                total    = plot_df[y_val].sum()
                avg      = plot_df[y_val].mean()
                peak_row = plot_df.loc[plot_df[y_val].idxmax()]
                ms1, ms2, ms3, ms4 = st.columns(4)
                for w, lbl, val in [
                    (ms1, f"Total {y_val}",     f"{total:,.2f}"),
                    (ms2, f"Avg / {date_freq}", f"{avg:,.2f}"),
                    (ms3, "Peak period",        str(peak_row[x_col])),
                    (ms4, "Peak value",         f"{peak_row[y_val]:,.2f}"),
                ]:
                    with w:
                        st.markdown(f"""
                        <div style='background:#eff6ff;border:1px solid #bfdbfe;
                             border-radius:8px;padding:10px;text-align:center;'>
                            <div style='font-family:"JetBrains Mono",monospace;font-size:1rem;
                                 font-weight:700;color:#2563eb;'>{val}</div>
                            <div style='font-size:0.7rem;color:#6b7280;margin-top:3px;
                                 text-transform:uppercase;letter-spacing:1px;'>{lbl}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # ── NORMAL PATH ─────────────────────────────────────────────────
            else:
                group_val  = (None if st.session_state.get("custom_group","— None —") == "— None —"
                              else st.session_state.get("custom_group"))
                agg_func   = st.session_state.get("custom_agg",    "None (raw)")
                sort_order = st.session_state.get("custom_sort",   "None")
                show_labels= st.session_state.get("custom_labels", False)
            
                # ── LINE CHART ──────────────────────────────────────────────────
                if chart_type == "Line" and y_val:
                   
                    is_date = False
                    try:
                        parsed = pd.to_datetime(plot_df[x_col], errors="coerce")
                        if parsed.notna().sum() > len(plot_df) * 0.5:
                            plot_df[x_col] = parsed
                            is_date = True
                    except Exception:
                        pass
            
                    agg_choice = agg_func.lower().replace("none (raw)", "sum")
                    fn = {"mean":"mean","sum":"sum","count":"count",
                          "median":"median","max":"max","min":"min"}.get(agg_choice, "sum")
            
                    if is_date:
                        plot_df["_period"] = plot_df[x_col].dt.to_period("M").astype(str)
                        grp_col = "_period"
                    else:
                        grp_col = x_col
            
                    if color_val and color_val in plot_df.columns:
                        plot_df = plot_df.groupby([grp_col, color_val])[y_val].agg(fn).reset_index()
                    else:
                        plot_df = plot_df.groupby(grp_col)[y_val].agg(fn).reset_index()
                        color_val = None
            
                    plot_df = plot_df.sort_values(grp_col)
                    fig = px.line(
                        plot_df, x=grp_col, y=y_val, color=color_val,
                        template="plotly_white", height=420, markers=True,
                        labels={grp_col: x_col},
                        title=f"{y_val} by {x_col} (Monthly)"
                    )
                    fig.update_layout(
                        paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc",
                        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
                    )
                    if show_labels:
                        fig.update_traces(
                            text=plot_df[y_val],
                            texttemplate="%{text:,.0f}",
                            textposition="top center"
                        )
                    st.plotly_chart(fig, use_container_width=True, key="line_chart_normal")
            
                # ── BAR CHART ────────────────────────────────────────────────────
                elif chart_type == "Bar" and y_val:
                    if agg_func != "None (raw)" and group_val:
                        agg_map = {"Mean":"mean","Sum":"sum","Count":"count",
                                   "Median":"median","Max":"max","Min":"min"}
                        plot_df = (plot_df.groupby(group_val)[y_val]
                                   .agg(agg_map.get(agg_func,"sum")).reset_index())
                        x_col_plot = group_val
                    else:
                        x_col_plot = x_col
            
                    if sort_order == "X ascending":   plot_df = plot_df.sort_values(x_col_plot, ascending=True)
                    elif sort_order == "X descending": plot_df = plot_df.sort_values(x_col_plot, ascending=False)
                    elif sort_order == "Y ascending":  plot_df = plot_df.sort_values(y_val, ascending=True)
                    elif sort_order == "Y descending": plot_df = plot_df.sort_values(y_val, ascending=False)
            
                    fig = px.bar(
                        plot_df, x=x_col_plot, y=y_val, color=color_val,
                        template="plotly_white", height=420, barmode="group",
                        text=y_val if show_labels else None
                    )
                    if show_labels:
                        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
                    fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc")
                    st.plotly_chart(fig, use_container_width=True, key="bar_chart_normal")
            
                # ── AREA CHART ───────────────────────────────────────────────────
                elif chart_type == "Area" and y_val:
                    fig = px.area(
                        plot_df, x=x_col, y=y_val, color=color_val,
                        template="plotly_white", height=420,
                        text=y_val if show_labels else None
                    )
                    if show_labels:
                        fig.update_traces(texttemplate="%{text:,.0f}", textposition="top center")
                    fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc")
                    st.plotly_chart(fig, use_container_width=True, key="area_chart_normal")
            
                # ── HISTOGRAM ────────────────────────────────────────────────────
                elif chart_type == "Histogram":
                    fig = px.histogram(
                        plot_df, x=x_col, color=color_val,
                        template="plotly_white", height=420
                    )
                    fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc")
                    st.plotly_chart(fig, use_container_width=True, key="hist_chart_normal")
            
                elif not y_val:
                    st.info("Please select a Y Axis column to plot.")
        except Exception as e:
            st.error(f"Chart error: {e}")

            
    # ══════════════════════════════════════════
    # TAB 2 — SCATTER
    # ══════════════════════════════════════════
    with tab2:
        st.markdown(
            "<div class='section-header'><h3>Scatter Plot</h3></div>",
            unsafe_allow_html=True
        )
        if len(num_cols) < 2:
            st.info("Need at least 2 numerical columns.")
        else:
            with st.container():
                st.markdown("""
                <div style='background:#ffffff;border:1px solid #e2e6f0;border-radius:12px;
                     padding:16px 20px;margin-bottom:16px;'>
                <div style='font-size:0.8rem;color:#6b7280;font-weight:600;
                     text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;'>
                     Scatter Settings</div>
                """, unsafe_allow_html=True)

                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1: sx = st.selectbox("X Axis",  num_cols,        key="scatter_x")
                with sc2: sy = st.selectbox("Y Axis",  num_cols[::-1],  key="scatter_y")
                with sc3: sc = st.selectbox("Color by", ["— None —"] + cat_cols + num_cols, key="scatter_color")
                with sc4: sz = st.selectbox("Size by (bubble)", ["— None —"] + num_cols, key="scatter_size")

                sp1, sp2, sp3, sp4 = st.columns(4)
                with sp1: add_trendline = st.checkbox("Show trendline",  value=True, key="scatter_trend")
                with sp2: opacity       = st.slider("Opacity", 0.1, 1.0, 0.7, key="scatter_opacity")
                with sp3: marker_size   = st.slider("Marker size", 3, 20, 6,  key="scatter_msize")
                with sp4: facet_col     = st.selectbox("Facet by", ["— None —"] + cat_cols, key="scatter_facet")

                sb1, sb2, sb3, _ = st.columns([1, 1, 1, 3])
                with sb1: scatter_plot_btn  = st.button("🔵 Plot Scatter", key="scatter_plot_btn",  type="primary")
                with sb2: scatter_reset_btn = st.button("🔄 Reset",        key="scatter_reset_btn")
                with sb3: scatter_data_btn  = st.button("🗃️ Show Data",   key="scatter_data_btn")
                st.markdown("</div>", unsafe_allow_html=True)

            
            sc_color = None if sc         == "— None —" else sc
            sc_size  = None if sz         == "— None —" else sz
            facet    = None if facet_col  == "— None —" else facet_col

            try:
                fig_sc = px.scatter(
                    x=sx, y=sy, color=sc_color, size=sc_size,
                    template="plotly_white", height=450,
                    trendline="ols" if add_trendline and sc_color is None else None,
                    opacity=opacity, facet_col=facet
                )
                fig_sc.update_traces(marker=dict(size=marker_size))
                fig_sc.update_layout(
                    paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc"
                )
                st.plotly_chart(fig_sc, use_container_width=True)

                r         = filtered_df2[[sx, sy]].dropna().corr().iloc[0, 1]
                direction = "positive" if r > 0 else "negative"
                strength  = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"
                st.markdown(f"""
                <div style='background:#f8f9fc;border:1px solid #e2e6f0;border-radius:8px;
                     padding:12px 16px;font-size:0.88rem;color:#374151;'>
                    <b>Pearson r = {r:.4f}</b> — {strength} {direction} correlation
                    &nbsp;|&nbsp; <b>N = {len(filtered_df2[[sx,sy]].dropna()):,} rows</b>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Scatter error: {e}")
            if scatter_data_btn:
                st.dataframe(
                    filtered_df2[[sx, sy]].dropna(),
                    use_container_width=True, height=280
                )

    # ══════════════════════════════════════════
    # TAB 3 — BOX & VIOLIN
    # ══════════════════════════════════════════
    with tab3:
        st.markdown(
            "<div class='section-header'><h3>Box & Violin Plots</h3></div>",
            unsafe_allow_html=True
        )
        if not num_cols:
            st.info("No numerical columns.")
        else:
            with st.container():
                st.markdown("""
                <div style='background:#ffffff;border:1px solid #e2e6f0;border-radius:12px;
                     padding:16px 20px;margin-bottom:16px;'>
                <div style='font-size:0.8rem;color:#6b7280;font-weight:600;
                     text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;'>
                     Box / Violin Settings</div>
                """, unsafe_allow_html=True)

                bv1, bv2, bv3, bv4 = st.columns(4)
                with bv1: bv_col    = st.selectbox("Numerical column", num_cols, key="bv_col")
                with bv2: bv_group  = st.selectbox("Group by", ["— None —"] + cat_cols, key="bv_group")
                with bv3: bv_type   = st.radio("Plot type", ["Box","Violin","Both"], key="bv_type", horizontal=True)
                with bv4: bv_points = st.radio("Show points", ["outliers","all","none"], key="bv_points", horizontal=True)

               

            bv_group_val = None if bv_group == "— None —" else bv_group
            pts = False if bv_points == "none" else bv_points

            try:
                if bv_type == "Box":
                    fig_bv = px.box(
                        filtered_df3, y=bv_col, x=bv_group_val,
                        color=bv_group_val, template="plotly_white",
                        height=420, points=pts
                    )
                elif bv_type == "Violin":
                    fig_bv = px.violin(
                        filtered_df3, y=bv_col, x=bv_group_val,
                        color=bv_group_val, template="plotly_white",
                        height=420, box=True, points=pts
                    )
                else:
                    fig_bv = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=[f"Box — {bv_col}", f"Violin — {bv_col}"]
                    )
                    for val in (filtered_df3[bv_group_val].unique()
                                if bv_group_val else [None]):
                        subset = (filtered_df3[filtered_df3[bv_group_val] == val][bv_col].dropna()
                                  if bv_group_val else filtered_df3[bv_col].dropna())
                        name = str(val) if val is not None else bv_col
                        fig_bv.add_trace(
                            go.Box(y=subset, name=name, boxpoints=pts), row=1, col=1
                        )
                        fig_bv.add_trace(
                            go.Violin(y=subset, name=name, box_visible=True), row=1, col=2
                        )
                    fig_bv.update_layout(
                        template="plotly_white", height=420, showlegend=False
                    )
                fig_bv.update_layout(
                    paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc"
                )
                st.plotly_chart(fig_bv, use_container_width=True)
            except Exception as e:
                st.error(f"Plot error: {e}")


    # ══════════════════════════════════════════
    # TAB 4 — CATEGORY BREAKDOWN
    # ══════════════════════════════════════════
    with tab4:
        st.markdown(
            "<div class='section-header'><h3>Category Breakdown</h3></div>",
            unsafe_allow_html=True
        )
        if not cat_cols:
            st.info("No categorical columns.")
        else:
            with st.container():
                st.markdown("""
                <div style='background:#ffffff;border:1px solid #e2e6f0;border-radius:12px;
                     padding:16px 20px;margin-bottom:16px;'>
                <div style='font-size:0.8rem;color:#6b7280;font-weight:600;
                     text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;'>
                     Category Settings</div>
                """, unsafe_allow_html=True)

                cb1, cb2, cb3, cb4 = st.columns(4)
                with cb1: cat_sel  = st.selectbox("Categorical column", cat_cols,                  key="cat_break_col")
                with cb2: num_sel  = st.selectbox("Numerical column",   ["— Count —"] + num_cols,  key="cat_break_num")
                with cb3: cat_agg  = st.selectbox("Aggregation", ["Count","Mean","Sum","Median","Max","Min"], key="cat_agg")
                with cb4: cat_type = st.radio("Chart", ["Bar","Pie","Donut"], key="cat_break_type", horizontal=True)

                cp1, cp2, cp3, cp4 = st.columns(4)
                with cp1: top_n     = st.slider("Top N categories", 3, 30, 10, key="cat_topn")
                with cp2: sort_cats = st.radio("Sort", ["By value","Alphabetical"], key="cat_sort", horizontal=True)
                with cp3: show_pct  = st.checkbox("Show % labels", value=True, key="cat_pct")
                with cp4: cat_color = st.selectbox(
                    "Color scale", ["Blues","Viridis","Plasma","Teal","Sunset"],
                    key="cat_colorscale"
                )

          

            
            try:
                agg_map = {
                    "Count": "count", "Mean": "mean", "Sum": "sum",
                    "Median": "median", "Max": "max", "Min": "min"
                }
                if num_sel == "— Count —" or cat_agg == "Count":
                    data = filtered_df4[cat_sel].value_counts().reset_index()
                    data.columns = [cat_sel, "Value"]
                else:
                    data = (filtered_df4
                            .groupby(cat_sel)[num_sel]
                            .agg(agg_map[cat_agg]).reset_index())
                    data.columns = [cat_sel, "Value"]

                data = (data.sort_values("Value", ascending=False)
                        if sort_cats == "By value"
                        else data.sort_values(cat_sel))
                data = data.head(top_n)

                text_vals = (
                    data["Value"].apply(
                        lambda x: f"{x:.1f} ({x / data['Value'].sum() * 100:.1f}%)"
                    ) if show_pct else data["Value"]
                )

                if cat_type == "Bar":
                    fig_cb = px.bar(
                        data, x=cat_sel, y="Value",
                        template="plotly_white", height=420,
                        color="Value",
                        color_continuous_scale=cat_color,
                        text=text_vals
                    )
                    fig_cb.update_traces(textposition="outside")
                    fig_cb.update_layout(coloraxis_showscale=False)
                elif cat_type == "Pie":
                    fig_cb = px.pie(
                        data, names=cat_sel, values="Value",
                        template="plotly_white", height=420
                    )
                else:
                    fig_cb = px.pie(
                        data, names=cat_sel, values="Value",
                        hole=0.45, template="plotly_white", height=420
                    )

                fig_cb.update_layout(
                    paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fc"
                )
                st.plotly_chart(fig_cb, use_container_width=True)

            except Exception as e:
                st.error(f"Chart error: {e}")

           

    nav_buttons("Visualizations")
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
