# app.py
# ----------------------------------------------------------
# 依存: streamlit, pandas, numpy, plotly, openpyxl
# pip install streamlit pandas numpy plotly openpyxl
# ----------------------------------------------------------
import os
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="方式別 人口×能力 可視化", layout="wide")

# ========== ユーティリティ ==========
@st.cache_data(show_spinner=False)
def read_excel_any(path_or_bytes):
    """Excel読込（PathまたはBytesIO）。先頭シートを読み、列名をトリム。"""
    df = pd.read_excel(path_or_bytes, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def detect_col(cols, patterns):
    """列名のゆるめ検出: patterns(正規表現)のどれかに最初にマッチした列を返す"""
    for pat in patterns:
        for c in cols:
            if re.search(pat, c):
                return c
    return None

def to_numeric(x):
    """カンマ・空白・ダッシュ類を除去して数値化"""
    if pd.isna(x):
        return np.nan
    s = str(x)
    s = re.sub(r"[,\s]", "", s)
    s = s.replace("―", "").replace("—", "").replace("-", "")
    try:
        return float(s)
    except ValueError:
        return np.nan

def bom_csv(df: pd.DataFrame) -> bytes:
    """Excelで文字化けしないBOM付きCSVをbytesで返す"""
    return df.to_csv(index=False).encode("utf-8-sig")

@st.cache_data(show_spinner=False)
def make_summary(df, process_col, xcol, ycol):
    g = df.groupby(process_col, dropna=False)
    summ = g.agg(
        件数=(xcol, "count"),
        X平均=(xcol, "mean"),
        X中央値=(xcol, "median"),
        Y平均=(ycol, "mean"),
        Y中央値=(ycol, "median")
    ).reset_index()
    return summ

# ========== 0) データ読み込み ==========
st.sidebar.header("データ読み込み")
use_desktop = st.sidebar.checkbox("デスクトップ既定パスを使う", value=True)
default_path = os.path.expanduser("~/Desktop/下水処理場ガイド2025（Excel版）.xlsx")
excel_path = st.sidebar.text_input("既定パス（必要なら変更）", value=default_path)
uploaded = st.sidebar.file_uploader("またはExcelをアップロード（.xlsx）", type=["xlsx"])
load_btn = st.sidebar.button("読み込む", type="primary")

df, col_map = None, {}

if load_btn:
    try:
        if use_desktop:
            path = os.path.expanduser(excel_path)
            if not os.path.exists(path):
                st.sidebar.error(f"ファイルが見つかりません: {path}")
            else:
                df = read_excel_any(path)
        else:
            if uploaded is None:
                st.sidebar.error("ファイルを選択してください。")
            else:
                df = read_excel_any(uploaded)
    except Exception as e:
        st.sidebar.exception(e)

st.title("①〜④構成：方式別で『処理人口 × 処理能力』を見る")

if df is None:
    st.info("左のサイドバーでExcelを指定して「読み込む」を押してください。")
    st.stop()

# ========== 列の自動検出 & 手動補正 ==========
cols = list(df.columns)
col_process = detect_col(cols, [r"水処理方式.*現有", r"水処理方式", r"処理方式", r"方式"])
col_pop_total = detect_col(cols, [r"処理人口.*全体.*計画", r"全体.*計画.*処理人口", r"全体計画", r"人口.*計画"])
col_cap_max = detect_col(cols, [r"処理能力.*日.*最大.*現有", r"処理能力.*最大.*現有", r"処理能力.*最大", r"処理能力"])

with st.expander("列の自動検出結果（必要なら手動で修正）", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        col_process = st.selectbox("水処理方式現有 列", options=cols,
                                   index=cols.index(col_process) if col_process in cols else 0)
    with c2:
        col_pop_total = st.selectbox("処理人口（人）全体計画 列", options=cols,
                                     index=cols.index(col_pop_total) if col_pop_total in cols else 0)
    with c3:
        col_cap_max = st.selectbox("処理能力（㎥/日最大）現有 列", options=cols,
                                   index=cols.index(col_cap_max) if col_cap_max in cols el
