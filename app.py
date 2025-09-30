# app.py
# ----------------------------------------------------------
# 依存: streamlit, pandas, numpy, plotly, openpyxl
# pip install streamlit pandas numpy plotly openpyxl
# ----------------------------------------------------------
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="方式別 人口×能力 可視化", layout="wide")

# ========= ユーティリティ =========
def normalize_cols(cols):
    """列名をトリム＆重複を一意化（A, A -> A, A.1...）"""
    cols = [str(c).strip() for c in cols]
    seen = {}
    out = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
    return out

def detect_col(cols, patterns):
    """正規表現 patterns のどれかに最初にマッチした列名を返す。なければ None。"""
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
    """Excelで文字化けしないBOM付きCSV"""
    return df.to_csv(index=False).encode("utf-8-sig")

def make_summary(df, process_col, xcol, ycol):
    g = df.groupby(process_col, dropna=False)
    summ = g.agg(
        件数=(xcol, "count"),
        X平均=(xcol, "mean"),
        X中央値=(xcol, "median"),
        Y平均=(ycol, "mean"),
        Y中央値=(ycol, "median"),
    ).reset_index()
    return summ

@st.cache_data(show_spinner=False)
def read_excel_with_header(file_like, sheet_name=None, header_row_zero=0):
    """指定シート＆ヘッダー行で読み込み。file_like は BytesIO 等。"""
    df = pd.read_excel(file_like, sheet_name=sheet_name, header=header_row_zero, engine="openpyxl")
    if isinstance(df, dict):  # sheet_name=None のとき複数返る
        first = list(df.keys())[0]
        df = df[first]
    df.columns = normalize_cols(df.columns)
    return df

# ========= 0) Excelアップロード（ドラッグ＆ドロップ） =========
st.sidebar.header("データ読み込み")
uploaded = st.sidebar.file_uploader("Excelファイル（.xlsx）をドラッグ＆ドロップ", type=["xlsx"])

if uploaded is None:
    st.title("①〜④構成：方式別で『処理人口 × 処理能力』を見る")
    st.info("左サイドバーにExcelをドラッグ＆ドロップしてください。")
    st.stop()

# Bytes を一度だけ取得し、以降は BytesIO を都度生成（読み取り位置問題を防止）
data_bytes = uploaded.getvalue()

# シート名一覧
with pd.ExcelFile(io.BytesIO(data_bytes), engine="openpyxl") as xls:
    sheets = xls.sheet_names

sheet_name = st.sidebar.selectbox("シートを選択", options=sheets, index=0)
header_row_1based = st.sidebar.number_input("ヘッダー行（1 = 1行目）", min_value=1, value=1, step=1)
header_row_zero = int(header_row_1based - 1)

# 読み込み
df = read_excel_with_header(io.BytesIO(data_bytes), sheet_name=sheet_name, header_row_zero=header_row_zero)
cols = list(df.columns)

st.title("①〜④構成：方式別で『処理人口 × 処理能力』を見る")
with st.expander("データ先頭プレビュー（列名を確認）", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

# ========= 列の自動検出＋必須の手動選択 =========
auto_process = detect_col(
    cols, [r"水処理方式.*(現有|現況|既存|現在)", r"水処理方式", r"処理方式", r"方式"]
)
auto_pop = detect_col(
    cols, [r"処理人口.*(全体|総).*(計画|予定)", r"(全体|総).*(計画|予定).*(処理人口)", r"処理人口.*計画", r"人口.*計画"]
)
auto_cap = detect_col(
    cols, [r"処理能力.*(㎥|m3|m³).*/?日.*(最大)?.*(現有|現況|既存|現在)", r"処理能力.*最大", r"処理能力"]
)

def pick_col_ui(label, auto_guess, options):
    opts = ["-- 列を選択 --"] + options
    idx = 0
    if auto_guess in options:
        idx = options.index(auto_guess) + 1
    return st.selectbox(label, options=opts, index=idx)

st.subheader("列の指定（自動検出の結果を確認して、必要なら変更）")
c1, c2, c3 = st.columns(3)
with c1:
    sel_process = pick_col_ui("水処理方式現有 列", auto_process, cols)
with c2:
    sel_pop = pick_col_ui("処理人口（人）全体計画 列", auto_pop, cols)
with c3:
    sel_cap = pick_col_ui("処理能力（㎥/日最大）現有 列", auto_cap, cols)

missing_labels = []
if sel_process.startswith("--"): missing_labels.append("水処理方式現有")
if sel_pop.startswith("--"):     missing_labels.append("処理人口（人）全体計画")
if sel_cap.startswith("--"):     missing_labels.append("処理能力（㎥/日最大）現有")
if missing_labels:
    st.error("次の列を指定してください： " + "、".join(missing_labels))
    st.stop()

# ========= クレンジング：元データ保持＋ヘルパー列 =========
df_all = df.copy()  # 全列保持（④で表示・DL用）
df_all["_process"]   = df_all[sel_process].astype(str).str.strip()
df_all["_pop_total"] = df_all[sel_pop].apply(to_numeric)
df_all["_cap_max"]   = df_all[sel_cap].apply(to_numeric)

valid_mask = (
    df_all["_process"].notna() & (df_all["_process"] != "") &
    df_all["_pop_total"].notna() &
    df_all["_cap_max"].notna()
)

# ========= ① 方式選択（初期=全選択） =========
st.markdown("## ① 「水処理方式現有」を選択")
all_procs = sorted(df_all.loc[valid_mask, "_process"].unique().tolist())
selected_procs = st.multiselect("方式（複数選択可）", options=all_procs, default=all_procs)

mask = valid_mask if len(selected_procs) == 0 else (valid_mask & df_all["_process"].isin(selected_procs))
filtered_full = df_all.loc[mask].copy()  # ← 全列保持のフィルタ済みデータ

if filtered_full.empty:
    st.warning("選択された方式に該当するデータがありません。方式または列指定を見直してください。")
    st.stop()

# ========= ② 比較したい列を選択 =========
st.markdown("## ② 比較したい列を選択")
cc1, cc2 = st.columns(2)
with cc1:
    x_label = st.radio("横軸", ["処理人口（人）全体計画", "処理能力（㎥/日最大）現有"], index=0, horizontal=True)
with cc2:
    y_label = st.radio("縦軸", ["処理人口（人）全体計画", "処理能力（㎥/日最大）現有"], index=1, horizontal=True)

if x_label == y_label:
    st.warning("横軸と縦軸は異なる項目を選んでください。")
    st.stop()

opt1, opt2, opt3 = st.columns(3)
with opt1:
    logx = st.checkbox("横軸を対数（log10）", value=False)
with opt2:
    logy = st.checkbox("縦軸を対数（log10）", value=False)
with opt3:
    trend = st.checkbox("トレンド線（OLS）", value=False)

# ========= ③ 図・評価 =========
st.markdown("## ③ 図・評価")
# 描画用の最小データ（方式＋数値2列）
plot_df = filtered_full.rename(
    columns={"_process": "水処理方式現有", "_pop_total": "処理人口（人）全体計画", "_cap_max": "処理能力（㎥/日最大）現有"}
)[["水処理方式現有", "処理人口（人）全体計画", "処理能力（㎥/日最大）現有"]].copy()

plot_df = plot_df.rename(columns={x_label: "X", y_label: "Y"})[["水処理方式現有", "X", "Y"]]
plot_df = plot_df[(plot_df["X"] > 0) & (plot_df["Y"] > 0)]
if plot_df.empty:
    st.warning("有効なデータがありません。列指定や方式の選択を見直してください。")
    st.stop()

trend_arg = "ols" if trend else None
fig = px.scatter(
    plot_df,
    x="X", y="Y",
    color="水処理方式現有",
    hover_data={"水処理方式現有": True, "X": ":,", "Y": ":,"},
    trendline=trend_arg,
    trendline_scope="trace",  # 方式ごと
)
fig.update_traces(marker=dict(size=10, opacity=0.85))
if logx:
    fig.update_xaxes(type="log")
if logy:
    fig.update_yaxes(type="log")
fig.update_layout(
    xaxis_title=x_label,
    yaxis_title=y_label,
    legend_title="水処理方式",
    margin=dict(l=15, r=15, t=30, b=15),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("方式別サマリ")
summary_df = make_summary(plot_df, "水処理方式現有", "X", "Y")
for c in ["X平均", "X中央値", "Y平均", "Y中央値"]:
    summary_df[c] = summary_df[c].round(2)
st.dataframe(summary_df, use_container_width=True)

corr = np.corrcoef(plot_df["X"], plot_df["Y"])[0, 1]
st.caption(f"全体相関係数（Pearson r）: **{corr:.3f}**")

# ========= ④ 詳細データ（全列） =========
st.markdown("## ④ 詳細データ（全列）")
detail_df = filtered_full.drop(columns=["_process", "_pop_total", "_cap_max"], errors="ignore")
st.dataframe(detail_df.reset_index(drop=True), use_container_width=True, height=380)

c3, c4 = st.columns(2)
with c3:
    st.download_button(
        "CSVで保存（全列・BOM付き）",
        data=bom_csv(detail_df),
        file_name="filtered_full_columns.csv",
        mime="text/csv",
    )
with c4:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        detail_df.to_excel(writer, index=False, sheet_name="filtered_full_columns")
        summary_df.to_excel(writer, index=False, sheet_name="summary_by_process")
    st.download_button(
        "Excelで保存（全列＋サマリ）",
        data=out.getvalue(),
        file_name="filtered_full_columns_and_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
