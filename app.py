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

# ===== ユーティリティ =====
def detect_col(cols, patterns):
    for pat in patterns:
        for c in cols:
            if re.search(pat, c):
                return c
    return None

def to_numeric(x):
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
    return df.to_csv(index=False).encode("utf-8-sig")

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

# ===== 0) Excelファイルをアップロード =====
st.sidebar.header("データ読み込み")
uploaded = st.sidebar.file_uploader("Excelファイルをアップロード（.xlsx）", type=["xlsx"])

st.title("①〜④構成：方式別で『処理人口 × 処理能力』を見る")

if uploaded is None:
    st.info("サイドバーにExcelファイルをドラッグ＆ドロップしてください。")
    st.stop()

# ===== データ読込 =====
df = pd.read_excel(uploaded, engine="openpyxl")
df.columns = [str(c).strip() for c in df.columns]
cols = list(df.columns)

col_process = detect_col(cols, [r"水処理方式.*現有", r"水処理方式", r"処理方式", r"方式"])
col_pop_total = detect_col(cols, [r"処理人口.*全体.*計画", r"全体.*計画.*処理人口", r"全体計画", r"人口.*計画"])
col_cap_max = detect_col(cols, [r"処理能力.*日.*最大.*現有", r"処理能力.*最大.*現有", r"処理能力.*最大", r"処理能力"])

# 必要列だけ抽出
work = pd.DataFrame({
    "水処理方式現有": df[col_process].astype(str).str.strip(),
    "処理人口（人）全体計画": df[col_pop_total].apply(to_numeric),
    "処理能力（㎥/日最大）現有": df[col_cap_max].apply(to_numeric),
})
work = work.dropna()

# ===== ① 方式選択（初期=全選択） =====
st.markdown("## ① 「水処理方式現有」を選択")
all_procs = sorted(work["水処理方式現有"].unique().tolist())
selected_procs = st.multiselect("方式（複数選択可）", options=all_procs, default=all_procs)
filtered = work if len(selected_procs) == 0 else work[work["水処理方式現有"].isin(selected_procs)].copy()

# ===== ② 比較列選択 =====
st.markdown("## ② 比較したい列を選択")
c1, c2 = st.columns(2)
with c1:
    x_label = st.radio("横軸", ["処理人口（人）全体計画", "処理能力（㎥/日最大）現有"], index=0, horizontal=True)
with c2:
    y_label = st.radio("縦軸", ["処理人口（人）全体計画", "処理能力（㎥/日最大）現有"], index=1, horizontal=True)

if x_label == y_label:
    st.warning("横軸と縦軸は異なる項目を選んでください。")
    st.stop()

logx = st.checkbox("横軸を対数（log10）", value=False)
logy = st.checkbox("縦軸を対数（log10）", value=False)
trend = st.checkbox("トレンド線（OLS）", value=False)

# ===== ③ 図・評価 =====
st.markdown("## ③ 図・評価")

plot_df = filtered.rename(columns={x_label: "X", y_label: "Y"})[["水処理方式現有", "X", "Y"]].copy()
plot_df = plot_df[(plot_df["X"] > 0) & (plot_df["Y"] > 0)]

trend_arg = "ols" if trend else None
fig = px.scatter(
    plot_df, x="X", y="Y", color="水処理方式現有",
    hover_data={"水処理方式現有": True, "X": ":,", "Y": ":,"},
    trendline=trend_arg, trendline_scope="trace"
)
fig.update_traces(marker=dict(size=10, opacity=0.85))
if logx: fig.update_xaxes(type="log")
if logy: fig.update_yaxes(type="log")
fig.update_layout(xaxis_title=x_label, yaxis_title=y_label, legend_title="水処理方式")
st.plotly_chart(fig, use_container_width=True)

st.subheader("方式別サマリ")
summary_df = make_summary(plot_df, "水処理方式現有", "X", "Y")
for c in ["X平均", "X中央値", "Y平均", "Y中央値"]:
    summary_df[c] = summary_df[c].round(2)
st.dataframe(summary_df, use_container_width=True)

corr = np.corrcoef(plot_df["X"], plot_df["Y"])[0, 1]
st.caption(f"全体相関係数（Pearson r）: **{corr:.3f}**")

# ===== ④ 詳細データ =====
st.markdown("## ④ 詳細データ")
st.dataframe(filtered.reset_index(drop=True), use_container_width=True, height=360)

c3, c4 = st.columns(2)
with c3:
    st.download_button("CSVで保存", data=bom_csv(filtered),
                       file_name="filtered_data.csv", mime="text/csv")
with c4:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        filtered.to_excel(writer, index=False, sheet_name="filtered")
        summary_df.to_excel(writer, index=False, sheet_name="summary")
    st.download_button("Excelで保存", data=out.getvalue(),
                       file_name="filtered_and_summary.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
