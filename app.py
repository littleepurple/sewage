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
def read_excel_any(path_or_bytes, guess_max=200_000):
    """Excel読込（PathまたはBytesIO）。先頭シートを読み、列名をトリム。"""
    if isinstance(path_or_bytes, (str, os.PathLike)):
        df = pd.read_excel(path_or_bytes, engine="openpyxl")
    else:
        df = pd.read_excel(path_or_bytes, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def detect_col(cols, patterns):
    """列名のゆるめ検出: patterns のいずれかと正規表現マッチした最初の列を返す"""
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
    # 方式別サマリ
    g = df.groupby(process_col, dropna=False)
    summ = g.agg(
        件数=(xcol, "count"),
        X平均=(xcol, "mean"),
        X中央値=(xcol, "median"),
        Y平均=(ycol, "mean"),
        Y中央値=(ycol, "median")
    ).reset_index()
    return summ

# ========== サイドバー：入力 ==========
st.sidebar.header("① データ読み込み")
use_desktop = st.sidebar.checkbox("デスクトップ既定パスを使う", value=True)
default_path = os.path.expanduser("~/Desktop/下水処理場ガイド2025（Excel版）.xlsx")
excel_path = st.sidebar.text_input("既定パス（必要なら変更）", value=default_path)
uploaded = st.sidebar.file_uploader("またはExcelをアップロード（.xlsx）", type=["xlsx"])
load_btn = st.sidebar.button("読み込む", type="primary")

st.sidebar.markdown("---")
st.sidebar.header("② 表示設定")
use_logx = st.sidebar.checkbox("横軸を対数（log10）", value=False)
use_logy = st.sidebar.checkbox("縦軸を対数（log10）", value=False)
add_trend = st.sidebar.checkbox("トレンド線（OLS）を表示", value=False)
trend_scope = st.sidebar.radio("トレンド線の範囲", ["全体", "方式ごと"], index=0, disabled=not add_trend)

# ========== メイン ==========
st.title("水処理方式を選んで『処理人口×処理能力』の関係を見る")

df = None
col_map = {}

if load_btn:
    try:
        if use_desktop:
            path = os.path.expanduser(excel_path)
            if not os.path.exists(path):
                st.error(f"ファイルが見つかりません: {path}")
            else:
                df = read_excel_any(path)
        else:
            if uploaded is None:
                st.error("ファイルを選択してください。")
            else:
                df = read_excel_any(uploaded)

        if df is not None:
            cols = list(df.columns)

            # 列の自動検出
            # 「水処理方式現有」
            col_process = detect_col(
                cols,
                patterns=[r"水処理方式.*現有", r"水処理方式", r"処理方式", r"方式"]
            )
            # 「処理人口（人）全体計画」
            col_pop_total = detect_col(
                cols,
                patterns=[r"処理人口.*全体.*計画", r"全体.*計画.*処理人口", r"全体計画", r"人口.*計画"]
            )
            # 「処理能力（㎥/日最大）現有」
            col_cap_max = detect_col(
                cols,
                patterns=[r"処理能力.*日.*最大.*現有", r"処理能力.*最大.*現有", r"処理能力.*最大", r"処理能力"]
            )

            col_map = {
                "水処理方式現有": col_process,
                "処理人口（人）全体計画": col_pop_total,
                "処理能力（㎥/日最大）現有": col_cap_max
            }

            st.success("読み込み完了")
            with st.expander("列の自動検出結果（必要なら手動で直してください）", expanded=False):
                c1, c2, c3 = st.columns(3)
                with c1:
                    col_map["水処理方式現有"] = st.selectbox(
                        "水処理方式現有 列",
                        options=cols,
                        index=cols.index(col_process) if col_process in cols else 0
                    )
                with c2:
                    col_map["処理人口（人）全体計画"] = st.selectbox(
                        "処理人口（人）全体計画 列",
                        options=cols,
                        index=cols.index(col_pop_total) if col_pop_total in cols else 0
                    )
                with c3:
                    col_map["処理能力（㎥/日最大）現有"] = st.selectbox(
                        "処理能力（㎥/日最大）現有 列",
                        options=cols,
                        index=cols.index(col_cap_max) if col_cap_max in cols else 0
                    )

            # クレンジング
            proc_col = col_map["水処理方式現有"]
            pop_col = col_map["処理人口（人）全体計画"]
            cap_col = col_map["処理能力（㎥/日最大）現有"]

            df["_process"] = df[proc_col].astype(str).str.strip()
            df["_pop_total"] = df[pop_col].apply(to_numeric)
            df["_cap_max"] = df[cap_col].apply(to_numeric)
            df = df[["_process", "_pop_total", "_cap_max"]].rename(
                columns={"_process": "水処理方式現有",
                         "_pop_total": "処理人口（人）全体計画",
                         "_cap_max": "処理能力（㎥/日最大）現有"}
            )
            df = df.dropna(subset=["水処理方式現有", "処理人口（人）全体計画", "処理能力（㎥/日最大）現有"])

    except Exception as e:
        st.exception(e)

if df is None:
    st.info("左のサイドバーからExcelを指定して「読み込む」を押してください。")
    st.stop()

# 方式の選択
all_procs = sorted(df["水処理方式現有"].unique().tolist())
selected_procs = st.multiselect("③ 水処理方式（複数選択可・未選択=全選択）", options=all_procs, default=all_procs)
if len(selected_procs) == 0:
    filtered = df.copy()
else:
    filtered = df[df["水処理方式現有"].isin(selected_procs)].copy()

# 軸の選択
axis_map = {
    "処理人口（人）全体計画": "処理人口（人）全体計画",
    "処理能力（㎥/日最大）現有": "処理能力（㎥/日最大）現有"
}
c1, c2 = st.columns(2)
with c1:
    x_label = st.radio("④ 横軸を選択", list(axis_map.keys()), index=0, horizontal=True)
with c2:
    y_label = st.radio("⑤ 縦軸を選択", list(axis_map.keys()), index=1, horizontal=True)

if x_label == y_label:
    st.warning("横軸と縦軸は異なる項目を選んでください。")
    st.stop()

# 描画用データ
plot_df = filtered.rename(columns={x_label: "X", y_label: "Y"})
plot_df = plot_df[(plot_df["X"] > 0) & (plot_df["Y"] > 0)]

# 散布図
if plot_df.empty:
    st.warning("有効なデータがありません。列指定や方式の選択を見直してください。")
    st.stop()

st.subheader("散布図：方式ごとの「{} × {}」".format(x_label, y_label))

trend_arg = None
trend_scope_arg = "overall"
if add_trend:
    trend_arg = "ols"
    trend_scope_arg = "trace" if trend_scope == "方式ごと" else "overall"

fig = px.scatter(
    plot_df,
    x="X", y="Y",
    color="水処理方式現有",
    hover_data={"水処理方式現有": True, "X": ":,", "Y": ":,"},
    trendline=trend_arg,
    trendline_scope=trend_scope_arg,
)

fig.update_traces(marker=dict(size=10, opacity=0.8), selector=dict(mode="markers"))

# 軸スケール
if use_logx:
    fig.update_xaxes(type="log")
if use_logy:
    fig.update_yaxes(type="log")

fig.update_layout(
    xaxis_title=x_label,
    yaxis_title=y_label,
    legend_title="水処理方式",
    margin=dict(l=20, r=20, t=40, b=20),
)

st.plotly_chart(fig, use_container_width=True)

# 方式別サマリ
st.subheader("方式別サマリ")
summary_df = make_summary(plot_df.join(filtered["水処理方式現有"]), "水処理方式現有", "X", "Y")
summary_df_display = summary_df.copy()
summary_df_display["X平均"] = summary_df_display["X平均"].round(2)
summary_df_display["X中央値"] = summary_df_display["X中央値"].round(2)
summary_df_display["Y平均"] = summary_df_display["Y平均"].round(2)
summary_df_display["Y中央値"] = summary_df_display["Y中央値"].round(2)
st.dataframe(summary_df_display, use_container_width=True)

# 相関係数（全体）
corr = np.corrcoef(plot_df["X"], plot_df["Y"])[0, 1]
st.caption(f"全体相関係数（Pearson r）: **{corr:.3f}**")

# ダウンロード
st.subheader("ダウンロード")
c3, c4 = st.columns(2)
with c3:
    st.download_button(
        "フィルタ後データ（CSV, UTF-8 BOM）",
        data=bom_csv(filtered),
        file_name="filtered_data.csv",
        mime="text/csv"
    )
with c4:
    # Excelをその場で生成
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        filtered.to_excel(writer, index=False, sheet_name="filtered")
        summary_df_display.to_excel(writer, index=False, sheet_name="summary")
    st.download_button(
        "フィルタ後データ＋サマリ（Excel）",
        data=out.getvalue(),
        file_name="filtered_and_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.markdown("---")
with st.expander("補足"):
    st.write("- 列名は自動検出しますが、うまく拾えない場合は検出結果を手動で直してください。")
    st.write("- 文字列数値（カンマ・空白・ダッシュ等）は自動で数値化します。")
    st.write("- 対数軸を有効にすると、桁の異なるデータの分布比較が見やすくなります。")
