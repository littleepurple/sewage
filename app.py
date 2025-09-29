import io
import re
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="下水処理場ガイド分析（処理人口→処理方法）", layout="wide")
st.title("下水処理場ガイド分析（処理人口 → 処理方法）")

DEFAULT_PATH = r"c:\Users\z7718\Desktop\下水処理場ガイド2025（Excel版）.csv"

# 入力ソース
left_src, right_src = st.columns([1,2])
use_default = left_src.toggle("ローカルCSVの既定パスを使う", value=True)
uploaded = None
if not use_default:
    uploaded = right_src.file_uploader("CSVファイルをアップロード（4行ヘッダー想定）", type=["csv"])

# CSV読込（4行ヘッダー）
def read_multiheader_csv(raw_bytes: bytes | None, path: str | None) -> pd.DataFrame:
    if path:
        for enc in ["cp932", "utf-8-sig", "utf-8"]:
            try:
                return pd.read_csv(path, header=[0,1,2,3], encoding=enc)
            except Exception:
                continue
        raise ValueError("既定パスのCSV読込に失敗しました。文字コードをご確認ください。")
    if raw_bytes:
        for enc in ["cp932", "utf-8-sig", "utf-8"]:
            try:
                return pd.read_csv(io.BytesIO(raw_bytes), header=[0,1,2,3], encoding=enc)
            except Exception:
                continue
        raise ValueError("アップロードCSVの読込に失敗しました。文字コードをご確認ください。")
    raise ValueError("CSV入力がありません。")

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    def join_levels(col_tuple):
        parts = [str(x).strip() for x in col_tuple if (isinstance(x, str) and x.strip())]
        return " / ".join(parts) if parts else ""
    df = df.copy()
    df.columns = [join_levels(col) for col in df.columns]
    return df

def find_col(cols: list[str], key_word: str, sub_word: str | None=None) -> str | None:
    # 優先順位: 完全一致に近いもの → サブ語含むもの
    candidates = [c for c in cols if key_word in c]
    if sub_word:
        # サブ語を含む候補を優先
        sub_first = [c for c in candidates if sub_word in c]
        if sub_first:
            # 現有, 事業計画, 全体計画の優先順
            priority = ["現有", "直近", "全体"]
            sub_first.sort(key=lambda c: min([priority.index(p) for p in priority if p in c] + [9]))
            return sub_first[0]
    return candidates[0] if candidates else None

# 読込
try:
    if use_default and Path(DEFAULT_PATH).exists():
        df_raw = read_multiheader_csv(None, DEFAULT_PATH)
    else:
        if uploaded is None:
            st.info("CSVをアップロードするか、既定パスの読込をオンにしてください。")
            st.stop()
        df_raw = read_multiheader_csv(uploaded.read(), None)
except Exception as e:
    st.error(str(e))
    st.stop()

# 列名フラット化
df = flatten_columns(df_raw)

st.subheader("データプレビュー")
st.dataframe(df.head(50), use_container_width=True)

# 対象列の自動検出
population_col = find_col(df.columns.tolist(), "処理人口（人）", "現有")
method_col = find_col(df.columns.tolist(), "水処理方式", "現有")

with st.expander("列の確認・変更", expanded=True):
    c1, c2 = st.columns(2)
    population_col = c1.selectbox("処理人口の列", options=df.columns, index=(df.columns.tolist().index(population_col) if population_col in df.columns else 0))
    method_col = c2.selectbox("処理方法の列", options=df.columns, index=(df.columns.tolist().index(method_col) if method_col in df.columns else 0))

# クレンジング（人口は数値化、方法は文字列）
work = df.copy()

def clean_number_series(s: pd.Series) -> pd.Series:
    # 例: "232,900 " → 232900
    return (
        s.astype(str)
         .str.replace(r"[,\s]", "", regex=True)
         .str.replace("－", "-", regex=False)
         .replace({"": None, "nan": None})
         .astype(float)
    )

work[population_col] = clean_number_series(work[population_col])
work[method_col] = work[method_col].astype(str).str.strip()

# 人口候補の整形
st.subheader("処理人口で選択")
col_l, col_r = st.columns([1.2, 2.8])

# 人口のフィルタUI
with col_l:
    min_pop = int(pd.to_numeric(work[population_col], errors="coerce").min(skipna=True) or 0)
    max_pop = int(pd.to_numeric(work[population_col], errors="coerce").max(skipna=True) or 0)
    rng = st.slider("人口レンジ", min_value=min_pop, max_value=max_pop, value=(min_pop, max_pop), step=1)
    # クリック選択用テーブル（代表値リスト）
    show_unique = st.toggle("ユニーク人口を一覧してクリック選択", value=False)

# データフィルタ
filtered = work[(work[population_col] >= rng[0]) & (work[population_col] <= rng[1])]

with col_r:
    st.markdown("選択結果のサマリ")
    st.write({
        "件数": int(len(filtered)),
        "人口の最小-最大": f"{int(filtered[population_col].min(skipna=True)) if len(filtered)>0 else '-'} - {int(filtered[population_col].max(skipna=True)) if len(filtered)>0 else '-'}",
    })

if show_unique:
    # ユニーク人口を一覧 → 行クリックでさらに絞り込み
    st.markdown("ユニーク人口（上位1000）")
    unique_vals = (
        filtered[[population_col]]
        .dropna()
        .drop_duplicates()
        .sort_values(by=population_col)
        .head(1000)
        .reset_index(drop=True)
    )
    sel = st.data_editor(unique_vals, use_container_width=True, height=300, key="pop_table", num_rows="fixed")
    selected_rows = sel.index.tolist()  # data_editorは選択状態を取得しづらいので、代わりに選択値入力欄を提供
    chosen = st.multiselect("人口をクリックの代わりにここで選択", options=unique_vals[population_col].tolist(), max_selections=20)
    if chosen:
        filtered = filtered[filtered[population_col].isin(chosen)]

# フィルタ後テーブル
st.subheader("フィルタ後のデータ")
st.dataframe(filtered[[population_col, method_col]].join(
    work[[c for c in work.columns if c not in [population_col, method_col]]].head(0)  # 位置合わせのみ
), use_container_width=True, height=300)

# 可視化
st.subheader("可視化")
tab1, tab2, tab3 = st.tabs(["方法の分布", "人口×方法（ヒートマップ）", "人口別の方法内訳（積上げ）"])

with tab1:
    counts = filtered[method_col].value_counts(dropna=True).reset_index()
    counts.columns = [method_col, "count"]
    if counts.empty:
        st.info("データがありません。")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig_bar = px.bar(counts, x=method_col, y="count", text="count", title="処理方法の件数")
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)
        with c2:
            fig_pie = px.pie(counts, names=method_col, values="count", title="処理方法の構成比", hole=0.3)
            st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    pivot = (
        filtered
        .groupby([population_col, method_col], dropna=False)
        .size()
        .reset_index(name="count")
        .pivot(index=population_col, columns=method_col, values="count")
        .fillna(0)
        .sort_index()
    )
    if pivot.empty:
        st.info("データがありません。")
    else:
        fig_heat = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues", title="人口×方法の件数（ヒートマップ）")
        st.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    grp = (
        filtered
        .groupby([population_col, method_col], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values([population_col, method_col])
    )
    if grp.empty:
        st.info("データがありません。")
    else:
        fig_stack = px.bar(grp, x=population_col, y="count", color=method_col, title="人口別の方法内訳（積み上げ）")
        st.plotly_chart(fig_stack, use_container_width=True)

# クイック洞察
st.subheader("クイック洞察")
if len(filtered) > 0:
    top_method = filtered[method_col].value_counts().idxmax()
    st.write(f"選択条件で最も多い処理方法: {top_method}")
    top_by_pop = (
        filtered.groupby(population_col)[method_col]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index(name="最頻の処理方法")
        .sort_values(population_col)
    )
    st.dataframe(top_by_pop, use_container_width=True, height=300)
else:
    st.info("該当データがありません。")




