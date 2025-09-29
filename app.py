import io
import math
import chardet
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="2列カテゴリ比較アプリ", layout="wide")

st.title("任意の2列をカテゴリで比較 ✨")
st.caption("1つ目のカテゴリを選ぶと、2つ目のカテゴリ候補が自動で絞り込まれます。横軸・縦軸の可視化も対応。")

@st.cache_data(show_spinner=False)
def read_any_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
	ext = filename.lower().split(".")[-1]
	if ext in {"xls", "xlsx"}:
		return pd.read_excel(io.BytesIO(file_bytes))
	# CSV: 文字コード自動判別
	detected = chardet.detect(file_bytes or b"")
	enc = detected.get("encoding") or "utf-8"
	try:
		return pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
	except UnicodeDecodeError:
		return pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig")


def is_numeric(s: pd.Series) -> bool:
	return pd.api.types.is_numeric_dtype(s)


def is_datetime(s: pd.Series) -> bool:
	return pd.api.types.is_datetime64_any_dtype(s)


def coerce_datetimes(df: pd.DataFrame) -> pd.DataFrame:
	# 文字列っぽい日付を自動変換（失敗時はそのまま）
	df = df.copy()
	for c in df.columns:
		if df[c].dtype == object:
			try:
				df[c] = pd.to_datetime(df[c], errors="raise")
			except Exception:
				pass
	return df


def make_categorical(series: pd.Series, mode: str, bins: int, precision: int = 2) -> pd.Series:
	"""任意列をカテゴリ化。
	mode: 'auto' | 'as_is' | 'qcut' | 'cut'
	- auto: 既にカテゴリ/文字列/日時ならそのまま。数値ならqcut。
	- as_is: 変換せず（ただし日時はYYYY-MMに丸めてカテゴリ化）。
	- qcut: 分位数ビン
	- cut : 等幅ビン
	"""
	# 日時列は月単位でカテゴリ化（見やすさ重視）
	if is_datetime(series):
		return series.dt.to_period("M").astype(str)

	if mode == "as_is":
		if is_numeric(series):
			return series.astype("category")
		return series.astype(str)

	if not is_numeric(series):
		# 文字列・カテゴリはそのまま
		return series.astype(str)

	# 数値列: ビニング
	valid = series.dropna()
	if valid.empty:
		return series.astype("category")
	try:
		if mode in ("auto", "qcut"):
			binned = pd.qcut(series, q=min(max(bins, 2), 20), duplicates="drop")
		else:
			binned = pd.cut(series, bins=min(max(bins, 2), 50))
		# ラベル整形
		cats = binned.astype(str)
		return cats
	except Exception:
		# 一意値が少なく分割できない場合は丸めカテゴリ
		rounded = series.round(precision).astype(str)
		return rounded


# サイドバー: ファイル読み込み
with st.sidebar:
	st.header("データ入力")
	uploaded = st.file_uploader("CSV または Excel", type=["csv", "xlsx", "xls"])
	st.markdown("—")
	st.caption("数値列は自動でビニングしてカテゴリ化できます。")

if not uploaded:
	st.info("ファイルをサイドバーからアップロードしてください。")
	st.stop()

# 読み込み
raw_df = read_any_table(uploaded.getvalue(), uploaded.name)
df = coerce_datetimes(raw_df)

if df.empty or len(df.columns) < 2:
	st.error("有効なデータが見つかりません。")
	st.stop()

st.success(f"読み込み: {df.shape[0]} 行 × {df.shape[1]} 列")
with st.expander("データの先頭 (50行)"):
	st.dataframe(df.head(50), use_container_width=True)

# 列選択とカテゴリ化設定
col_config_1, col_config_2, misc_cfg = st.columns([1.2, 1.2, 0.8])

with col_config_1:
	st.subheader("1つ目のカテゴリ (X)")
	col_x = st.selectbox("列を選択", df.columns, index=0, key="col_x")
	mode_x = st.selectbox("カテゴリ化方法", ["auto", "as_is", "qcut", "cut"], index=0, key="mode_x",
		help="auto: 数値は分位数ビンに、文字列/日時はそのまま")
	bins_x = st.slider("ビン数 (数値列)", 2, 20, 5, key="bins_x")
	s_x = make_categorical(df[col_x], mode_x, bins_x)
	cats_x_all = pd.Index(s_x.dropna().unique()).astype(str).tolist()
	selected_cats_x = st.multiselect("カテゴリ値を選択 (複数可)", cats_x_all, default=cats_x_all[:min(10, len(cats_x_all))])

with col_config_2:
	st.subheader("2つ目のカテゴリ/数値 (Y)")
	col_y = st.selectbox("列を選択", df.columns, index=1 if len(df.columns) > 1 else 0, key="col_y")
	# Yはカテゴリ/数値の両対応
	as_category_y = st.toggle("Yをカテゴリとして扱う", value=not is_numeric(df[col_y]))
	mode_y = st.selectbox("Yのカテゴリ化方法", ["auto", "as_is", "qcut", "cut"], index=0, disabled=not as_category_y, key="mode_y")
	bins_y = st.slider("Yのビン数 (数値列)", 2, 20, 5, disabled=not as_category_y, key="bins_y")
	if as_category_y:
		s_y_full = make_categorical(df[col_y], mode_y, bins_y)
	else:
		s_y_full = df[col_y]

with misc_cfg:
	st.subheader("表示設定")
	sample_n = st.number_input("表示上限行数", min_value=100, max_value=200000, value=min(100000, len(df)), step=100)
	bar_orientation = st.selectbox("棒グラフの向き", ["vertical", "horizontal"], index=0)
	sort_desc = st.toggle("集計を降順で表示", value=True)

# 1つ目のカテゴリ選択でフィルタ
s_x = make_categorical(df[col_x], mode_x, bins_x)
mask_x = s_x.isin(selected_cats_x) if selected_cats_x else pd.Series([True] * len(df), index=df.index)
filtered_df = df[mask_x].copy()

# 2つ目の候補（自動でカテゴリ候補を絞り込み）
if as_category_y:
	s_y = make_categorical(filtered_df[col_y], mode_y, bins_y)
	cats_y_all = pd.Index(s_y.dropna().unique()).astype(str).tolist()
	default_y = cats_y_all[:min(10, len(cats_y_all))]
	selected_cats_y = st.multiselect("2つ目のカテゴリ値 (自動で絞り込み)", cats_y_all, default=default_y, key="selected_cats_y")
	mask_y = s_y.isin(selected_cats_y) if selected_cats_y else pd.Series([True] * len(filtered_df), index=filtered_df.index)
	filtered_df2 = filtered_df[mask_y]
else:
	selected_cats_y = []
	filtered_df2 = filtered_df

st.markdown("—")

# 絞り込み結果のプレビュー
st.write(f"現在の条件: X={col_x} / 選択カテゴリ数={len(selected_cats_x)} | Y={col_y}{' (カテゴリ扱い)' if as_category_y else ' (数値)'} / 行数={len(filtered_df2)}")
with st.expander("絞り込み後データ (先頭)", expanded=False):
	st.dataframe(filtered_df2.head(50), use_container_width=True)

# タブ: クロス集計 / 可視化 / 統計
	tab_ct, tab_plot, tab_stats = st.tabs(["クロス集計", "可視化", "統計/要約"]) 

with tab_ct:
	if as_category_y:
		cx = make_categorical(filtered_df2[col_x], mode_x, bins_x)
		cy = make_categorical(filtered_df2[col_y], mode_y, bins_y)
		ct = pd.crosstab(cx, cy)
		if sort_desc:
			ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]
		st.dataframe(ct, use_container_width=True)
		fig = px.imshow(ct, text_auto=True, aspect="auto", color_continuous_scale="Greens")
		st.plotly_chart(fig, use_container_width=True)
	else:
		# Yが数値のときはXカテゴリごとの集計
		cx = make_categorical(filtered_df2[col_x], mode_x, bins_x)
		agg = filtered_df2.groupby(cx, dropna=False)[col_y].mean().reset_index().rename(columns={col_y: f"{col_y} (mean)"})
		agg = agg.sort_values(agg.columns[-1], ascending=not sort_desc)
		st.dataframe(agg, use_container_width=True)

with tab_plot:
	if as_category_y:
		# カテゴリ×カテゴリ → 積み上げ/ヒートマップ/棒
		cx = make_categorical(filtered_df2[col_x], mode_x, bins_x)
		cy = make_categorical(filtered_df2[col_y], mode_y, bins_y)
		df_plot = (
			filtered_df2.assign(X=cx, Y=cy)
			.groupby(["X", "Y"]).size().reset_index(name="count")
		)
		if bar_orientation == "vertical":
			fig = px.bar(df_plot, x="X", y="count", color="Y", barmode="stack")
		else:
			fig = px.bar(df_plot, y="X", x="count", color="Y", barmode="stack", orientation="h")
		st.plotly_chart(fig, use_container_width=True)
	else:
		# カテゴリ(X) × 数値(Y) → 棒グラフ
		cx = make_categorical(filtered_df2[col_x], mode_x, bins_x)
		agg = filtered_df2.groupby(cx, dropna=False)[col_y].mean().reset_index(name=f"{col_y} (mean)")
		if bar_orientation == "vertical":
			fig = px.bar(agg, x=cx.name, y=f"{col_y} (mean)")
		else:
			fig = px.bar(agg, y=cx.name, x=f"{col_y} (mean)", orientation="h")
		st.plotly_chart(fig, use_container_width=True)

with tab_stats:
	# 要約統計
	if not as_category_y and is_numeric(filtered_df2[col_y]):
		st.subheader(f"{col_y} の要約統計 (絞り込み後)")
		st.dataframe(filtered_df2[col_y].describe().to_frame().T, use_container_width=True)
		# Xカテゴリごとの統計
		cx = make_categorical(filtered_df2[col_x], mode_x, bins_x)
		agg = filtered_df2.groupby(cx, dropna=False)[col_y].agg(["count", "mean", "median", "min", "max"]).reset_index()
		st.dataframe(agg, use_container_width=True)
	else:
		st.info("Yが数値のときに要約統計を表示します。")

# ダウンロード
dl = filtered_df2.head(int(sample_n))
st.download_button(
	label="絞り込みデータをCSVでダウンロード",
	data=dl.to_csv(index=False).encode("utf-8-sig"),
	file_name="filtered.csv",
	mime="text/csv",
)



