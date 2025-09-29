import streamlit as st
import pandas as pd
import altair as alt
import chardet
from io import StringIO, BytesIO
from typing import List, Tuple

st.set_page_config(page_title="下水処理場ガイド ビューア", layout="wide")
st.title("下水処理場ガイド2025 ビューア（全列比較対応）")

with st.sidebar:
	st.header("データ読込")
	uploaded = st.file_uploader("CSV/Excel を選択", type=["csv", "xlsx", "xlsm", "xls"])
	encoding_hint = st.selectbox("エンコーディング推定/指定（CSV時）", ["auto", "utf-8", "cp932", "shift_jis", "utf-16"], index=0)
	st.caption("ヘッダーが複数行相当のため、読み込み後に列名を自動整形します")

@st.cache_data(show_spinner=False)
def detect_encoding(sample_bytes: bytes) -> str:
	result = chardet.detect(sample_bytes)
	enc = result.get("encoding") or "utf-8"
	if enc and enc.lower() in {"shift_jis", "sjis", "ms932"}:
		return "cp932"
	return enc

@st.cache_data(show_spinner=True)
def load_dataframe(file, encoding_hint: str) -> pd.DataFrame:
	if file is None:
		return pd.DataFrame()
	name = (file.name or "").lower()
	raw = file.read()
	if name.endswith((".xlsx", ".xlsm", ".xls")):
		return pd.read_excel(BytesIO(raw), header=None, engine="openpyxl")
	encoding = detect_encoding(raw) if encoding_hint == "auto" else encoding_hint
	text = raw.decode(encoding, errors="replace")
	return pd.read_csv(StringIO(text), header=None)

@st.cache_data(show_spinner=False)
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
	if df.empty:
		return df
	# ヘッダー候補の検出を強化
	keywords = ["都道府県", "自治体コード", "処理場名", "処理能力", "処理方式"]
	header_row_idx = None
	max_search = min(100, len(df))
	for i in range(max_search):
		row_strs = df.iloc[i].astype(str).fillna("")
		joined = "\t".join(row_strs.values)
		if any(k in joined for k in keywords):
			header_row_idx = i
			break
	# キーワードで見つからなければ、情報量の多い行を採用（先頭50行）
	if header_row_idx is None:
		max_nonempty = -1
		fallback_search = min(50, len(df))
		for i in range(fallback_search):
			nonempty = (df.iloc[i].astype(str).str.strip() != "").sum()
			if nonempty > max_nonempty:
				max_nonempty = nonempty
				header_row_idx = i
	# 列名生成
	new_cols = (
		df.iloc[header_row_idx]
		.astype(str)
		.str.replace(r"\s+", " ", regex=True)
		.str.replace("\n", " ")
		.str.strip()
		.where(lambda s: s != "", other="col")
	)
	counts = {}
	fixed_cols = []
	for c in new_cols:
		base = c
		k = base
		n = 1
		while k in counts:
			n += 1
			k = f"{base}_{n}"
		counts[k] = 1
		fixed_cols.append(k)
	# 本体抽出（0行防止のフォールバックあり）
	body = df.iloc[header_row_idx + 1 :].copy()
	if body.shape[0] == 0:
		# ヘッダー行も含めて戻す（空回避）
		body = df.iloc[header_row_idx :].copy()
	# 列数調整
	if len(fixed_cols) != body.shape[1]:
		# 列数がずれた場合は切り詰め/パディング
		if len(fixed_cols) > body.shape[1]:
			fixed_cols = fixed_cols[: body.shape[1]]
		else:
			# 足りない分は連番列を追加
			extra = [f"col_extra_{i}" for i in range(1, body.shape[1] - len(fixed_cols) + 1)]
			fixed_cols = fixed_cols + extra
	body.columns = fixed_cols
	# 全空行は削除
	body = body.dropna(how="all").reset_index(drop=True)
	return body

@st.cache_data(show_spinner=False)
def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
	if df.empty:
		return df
	out = df.copy()
	for col in out.columns:
		s = (
			out[col]
			.astype(str)
			.str.replace(",", "", regex=False)
			.str.replace("㎡", "", regex=False)
			.str.replace("㎥/日最大", "", regex=False)
		)
		num = pd.to_numeric(s, errors="coerce")
		if num.notna().sum() > 0 and (num.notna().mean() > 0.6):
			out[col] = num
	return out

raw_df = load_dataframe(uploaded, encoding_hint)
if raw_df.empty:
	st.info("左のサイドバーから CSV または Excel ファイルを選択してください。")
	st.stop()

st.success(f"読み込み成功: 形状 {raw_df.shape}")

df = coerce_types(clean_columns(raw_df))
st.write("列数:", len(df.columns))

with st.expander("先頭プレビュー", expanded=True):
	st.dataframe(df.head(50), use_container_width=True)

with st.sidebar:
	st.header("フィルタ")
	selected_cols = st.multiselect("絞込対象カラム", df.columns.tolist())
	conditions = {}
	for c in selected_cols:
		series = df[c]
		if pd.api.types.is_numeric_dtype(series):
			vals = pd.to_numeric(series, errors="coerce")
			if vals.notna().any():
				min_v = float(vals.min(skipna=True))
				max_v = float(vals.max(skipna=True))
				left, right = st.slider(c, min_v, max_v, (min_v, max_v))
				conditions[c] = ("range", (left, right))
		else:
			uniq = sorted(series.dropna().astype(str).unique().tolist())[:5000]
			chosen = st.multiselect(c, uniq)
			if chosen:
				conditions[c] = ("in", chosen)

mask = pd.Series([True] * len(df))
for col, cond in conditions.items():
	kind, val = cond
	if kind == "in":
		mask &= df[col].astype(str).isin(val)
	else:
		left, right = val
		s = pd.to_numeric(df[col], errors="coerce")
		mask &= s.ge(left) & s.le(right)

fdf = df[mask].copy()

search = st.text_input("全体検索（部分一致、大小文字区別なし）")
if search:
	pat = search.lower()
	fdf = fdf[fdf.astype(str).apply(lambda row: pat in " ".join(map(str, row.values)).lower(), axis=1)]

st.metric("ヒット件数", len(fdf))

with st.sidebar:
	st.header("並び替え")
	sort_col = st.selectbox("ソート列", [None] + fdf.columns.tolist(), index=0)
	ascending = st.toggle("昇順", value=True)
	if sort_col:
		fdf = fdf.sort_values(by=sort_col, ascending=ascending)

# ========== 全列比較 ==========
st.subheader("全列比較")
compare_all = st.toggle("全列を比較対象にする", value=True, help="オフにすると下で列を選択できます")
if compare_all:
	compare_cols = list(fdf.columns)
else:
	compare_cols = st.multiselect("比較対象列を選ぶ（任意件数）", fdf.columns.tolist(), default=list(fdf.columns)[:10])

num_cols_all = [c for c in compare_cols if pd.api.types.is_numeric_dtype(fdf[c])]
cat_cols_all = [c for c in compare_cols if not pd.api.types.is_numeric_dtype(fdf[c])]

tabs = st.tabs(["相関ヒートマップ(数値)", "散布行列(数値)", "クロス集計(カテゴリ×カテゴリ)", "グループ統計(カテゴリ→数値)"])

# 相関ヒートマップ
with tabs[0]:
	if len(num_cols_all) >= 2:
		corr = fdf[num_cols_all].corr(numeric_only=True)
		corr_melt = corr.reset_index().melt(id_vars=corr.index.name or "index")
		corr_melt.columns = ["X", "Y", "相関"]
		chart = alt.Chart(corr_melt).mark_rect().encode(
			x=alt.X("X:O", sort=None),
			y=alt.Y("Y:O", sort=None),
			color=alt.Color("相関:Q", scale=alt.Scale(scheme="blueorange", domain=[-1, 1])),
			tooltip=["X", "Y", alt.Tooltip("相関:Q", format=".3f")],
		).properties(height=500)
		st.altair_chart(chart, use_container_width=True)
	else:
		st.info("数値列が2列以上必要です")

# 散布行列（多すぎると重いので最大10列を推奨）
with tabs[1]:
	max_matrix = st.number_input("最大列数(散布行列)", min_value=2, max_value=20, value=min(10, max(2, len(num_cols_all))))
	num_for_matrix = num_cols_all[: int(max_matrix)]
	if len(num_for_matrix) >= 2:
		base = alt.Chart(fdf.dropna(subset=num_for_matrix)).properties(width=120, height=120)
		scatter = base.mark_circle(size=20, opacity=0.5).encode(
			x=alt.X(alt.repeat("column"), type="quantitative"),
			y=alt.Y(alt.repeat("row"), type="quantitative"),
			tooltip=[alt.Tooltip(c, type="quantitative") for c in num_for_matrix],
		)
		chart = scatter.repeat(row=num_for_matrix, column=num_for_matrix)
		st.altair_chart(chart, use_container_width=True)
	else:
		st.info("数値列が2列以上必要です")

# クロス集計（カテゴリ×カテゴリ）
with tabs[2]:
	if len(cat_cols_all) >= 2:
		c1 = st.selectbox("カテゴリ1", cat_cols_all, key="crosstab_c1")
		c2 = st.selectbox("カテゴリ2", cat_cols_all, index=min(1, len(cat_cols_all)-1), key="crosstab_c2")
		limit = st.slider("各カテゴリの上位件数(頻度順)", 2, 50, 20)
		t1 = fdf[c1].astype(str).value_counts().head(limit).index
		t2 = fdf[c2].astype(str).value_counts().head(limit).index
		pivot = pd.crosstab(fdf[c1].astype(str).where(lambda s: s.isin(t1), other="その他"),
							fdf[c2].astype(str).where(lambda s: s.isin(t2), other="その他"))
		st.dataframe(pivot)
		melt = pivot.reset_index().melt(id_vars=c1)
		melt.columns = [c1, c2, "件数"]
		chart = alt.Chart(melt).mark_rect().encode(
			x=alt.X(f"{c2}:O", sort="-y"),
			y=alt.Y(f"{c1}:O", sort="-x"),
			color=alt.Color("件数:Q", scale=alt.Scale(scheme="blues")),
			tooltip=[c1, c2, "件数"],
		).properties(height=500)
		st.altair_chart(chart, use_container_width=True)
	else:
		st.info("カテゴリ列が2列以上必要です")

# グループ統計（カテゴリ→数値）
with tabs[3]:
	if len(cat_cols_all) >= 1 and len(num_cols_all) >= 1:
		gc = st.selectbox("カテゴリ列", cat_cols_all, key="group_cat")
		metrics = st.multiselect("数値列（統計対象）", num_cols_all, default=num_cols_all[:min(5, len(num_cols_all))])
		aggs = st.multiselect("集計関数", ["count", "mean", "median", "min", "max", "std", "sum"], default=["count", "mean", "median"])
		if metrics and aggs:
			g = fdf.groupby(gc)[metrics].agg(aggs)
			st.dataframe(g)
			# 選択1指標を棒グラフ
			col_metric = st.selectbox("可視化する列", metrics)
			col_agg = st.selectbox("可視化する集計", aggs)
			plot_df = g[col_metric][col_agg].reset_index().sort_values(by=col_agg, ascending=False)
			bar = alt.Chart(plot_df).mark_bar().encode(x=alt.X(gc, sort='-y'), y=alt.Y(col_agg, title=f"{col_metric} ({col_agg})"))
			st.altair_chart(bar.properties(height=400), use_container_width=True)
	else:
		st.info("カテゴリ列と数値列が必要です")

# ========== 表示 ==========
st.subheader("データ表示")
st.dataframe(fdf, use_container_width=True)

# ========== ドリルダウン ==========
st.subheader("インタラクティブ・ドリルダウン（数値→カテゴリ）")
num_cols = [c for c in fdf.columns if pd.api.types.is_numeric_dtype(fdf[c])]
cat_cols = [c for c in fdf.columns if not pd.api.types.is_numeric_dtype(fdf[c])]

# デフォルト推定
default_num = None
for key in ["処理人口", "人口", "処理能力", "面積"]:
	for c in fdf.columns:
		if key in str(c):
			default_num = c
			break
	if default_num:
		break
if not default_num and num_cols:
	default_num = num_cols[0]

default_cat = None
for key in ["処理方式", "水処理方式", "方式", "放流先"]:
	for c in fdf.columns:
		if key in str(c):
			default_cat = c
			break
	if default_cat:
		break
if not default_cat and cat_cols:
	default_cat = cat_cols[0]

col_a, col_b = st.columns(2)
with col_a:
	num_col = st.selectbox("数値列（例: 処理人口）", num_cols, index=(num_cols.index(default_num) if default_num in num_cols else 0) if num_cols else None)
with col_b:
	cat_col = st.selectbox("カテゴリ列（例: 処理方式）", cat_cols, index=(cat_cols.index(default_cat) if default_cat in cat_cols else 0) if cat_cols else None)

if num_col and cat_col:
	df2 = fdf.dropna(subset=[num_col, cat_col]).copy()
	try:
		import altair as alt
		brush = alt.selection_interval(encodings=['x'])
		hist = alt.Chart(df2).mark_bar().encode(
			x=alt.X(f"{num_col}:Q", bin=alt.Bin(maxbins=30), title=num_col),
			y=alt.Y('count()', title='件数')
		).properties(height=150)
		bars = alt.Chart(df2).mark_bar().encode(
			x=alt.X(f"{cat_col}:N", sort='-y', title=cat_col),
			y=alt.Y('count()', title='件数'),
			tooltip=[cat_col, alt.Tooltip('count()', title='件数')]
		).transform_filter(brush)
		st.altair_chart((hist.add_params(brush) & bars).resolve_scale(y='independent'), use_container_width=True)
	except Exception:
		st.warning("可視化でエラーが発生しました。列を変更してお試しください。")
	# 集計テーブル
	g = df2.groupby(pd.cut(pd.to_numeric(df2[num_col], errors='coerce'), bins=10))[cat_col].value_counts().rename('件数').reset_index()
	st.dataframe(g, use_container_width=True)
else:
	st.info("数値列とカテゴリ列を選択してください")

# ========== 図表(自由作図) ==========
st.subheader("図表作成")
num_cols = [c for c in fdf.columns if pd.api.types.is_numeric_dtype(fdf[c])]
cat_cols = [c for c in fdf.columns if not pd.api.types.is_numeric_dtype(fdf[c])]
chart_type = st.selectbox("チャート種別", ["棒", "折れ線", "散布図", "箱ひげ"])
x_col = st.selectbox("X軸 (カテゴリor数値)", [None] + cat_cols + num_cols)
y_col = st.selectbox("Y軸 (数値)", [None] + num_cols)
color_col = st.selectbox("色分け", [None] + cat_cols + num_cols)

if x_col and y_col:
	base = alt.Chart(fdf.dropna(subset=[x_col, y_col])).encode(x=x_col, y=y_col)
	if color_col:
		base = base.encode(color=color_col)
	if chart_type == "棒":
		ch = base.mark_bar()
	elif chart_type == "折れ線":
		ch = base.mark_line(point=True)
	elif chart_type == "散布図":
		ch = base.mark_circle(size=60)
	else:
		ch = alt.Chart(fdf).mark_boxplot().encode(x=x_col, y=y_col, color=color_col if color_col else alt.value("steelblue"))
	st.altair_chart(ch.properties(height=400), use_container_width=True)
else:
	st.info("X軸とY軸を選択するとチャートが表示されます")

# ========== ダウンロード ==========
st.subheader("ダウンロード")
st.download_button(
	"現在の絞込結果をCSVで保存",
	data=fdf.to_csv(index=False).encode("utf-8-sig"),
	file_name="filtered.csv",
	mime="text/csv",
)




