# ========== ③ 図・評価 =========
st.markdown("## ③ 図・評価")

# 最小カラムだけ抽出（方式 + 数値2列）
plot_df = filtered_full.rename(
    columns={
        "_process": "水処理方式現有",
        "_pop_total": "処理人口（人）全体計画",
        "_cap_max": "処理能力（㎥/日最大）現有",
    }
)[["水処理方式現有", "処理人口（人）全体計画", "処理能力（㎥/日最大）現有"]].copy()

# X/Y にリネーム（既存の "X"/"Y" があっても無視して最小列に限定）
plot_df = plot_df.rename(columns={x_label: "X", y_label: "Y"})[["水処理方式現有", "X", "Y"]]

# 列名の重複を排除（保険）
plot_df = plot_df.loc[:, ~plot_df.columns.duplicated()].copy()

# 数値化（保険）
plot_df["X"] = pd.to_numeric(plot_df["X"], errors="coerce")
plot_df["Y"] = pd.to_numeric(plot_df["Y"], errors="coerce")

# 位置ベースの numpy マスクで絞り込み（reindex 問題を回避）
m = (
    np.isfinite(plot_df["X"].to_numpy())
    & np.isfinite(plot_df["Y"].to_numpy())
    & (plot_df["X"].to_numpy() > 0)
    & (plot_df["Y"].to_numpy() > 0)
)
plot_df = plot_df.loc[m].reset_index(drop=True)

if plot_df.empty:
    st.warning("有効なデータがありません。列指定や方式の選択を見直してください。")
    st.stop()

# 散布図
trend_arg = "ols" if trend else None
fig = px.scatter(
    plot_df,
    x="X",
    y="Y",
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

# サマリ & 相関
st.subheader("方式別サマリ")
summary_df = make_summary(plot_df, "水処理方式現有", "X", "Y")
for c in ["X平均", "X中央値", "Y平均", "Y中央値"]:
    summary_df[c] = summary_df[c].round(2)
st.dataframe(summary_df, use_container_width=True)

corr = np.corrcoef(plot_df["X"], plot_df["Y"])[0, 1]
st.caption(f"全体相関係数（Pearson r）: **{corr:.3f}**")
