import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import io
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="下水処理場データ分析システム",
    page_icon="🏭",
    layout="wide"
)

def load_excel_data(uploaded_file):
    """A6セルからデータを読み込み"""
    try:
        df = pd.read_excel(uploaded_file, header=4)  # A6 = 5行目
        df.columns = df.columns.astype(str).str.strip()
        st.success(f"データの読み込みに成功しました！データ形状: {df.shape}")
        return df
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {str(e)}")
        return None

def clean_data(df):
    """データクリーニング：空白セルと混合型セルを除去"""
    df_cleaned = df.copy()
    original_shape = df_cleaned.shape
    
    # 完全に空白の行と列を除去
    df_cleaned = df_cleaned.dropna(how="all").dropna(axis=1, how="all")
    
    # 混合型列を除去
    columns_to_remove = []
    for col in df_cleaned.columns:
        numeric_count = 0
        non_numeric_count = 0
        for value in df_cleaned[col].dropna():
            try:
                float(str(value))
                numeric_count += 1
            except:
                non_numeric_count += 1
        if numeric_count > 0 and non_numeric_count > 0:
            columns_to_remove.append(col)
    
    if columns_to_remove:
        st.warning(f"混合型列を検出し、除去しました: {columns_to_remove}")
        df_cleaned = df_cleaned.drop(columns=columns_to_remove)
    
    # 空白セルを除去
    df_cleaned = df_cleaned.dropna()
    st.info(f"データクリーニング完了！元の形状: {original_shape} → クリーニング後: {df_cleaned.shape}")
    return df_cleaned

def identify_column_types(df):
    """数値列とテキスト列を識別"""
    numeric_columns = []
    text_columns = []
    for col in df.columns:
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        if not numeric_series.isna().any():
            numeric_columns.append(col)
        else:
            text_columns.append(col)
    return numeric_columns, text_columns

def cluster_text_data(df, text_column, n_clusters=3):
    """テキストデータのクラスタリング"""
    if text_column not in df.columns or len(df[text_column].dropna()) < 2:
        return df
    
    text_data = df[text_column].dropna().astype(str)
    vectorizer = TfidfVectorizer(max_features=100, stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(text_data)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    
    df_clustered = df.copy()
    text_to_cluster = dict(zip(text_data, cluster_labels))
    df_clustered[f"{text_column}_cluster"] = df_clustered[text_column].map(text_to_cluster)
    
    st.success(f"テキストデータのクラスタリング完了！{n_clusters}個のグループに分類")
    return df_clustered

def create_chart(df, x_col, y_col, chart_type):
    """チャート作成"""
    if chart_type == "scatter":
        fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col} 散布図")
    elif chart_type == "line":
        fig = px.line(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col} 折れ線グラフ")
    elif chart_type == "bar":
        if df[x_col].dtype in ["object", "string"]:
            summary = df.groupby(x_col)[y_col].mean().reset_index()
            fig = px.bar(summary, x=x_col, y=y_col, title=f"{x_col} vs {y_col} 棒グラフ")
        else:
            fig = px.histogram(df, x=x_col, title=f"{x_col} 分布ヒストグラム")
    elif chart_type == "box":
        if df[x_col].dtype in ["object", "string"]:
            fig = px.box(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col} 箱ひげ図")
        else:
            fig = px.box(df, y=y_col, title=f"{y_col} 箱ひげ図")
    elif chart_type == "heatmap":
        if df[x_col].dtype in ["int64", "float64"] and df[y_col].dtype in ["int64", "float64"]:
            fig = px.density_heatmap(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col} 密度ヒートマップ")
        else:
            st.warning("ヒートマップは数値データのみに適用可能です")
            return None
    
    fig.update_layout(height=500)
    return fig

def export_data(df, format_type):
    """データエクスポート"""
    if format_type == "xlsx":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="分析結果")
        return output.getvalue()
    elif format_type == "csv":
        return df.to_csv(index=False).encode("utf-8-sig")

def main():
    st.title("🏭 下水処理場データ分析システム")
    
    # ファイルアップロード
    uploaded_file = st.sidebar.file_uploader("Excelファイルをアップロードしてください", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        # データ読み込み
        df = load_excel_data(uploaded_file)
        if df is not None:
            # データクリーニング
            df_cleaned = clean_data(df)
            if not df_cleaned.empty:
                # 列タイプ識別
                numeric_columns, text_columns = identify_column_types(df_cleaned)
                
                # 分析設定
                st.sidebar.header("分析設定")
                x_column = st.sidebar.selectbox("X軸列", df_cleaned.columns.tolist())
                y_column = st.sidebar.selectbox("Y軸列", df_cleaned.columns.tolist())
                chart_type = st.sidebar.selectbox("チャートタイプ", ["scatter", "line", "bar", "box", "heatmap"])
                
                # テキストクラスタリング
                if text_columns:
                    cluster_text = st.sidebar.checkbox("テキストクラスタリングを有効化")
                    if cluster_text:
                        text_col = st.sidebar.selectbox("クラスタリング対象のテキスト列を選択", text_columns)
                        n_clusters = st.sidebar.slider("クラスター数", 2, 10, 3)
                        df_cleaned = cluster_text_data(df_cleaned, text_col, n_clusters)
                
                # データプレビュー表示
                st.subheader("データプレビュー")
                st.dataframe(df_cleaned.head(10))
                
                # チャート作成
                st.subheader("可視化分析")
                fig = create_chart(df_cleaned, x_column, y_column, chart_type)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # 統計分析
                st.subheader("統計分析")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**{x_column} 統計情報:**")
                    if df_cleaned[x_column].dtype in ["int64", "float64"]:
                        st.write(df_cleaned[x_column].describe())
                    else:
                        st.write(f"ユニーク値数: {df_cleaned[x_column].nunique()}")
                
                with col2:
                    st.write(f"**{y_column} 統計情報:**")
                    if df_cleaned[y_column].dtype in ["int64", "float64"]:
                        st.write(df_cleaned[y_column].describe())
                    else:
                        st.write(f"ユニーク値数: {df_cleaned[y_column].nunique()}")
                
                # 相関分析
                if df_cleaned[x_column].dtype in ["int64", "float64"] and df_cleaned[y_column].dtype in ["int64", "float64"]:
                    correlation = df_cleaned[x_column].corr(df_cleaned[y_column])
                    st.write(f"**相関係数:** {correlation:.4f}")
                
                # エクスポート機能
                st.subheader("データエクスポート")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Excelでエクスポート"):
                        data = export_data(df_cleaned, "xlsx")
                        st.download_button("Excelファイルをダウンロード", data, "分析結果.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                with col2:
                    if st.button("CSVでエクスポート"):
                        data = export_data(df_cleaned, "csv")
                        st.download_button("CSVファイルをダウンロード", data, "分析結果.csv", "text/csv")
                
                # 完全データ表示
                st.subheader("完全データ")
                st.dataframe(df_cleaned)
    else:
        st.info("Excelファイルをアップロードして分析を開始してください")

if __name__ == "__main__":
    main()
