import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import streamlit as st

def run_clustering_analysis():
    # 1. Data Preprocessing
    # Load dataset using pandas
    try:
        df = pd.read_csv("final_cleaned_data.csv")
    except FileNotFoundError:
        st.error("Dataset 'final_cleaned_data.csv' not found. Please ensure it is in the same directory.")
        return

    # Rename columns
    column_mapping = {
        "order_purchase_timestamp": "order_date",
        "total_order_value": "price",
        "category_en": "category",
        "customer_city": "city",
        "payment_types": "payment_type"
    }
    df = df.rename(columns=column_mapping)

    # Convert order_date to datetime format
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    
    # Clean data
    # Replace "unknown" category with "Others"
    if "category" in df.columns:
        df["category"] = df["category"].replace("unknown", "Others").fillna("Others")
    
    # Extract first value from payment_type (split by "|")
    if "payment_type" in df.columns:
        df["payment_type"] = df["payment_type"].astype(str).str.split("|").str[0]

    # 2. Feature Engineering
    # Create city-level aggregated features
    # Total Sales per city -> sum(price)
    # Number of Orders per city -> count(order_id)
    city_metrics = df.groupby("city").agg(
        total_sales=("price", "sum"),
        total_orders=("order_id", "count")
    ).reset_index()
    
    # Store this in a new DataFrame: columns -> [city, total_sales, total_orders]
    city_features = city_metrics[["city", "total_sales", "total_orders"]].copy()

    # Handle any potential missing values before clustering
    city_features = city_features.dropna()

    # 3. Clustering
    # Use K-Means from sklearn
    # Features: total_sales, total_orders
    X = city_features[["total_sales", "total_orders"]]

    # Choose k = 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    
    # Fit the model and assign cluster labels to each city
    # Convert cluster labels to string for categorical coloring in Plotly
    city_features["cluster"] = kmeans.fit_predict(X).astype(str)

    # 4. Visualization
    st.markdown('<div class="section-header">City-wise Sales Clustering</div>', unsafe_allow_html=True)

    # Create a scatter plot using Plotly Express
    fig = px.scatter(
        city_features,
        x="total_orders",
        y="total_sales",
        color="cluster",
        hover_name="city",
        labels={
            "total_orders": "Total Orders",
            "total_sales": "Total Sales (R$)",
            "cluster": "Cluster"
        },
        color_discrete_sequence=["#2E6F40", "#E78644", "#85b378"]
    )

    # Use a layout matching dashboard themes
    fig.update_layout(
        title_text="",
        paper_bgcolor="#c1d4c1",
        plot_bgcolor="#c1d4c1",
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(color="#1f2d1f"),
        title_font=dict(color="#1f2d1f"),
        legend_title_text="Sales Cluster"
    )
    
    fig.update_xaxes(
        title_font=dict(color="#1f2d1f"),
        tickfont=dict(color="#1f2d1f")
    )
    
    fig.update_yaxes(
        title_font=dict(color="#1f2d1f"),
        tickfont=dict(color="#1f2d1f")
    )

    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='rgba(255,255,255,0.5)')))

    # 5. Output / 7. Streamlit version
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    st.set_page_config(page_title="City Clustering Analysis", layout="wide")
    st.title("Advanced Analytics: City-wise Behavior")
    run_clustering_analysis()
