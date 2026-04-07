import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Dashboard", layout="wide")

# ------------------ DARK UI ------------------
st.markdown("""
<style>
.main {background-color: #0e1117;}
.card {
    background-color: #161b22;
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 0px 0px 20px rgba(0,255,255,0.08);
}
.metric {
    font-size: 28px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    return pd.read_csv("final_cleaned_data.csv")

df = load_data()

# ------------------ PROCESS DATA ------------------
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['month_year'] = df['order_purchase_timestamp'].dt.to_period('M').astype(str)

# ------------------ AGG DATA ------------------
monthly_sales = df.groupby('month_year')['total_price'].sum().reset_index()
monthly_orders = df.groupby('month_year').size().reset_index(name='orders')

# ------------------ TABS ------------------
tab1, tab2 = st.tabs([" Overview", " Trends"])

# =================================================
# 🔵 OVERVIEW (PROFESSIONAL)
# =================================================
with tab1:
    st.markdown("##  Business Overview")

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"<div class='card'><div class='metric'>₹ {int(df['total_price'].sum())}</div>Total Revenue</div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><div class='metric'>{len(df)}</div>Total Orders</div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'><div class='metric'>{df['customer_city'].nunique() if 'customer_city' in df else 0}</div>Cities Covered</div>", unsafe_allow_html=True)

    # ---------- MINI TREND ----------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader(" Quick Sales Snapshot")

    mini_fig = px.area(
        monthly_sales,
        x='month_year',
        y='total_price'
    )

    mini_fig.update_layout(
        plot_bgcolor='#161b22',
        paper_bgcolor='#161b22',
        font=dict(color='white'),
        xaxis_title="Month",
        yaxis_title="Sales"
    )

    st.plotly_chart(mini_fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- SUMMARY TABLE ----------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader(" Data Summary")

    summary = pd.DataFrame({
        "Metric": ["Total Revenue", "Total Orders", "Unique Cities"],
        "Value": [
            f"₹ {int(df['total_price'].sum())}",
            len(df),
            df['customer_city'].nunique() if 'customer_city' in df else 0
        ]
    })

    st.table(summary)

    st.markdown('</div>', unsafe_allow_html=True)

# =================================================
# 🟣 TRENDS
# =================================================
with tab2:
    st.markdown("##  Trend Analysis")

    # ---------- SALES LINE ----------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader(" Monthly Sales Trend")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_sales['month_year'],
        y=monthly_sales['total_price'],
        mode='lines+markers',
        line=dict(color='#00f5ff', width=3)
    ))

    fig.update_layout(
        plot_bgcolor='#161b22',
        paper_bgcolor='#161b22',
        font=dict(color='white'),
        xaxis_title="Month",
        yaxis_title="Total Sales (₹)"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- ORDERS LINE ----------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader(" Monthly Orders Trend")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=monthly_orders['month_year'],
        y=monthly_orders['orders'],
        mode='lines+markers',
        line=dict(color='#ff4b6e', width=3)
    ))

    fig2.update_layout(
        plot_bgcolor='#161b22',
        paper_bgcolor='#161b22',
        font=dict(color='white'),
        xaxis_title="Month",
        yaxis_title="Number of Orders"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- TOP MONTHS BAR ----------
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader(" Top Performing Months")

    top_months = monthly_sales.sort_values(by='total_price', ascending=False)

    fig3 = px.bar(
        top_months,
        x='month_year',
        y='total_price',
        color='total_price',
        color_continuous_scale='viridis'
    )

    fig3.update_layout(
        plot_bgcolor='#161b22',
        paper_bgcolor='#161b22',
        font=dict(color='white'),
        xaxis_title="Month",
        yaxis_title="Total Sales (₹)"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(" Top months indicate peak business performance.")

    st.markdown('</div>', unsafe_allow_html=True)