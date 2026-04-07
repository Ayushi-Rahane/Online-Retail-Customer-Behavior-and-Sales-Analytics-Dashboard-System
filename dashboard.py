import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from overview import render_kpi_section

# ----------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="Online Retail Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------------------------------
# DATA LOADING AND PREPROCESSING
# ----------------------------------------------------
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    # Load CSV file
    df = pd.read_csv(file_path)

    # Rename columns
    df = df.rename(columns={
        "order_purchase_timestamp": "order_date",
        "total_order_value": "price",
        "category_en": "category",
        "customer_city": "city",
        "payment_types": "payment_type"
    })

    # Convert order_date to datetime
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"])

    # Replace unknown categories with Others
    df["category"] = df["category"].replace("unknown", "Others")
    df["category"] = df["category"].fillna("Others")

    # Keep only first payment type (split by "|")
    df["payment_type"] = df["payment_type"].astype(str).str.split("|").str[0]

    return df

# ----------------------------------------------------
# CHART: CATEGORY-WISE SALES BAR CHART
# ----------------------------------------------------
def chart_category_sales(df: pd.DataFrame):
    category_sales = df.groupby("category", as_index=False)["price"].sum()
    category_sales = category_sales.sort_values(by="price", ascending=False).head(10)

    fig = px.bar(
        category_sales,
        x="category",
        y="price",
        title="Top 10 Categories by Sales",
        labels={"category": "Category", "price": "Total Sales"},
        color="price",
        color_continuous_scale=px.colors.sequential.Blues
    )
    fig.update_layout(xaxis_tickangle=-45, template="plotly_white", showlegend=False)
    return fig

# ----------------------------------------------------
# CHART: CITY-WISE SALES BAR CHART
# ----------------------------------------------------
def chart_city_sales(df: pd.DataFrame):
    city_sales = df.groupby("city", as_index=False)["price"].sum()
    city_sales = city_sales.sort_values(by="price", ascending=False).head(10)

    fig = px.bar(
        city_sales,
        x="city",
        y="price",
        title="Top 10 Cities by Sales",
        labels={"city": "City", "price": "Total Sales"},
        color="price",
        color_continuous_scale=px.colors.sequential.Teal
    )
    fig.update_layout(xaxis_tickangle=-45, template="plotly_white", showlegend=False)
    return fig

# ----------------------------------------------------
# CHART: PAYMENT METHOD DISTRIBUTION PIE CHART
# ----------------------------------------------------
def chart_payment_distribution(df: pd.DataFrame):
    payment_counts = df["payment_type"].value_counts().reset_index()
    payment_counts.columns = ["payment_type", "count"]

    fig = px.pie(
        payment_counts,
        names="payment_type",
        values="count",
        title="Payment Method Distribution",
        color_discrete_sequence=px.colors.sequential.Teal
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig

# ----------------------------------------------------
# CHART: SALES PREDICTION LINE CHART
# ----------------------------------------------------
def chart_sales_prediction(df: pd.DataFrame, predict_months: int = 6):
    # Aggregate monthly sales
    df = df.copy()
    df["year_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    monthly_sales = df.groupby("year_month", as_index=False)["price"].sum()
    monthly_sales = monthly_sales.sort_values(by="year_month")

    # Remove zero/null monthly sales and drop the last (potentially incomplete) month
    monthly_sales = monthly_sales.dropna(subset=["price"])
    monthly_sales = monthly_sales[monthly_sales["price"] > 0]
    monthly_sales = monthly_sales.iloc[:-1].reset_index(drop=True)

    if len(monthly_sales) < 2:
        return None

    # Prepare X (month index) and Y (total sales) for regression
    n_records = len(monthly_sales)
    X = np.arange(n_records).reshape(-1, 1)
    Y = monthly_sales["price"].values

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X, Y)

    # Predict only for future months (after last actual date)
    future_X = np.arange(n_records, n_records + predict_months).reshape(-1, 1)
    predicted_Y = model.predict(future_X)

    # Generate future date labels
    last_date = monthly_sales["year_month"].iloc[-1]
    future_dates = pd.Series([last_date + pd.DateOffset(months=i) for i in range(1, predict_months + 1)])
    actual_dates = monthly_sales["year_month"].reset_index(drop=True)

    # Bridge connection from last actual point to first predicted point
    bridge_x = pd.concat([pd.Series([actual_dates.iloc[-1]]), future_dates])
    bridge_y = np.concatenate(([Y[-1]], predicted_Y))

    # Build Plotly figure
    fig = go.Figure()

    # Actual sales line
    fig.add_trace(go.Scatter(
        x=actual_dates,
        y=Y,
        mode="lines+markers",
        name="Actual Sales",
        line=dict(color="steelblue", width=2)
    ))

    # Predicted sales line — dashed, starts after actual data
    fig.add_trace(go.Scatter(
        x=bridge_x,
        y=bridge_y,
        mode="lines",
        name="Predicted Sales",
        line=dict(color="tomato", width=2, dash="dash")
    ))

    fig.update_layout(
        title="Sales Prediction (Actual vs Predicted)",
        xaxis_title="Month",
        yaxis_title="Total Sales",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

# ----------------------------------------------------
# MAIN DASHBOARD
# ----------------------------------------------------
def main():
    # Load data
    df = load_data("final_cleaned_data.csv")

    # Custom Dashboard header matching the UI exact layout
    st.markdown("""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 24px;
            margin-bottom: 24px;
            border-bottom: 1px solid #e2e8f0;
        }
        .header-left h1 {
            font-size: 28px;
            font-weight: 800;
            color: #0f172a;
            margin: 0;
            padding: 0;
        }
        .header-subtitle {
            font-size: 15px;
            color: #64748b;
            margin-top: 4px;
        }
        .header-right {
            display: flex;
            align-items: center;
            gap: 24px;
        }
        .search-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 8px 16px;
            color: #94a3b8;
            font-size: 14px;
            width: 250px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .notification-icon {
            color: #3b82f6;
            font-size: 20px;
            position: relative;
        }
        .notification-icon::after {
            content: '';
            position: absolute;
            top: 2px;
            right: 0px;
            width: 8px;
            height: 8px;
            background: #ef4444;
            border-radius: 50%;
            border: 2px solid white;
        }
        .user-profile {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .user-info {
            text-align: right;
            line-height: 1.2;
        }
        .user-name {
            font-size: 14px;
            font-weight: 700;
            color: #1e293b;
        }
        .user-role {
            font-size: 12px;
            color: #64748b;
        }
        .user-avatar {
            width: 40px;
            height: 40px;
            background-color: #3b82f6;
            color: white;
            border-radius: 50%;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }
        /* Removing default Streamlit top padding to fit header snugly */
        div.block-container {
            padding-top: 2rem;
        }
        </style>
        
        <div class="header-container">
            <div class="header-left">
                <h1>Online Retail Analytics Dashboard</h1>
                <div class="header-subtitle">Customer Behavior and Sales Insights</div>
            </div>
            <div class="header-right">
                <div class="search-box">
                    <i class="fa-solid fa-magnifying-glass"></i> Search reports...
                </div>
                <div class="notification-icon">
                    <i class="fa-solid fa-bell"></i>
                </div>
                <div class="user-profile">
                    <div class="user-info">
                        <div class="user-name">Sarah Analyst</div>
                        <div class="user-role">Data Team</div>
                    </div>
                    <div class="user-avatar">
                        <i class="fa-solid fa-user"></i>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------
    # SECTION 1: Overview (KPI Module)
    # --------------------------------------------------
    render_kpi_section(df)

    # --------------------------------------------------
    # SECTION 3: Comparative Analysis (My Module)
    # --------------------------------------------------
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<h3 style="color:#0f172a; font-size: 20px;">Comparative Analysis</h3>', unsafe_allow_html=True)

    if df.empty:
        st.warning("No data available.")
    else:
        # Row 1: Category and City charts side by side
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(chart_category_sales(df), use_container_width=True)

        with col2:
            st.plotly_chart(chart_city_sales(df), use_container_width=True)

        # Row 2: Payment distribution chart (centered, narrower)
        col3, col4, col5 = st.columns([1, 2, 1])
        with col4:
            st.plotly_chart(chart_payment_distribution(df), use_container_width=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<h3 style="color:#0f172a; font-size: 20px;">Advanced Analysis — Sales Prediction</h3>', unsafe_allow_html=True)

    if df.empty:
        st.warning("No data available.")
    else:
        prediction_fig = chart_sales_prediction(df, predict_months=6)
        if prediction_fig:
            st.plotly_chart(prediction_fig, use_container_width=True)
        else:
            st.warning("Not enough data to generate a sales prediction.")

if __name__ == "__main__":
    main()
