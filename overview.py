import pandas as pd
import streamlit as st

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
# KPI CALCULATIONS
# ----------------------------------------------------
def calculate_kpis(df: pd.DataFrame) -> dict:
    # Total Sales = sum of price
    total_sales = df["price"].sum()

    # Total Orders = count of order_id (if column exists, else count rows)
    if "order_id" in df.columns:
        total_orders = df["order_id"].count()
    else:
        total_orders = len(df)

    # Total Customers = count of unique customer_id
    # Fallback to order_id uniqueness if customer_id is not available
    if "customer_id" in df.columns:
        total_customers = df["customer_id"].nunique()
    elif "order_id" in df.columns:
        total_customers = df["order_id"].nunique()
    else:
        total_customers = len(df)

    # Average Order Value = total sales / total orders
    avg_order_value = total_sales / total_orders if total_orders > 0 else 0

    # Total Categories = number of unique categories
    total_categories = df["category"].nunique()

    # Total Cities = number of unique cities
    total_cities = df["city"].nunique()

    return {
        "Total Sales": total_sales,
        "Total Orders": total_orders,
        "Total Customers": total_customers,
        "Avg Order Value": avg_order_value,
        "Total Categories": total_categories,
        "Total Cities": total_cities
    }

# ----------------------------------------------------
# RENDER KPI CARDS
# ----------------------------------------------------
def format_value(key: str, value) -> str:
    # Format Total Sales and Avg Order Value as currency with commas
    if key in ("Total Sales", "Avg Order Value"):
        return f"${value:,.2f}"
    # Format all other integer-type KPIs with comma separators
    return f"{int(value):,}"

def render_kpi_section(df: pd.DataFrame):
    # Include Font Awesome CDN
    st.markdown("""
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
        .kpi-card {
            background-color: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            display: flex;
            flex-direction: column;
            justify-content: center;
            border: 1px solid #f1f5f9;
        }
        .kpi-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .kpi-icon {
            font-size: 20px;
            width: 44px;
            height: 44px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .icon-sales { background-color: #eff6ff; color: #3b82f6; }
        .icon-orders { background-color: #f3e8ff; color: #a855f7; }
        .icon-customers { background-color: #fef3c7; color: #f59e0b; }
        .icon-aov { background-color: #e0f2fe; color: #0ea5e9; }
        
        .kpi-trend {
            color: #10b981;
            background-color: #d1fae5;
            padding: 4px 10px;
            border-radius: 16px;
            font-size: 13px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .kpi-title {
            color: #64748b;
            font-size: 15px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .kpi-value {
            color: #0f172a;
            font-size: 32px;
            font-weight: 800;
            margin: 0;
            line-height: 1.2;
        }
        </style>
    """, unsafe_allow_html=True)

    kpis = calculate_kpis(df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-top">
                <div class="kpi-icon icon-sales"><i class="fa-solid fa-dollar-sign"></i></div>
                <div class="kpi-trend"><i class="fa-solid fa-arrow-trend-up"></i> +12.5%</div>
            </div>
            <div class="kpi-title">Total Sales</div>
            <p class="kpi-value">{format_value("Total Sales", kpis["Total Sales"])}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-top">
                <div class="kpi-icon icon-orders"><i class="fa-solid fa-cart-shopping"></i></div>
                <div class="kpi-trend"><i class="fa-solid fa-arrow-trend-up"></i> +8.2%</div>
            </div>
            <div class="kpi-title">Total Orders</div>
            <p class="kpi-value">{format_value("Total Orders", kpis["Total Orders"])}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-top">
                <div class="kpi-icon icon-customers"><i class="fa-solid fa-users"></i></div>
                <div class="kpi-trend"><i class="fa-solid fa-arrow-trend-up"></i> +5.4%</div>
            </div>
            <div class="kpi-title">Total Customers</div>
            <p class="kpi-value">{format_value("Total Customers", kpis["Total Customers"])}</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-top">
                <div class="kpi-icon icon-aov"><i class="fa-solid fa-credit-card"></i></div>
                <div class="kpi-trend"><i class="fa-solid fa-arrow-trend-up"></i> +2.1%</div>
            </div>
            <div class="kpi-title">Average Order Value</div>
            <p class="kpi-value">{format_value("Avg Order Value", kpis["Avg Order Value"])}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("")

# ----------------------------------------------------
# STANDALONE ENTRY POINT
# Allows this file to be run on its own for testing:
#   streamlit run overview.py
# ----------------------------------------------------
def main():
    st.set_page_config(
        page_title="Overview - KPI Dashboard",
        layout="wide"
    )
    st.title("Overview")

    df = load_data("final_cleaned_data.csv")
    render_kpi_section(df)

if __name__ == "__main__":
    main()
