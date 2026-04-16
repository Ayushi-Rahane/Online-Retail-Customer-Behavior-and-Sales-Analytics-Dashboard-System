import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from city_clustering import run_clustering_analysis

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Olist Sales Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<style>
:root {
    --blue          : #4a90e2;
    --red           : #e05a5a;
    --orange        : #f59e0b;
    --yellow        : #eab308;
    --primary       : #85b378;
    --muted         : rgba(55,65,81,0.6);
    --border        : rgba(0,0,0,0.08);
    --font-display  : 'Syne', sans-serif;
    --font-body     : 'DM Sans', sans-serif;
}

/* Selected values inside multiselect */
span[data-baseweb="tag"] {
    background-color: #22c55e !important;
    color: #212e21 !important;
}
input[data-testid="stDateInputField"] {
    color: #1f2d1f !important;
    font-weight: 500;
}
div[data-baseweb="select"] span {
    color: #1f2937 !important;
}
div[data-baseweb="select"] input {
    color: #1f2937 !important;
}

/* Global reset */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    color: #374151 !important;
}

/* App background */
.stApp {
    background: #9bbaa1 !important;
    min-height: 100vh;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #789c7f !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }

.sidebar-inner { padding: 28px 20px; }

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 32px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
}
.sidebar-brand .brand-icon {
    width: 38px; height: 38px;
    background: var(--primary);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; color: #061f33;
}
.sidebar-brand .brand-name {
    font-family: var(--font-display) !important;
    font-size: 17px; font-weight: 800;
    color: #ffffff !important;
    line-height: 1.1;
}
.sidebar-brand .brand-sub {
    font-size: 11px;
    color: #212e21 !important;
}

.filter-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #212e21 !important;
    margin: 20px 0 8px 0;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #fff7ed !important;
    border: 1px solid #f59e0b !important;
    border-radius: 10px !important;
    color: #1f2937 !important;
}

div[data-testid="stSlider"] div[role="slider"] {
    background: var(--primary) !important;
    color: #374151 !important;
}
div[data-testid="stSlider"] div[data-testid="stTickBar"] > div {
    background: var(--primary) !important;
    color: #374151 !important;
}

/* Tabs */
div[data-testid="stTabs"] button {
    font-family: var(--font-display) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #212e21 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 8px 20px !important;
    border: none !important;
    background: transparent !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #f59e0b !important;
    background: #c1d4c1 !important;
}
div[data-testid="stTabs"] > div:first-child {
    border-bottom: 1px solid var(--border) !important;
}

.chart-box { background-color: #c1d4c1; }

.js-plotly-plot, .plotly {
    background-color: transparent !important;
}

/* KPI cards */
.kpi-row { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 180px;
    background: #c1d4c1;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px 24px;
    backdrop-filter: blur(6px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(92,163,178,0.15);
}
.kpi-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px; }
.kpi-card:nth-child(1) .kpi-icon { background: rgba(74,144,226,0.15); color: var(--blue); }
.kpi-card:nth-child(2) .kpi-icon { background: rgba(224,90,90,0.15); color: var(--red); }
.kpi-card:nth-child(3) .kpi-icon { background: rgba(245,158,11,0.15); color: var(--orange); }
.kpi-card:nth-child(4) .kpi-icon { background: rgba(234,179,8,0.15); color: var(--yellow); }
.kpi-card:nth-child(5) .kpi-icon { background: rgba(133,179,120,0.15); color: var(--primary); }
.kpi-card:nth-child(6) .kpi-icon { background: rgba(74,144,226,0.15); color: var(--blue); }
.kpi-badge {
    font-size: 11px; font-weight: 700;
    padding: 4px 9px; border-radius: 20px;
    background: rgba(133,179,120,0.15);
    color: var(--primary);
}
.kpi-label {
    font-size: 12px; font-weight: 500;
    color: #6b7c72 !important;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif !important;
    font-size: 25px; font-weight: 800;
    color: #1f2a24 !important;
    line-height: 1;
}
.kpi-sub {
    font-size: 11px;
    color: var(--muted) !important;
    margin-top: 6px;
}

.section-header {
    background: #EAF4E4;
    padding: 10px 14px;
    border-radius: 10px;
    font-weight: 700;
    color: #1F2D1F;
    border-left: 6px solid #FF8C42;
    margin-bottom: 10px;
}
.section-header .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--primary);
    display: inline-block;
}

.page-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 28px;
    padding-bottom: 18px;
    border-bottom: 1px solid var(--border);
}
.page-title {
    font-family: var(--font-display) !important;
    font-size: 26px; font-weight: 800;
    color: #ffffff !important;
    line-height: 1.1;
}
.page-subtitle {
    font-size: 13px;
    color: #212e21 !important;
    margin-top: 4px;
}
.page-badge {
    background: rgba(133,179,120,0.15);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 6px 16px;
    font-size: 12px; font-weight: 600;
    color: var(--primary);
}

input[type="date"] { color: #212e21 !important; }

div[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
}

details summary {
    color: var(--primary) !important;
    font-size: 13px !important;
}

hr { border-color: var(--border) !important; }

label, .stSelectbox label, .stMultiSelect label,
.stDateInput label, .stSlider label {
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    background-color: #ffff33 !important;
    color: #212e21 !important;
    border: 1px solid #f59e0b !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 28px !important; padding-bottom: 40px !important; }

.pred-card {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(12px);
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file_path: str = "final_cleaned_data.csv") -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = df.rename(columns={
        "order_purchase_timestamp": "order_date",
        "total_order_value"       : "price",
        "category_en"             : "category",
        "customer_city"           : "city",
        "payment_types"           : "payment_type",
    })
    df["order_date"]   = pd.to_datetime(df["order_date"], errors="coerce")
    df                 = df.dropna(subset=["order_date"])
    df["category"]     = df["category"].replace("unknown", "Others").fillna("Others")
    df["payment_type"] = df["payment_type"].astype(str).str.split("|").str[0]
    df["year_month"]   = df["order_date"].dt.to_period("M").dt.to_timestamp()
    return df

# ─────────────────────────────────────────────────────────────
# PLOTLY THEME HELPER
# ─────────────────────────────────────────────────────────────
def theme(fig):
    fig.update_layout(
        paper_bgcolor="#c1d4c1",
        plot_bgcolor="#c1d4c1",
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(color="#1f2d1f"),
        title_font=dict(color="#1f2d1f")
    )
    fig.update_xaxes(title_font=dict(color="#1f2d1f"), tickfont=dict(color="#1f2d1f"))
    fig.update_yaxes(title_font=dict(color="#1f2d1f"), tickfont=dict(color="#1f2d1f"))
    return fig

def style_table(df):
    return df.style.set_properties(
        **{
            "background-color": "#c1d4c1",
            "color": "#1f2d1f",
            "border-color": "#ffffff"
        }
    ).set_table_styles([
        {
            "selector": "th",
            "props": [
                ("background-color", "#f97316"),
                ("color", "white"),
                ("font-weight", "bold"),
                ("text-align", "center")
            ]
        },
        {
            "selector": "td",
            "props": [("text-align", "center")]
        }
    ])

# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────
try:
    raw_df = load_data()
except FileNotFoundError:
    st.error("⚠️  `final_cleaned_data.csv` not found. Place it in the same folder as app.py.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-inner">
        <div class="sidebar-brand">
            <div class="brand-icon"><i class="fa-solid fa-chart-line"></i></div>
            <div>
                <div class="brand-name">Olist Analytics</div>
                <div class="brand-sub">Sales Intelligence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="filter-label"> Location</div>', unsafe_allow_html=True)
    all_cities = sorted(raw_df["city"].dropna().unique().tolist())
    selected_cities = st.multiselect(
        "City",
        ["Select All"] + all_cities,
        default=[],
        placeholder="All cities",
        label_visibility="collapsed"
    )

    st.markdown('<div class="filter-label"> Category</div>', unsafe_allow_html=True)
    all_cats = sorted(raw_df["category"].dropna().unique().tolist())
    selected_cats = st.multiselect(
        "Category",
        ["Select All"] + all_cats,
        default=[],
        placeholder="All categories",
        label_visibility="collapsed"
    )

    st.markdown('<div class="filter-label"> Date Range</div>', unsafe_allow_html=True)
    min_date = raw_df["order_date"].min().date()
    max_date = raw_df["order_date"].max().date()
    date_start = st.date_input("From", value=min_date, min_value=min_date, max_value=max_date,label_visibility="collapsed")
    date_end   = st.date_input("To",   value=max_date, min_value=min_date, max_value=max_date,label_visibility="collapsed")

    st.markdown('<div class="filter-label"> Payment Type</div>', unsafe_allow_html=True)
    all_pay = sorted(raw_df["payment_type"].dropna().unique().tolist())
    selected_pay = st.multiselect(
        "Payment",
        ["Select All"] + all_pay,
        default=["Select All"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:11px; color:#1f2d1f; text-align:center; padding:4px 0;">
        Dataset: {len(raw_df):,} orders &nbsp;|&nbsp; Olist Brazil
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────────────────────
df = raw_df.copy()

if selected_cities:
    if "Select All" in selected_cities:
        df = df[df["city"].isin(all_cities)]
    else:
        df = df[df["city"].isin(selected_cities)]

if selected_cats:
    if "Select All" in selected_cats:
        df = df[df["category"].isin(all_cats)]
    else:
        df = df[df["category"].isin(selected_cats)]

df = df[
    (df["order_date"].dt.date >= date_start) &
    (df["order_date"].dt.date <= date_end)
]

if selected_pay:
    if "Select All" in selected_pay:
        df = df[df["payment_type"].isin(all_pay)]
    else:
        df = df[df["payment_type"].isin(selected_pay)]

# ─────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
    <div>
        <div class="page-title">Sales Analytics Dashboard</div>
        <div class="page-subtitle">
            Showing {len(df):,} orders &nbsp;·&nbsp;
            {date_start.strftime('%b %d, %Y')} → {date_end.strftime('%b %d, %Y')}
        </div>
    </div>
    <div class="page-badge"><i class="fa-solid fa-circle" style="font-size:7px;margin-right:6px;color:#4ade80;"></i>Live View</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  Overview",
    "  Trends",
    "  Comparative",
    "  Advanced"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
with tab1:
    st.write("")

    total_sales      = df["price"].sum()
    total_orders     = df["order_id"].count() if "order_id" in df.columns else len(df)
    total_customers  = df["customer_id"].nunique() if "customer_id" in df.columns else df["order_id"].nunique()
    avg_order_value  = total_sales / total_orders if total_orders > 0 else 0
    total_categories = df["category"].nunique()
    total_cities     = df["city"].nunique()

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-top">
                <div class="kpi-icon"><i class="fa-solid fa-dollar-sign"></i></div>
                <span class="kpi-badge">↑ 12.5%</span>
            </div>
            <div class="kpi-label">Total Sales</div>
            <div class="kpi-value">R${total_sales:,.0f}</div>
            <div class="kpi-sub">Revenue generated</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-top">
                <div class="kpi-icon"><i class="fa-solid fa-cart-shopping"></i></div>
                <span class="kpi-badge">↑ 8.2%</span>
            </div>
            <div class="kpi-label">Total Orders</div>
            <div class="kpi-value">{total_orders:,}</div>
            <div class="kpi-sub">Placed in period</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-top">
                <div class="kpi-icon"><i class="fa-solid fa-users"></i></div>
                <span class="kpi-badge">↑ 5.4%</span>
            </div>
            <div class="kpi-label">Total Customers</div>
            <div class="kpi-value">{total_customers:,}</div>
            <div class="kpi-sub">Unique customers</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-top">
                <div class="kpi-icon"><i class="fa-solid fa-receipt"></i></div>
                <span class="kpi-badge">↑ 2.1%</span>
            </div>
            <div class="kpi-label">Avg Order Value</div>
            <div class="kpi-value">R${avg_order_value:,.2f}</div>
            <div class="kpi-sub">Per transaction</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-top">
                <div class="kpi-icon"><i class="fa-solid fa-tag"></i></div>
                <span class="kpi-badge">Active</span>
            </div>
            <div class="kpi-label">Categories</div>
            <div class="kpi-value">{total_categories}</div>
            <div class="kpi-sub">Product segments</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-top">
                <div class="kpi-icon"><i class="fa-solid fa-location-dot"></i></div>
                <span class="kpi-badge">Active</span>
            </div>
            <div class="kpi-label">Cities Covered</div>
            <div class="kpi-value">{total_cities}</div>
            <div class="kpi-sub">Geographic reach</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_snap, col_status = st.columns([2, 1])

    with col_snap:
        st.markdown('<div class="section-header">Quick Sales Snapshot</div>', unsafe_allow_html=True)
        snap = df.groupby("year_month", as_index=False)["price"].sum()
        if not snap.empty:
            fig_snap = px.area(snap, x="year_month", y="price", color_discrete_sequence=["#E78644"])
            fig_snap.update_traces(fill="tozeroy",line=dict(color="#C04000", width=2.5),fillcolor="rgba(231,134,68,0.25)")
            fig_snap.update_xaxes(title="")
            fig_snap.update_yaxes(title="Sales (R$)")
            fig_snap.update_layout(title_text="")
            theme(fig_snap)
            st.plotly_chart(fig_snap, use_container_width=True)

    with col_status:
        st.markdown('<div class="section-header">Order Status</div>', unsafe_allow_html=True)
        if "order_status" in df.columns:
            status_ct = df["order_status"].value_counts().reset_index()
            status_ct.columns = ["Status", "Count"]
            fig_status = px.pie(status_ct, names="Status", values="Count",
                                color_discrete_sequence=["#E78644","#2E6F40","#C04000","#E1C699","#b85c2e","#1f4d2e"])
            fig_status.update_traces(textposition="inside", textinfo="percent+label",hole=0.45,marker=dict(line=dict(color="rgba(0,0,0,0)", width=0)))
            fig_status.update_layout(title_text="", showlegend=False)
            theme(fig_status)
            st.plotly_chart(fig_status, use_container_width=True)

    st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)

    preview_cols = [c for c in [
        "order_id","order_date","city","category",
        "price","payment_type","review_score"
    ] if c in df.columns]

    st.dataframe(
        style_table(df[preview_cols].head(50)),
        use_container_width=True
    )
# ══════════════════════════════════════════════════════════════
# TAB 2 — TRENDS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.write("")

    monthly_sales  = df.groupby("year_month", as_index=False)["price"].sum()
    monthly_orders = df.groupby("year_month").size().reset_index(name="orders")

    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown('<div class="section-header">Monthly Sales Trend</div>', unsafe_allow_html=True)
        if not monthly_sales.empty:
            fig_ms = go.Figure()
            fig_ms.add_trace(go.Scatter(
                x=monthly_sales["year_month"], y=monthly_sales["price"],
                mode="lines+markers",
                line=dict(color="#2E6F40", width=2.5),
                marker=dict(size=6, color="#2E6F40"),
                fill="tozeroy",
                fillcolor="rgba(46,111,64,0.2)"
            ))
            fig_ms.update_xaxes(title="Month")
            fig_ms.update_yaxes(title="Total Sales (R$)")
            fig_ms.update_layout(title_text="")
            theme(fig_ms)
            st.plotly_chart(fig_ms, use_container_width=True)

    with col_t2:
        st.markdown('<div class="section-header">Monthly Orders Trend</div>', unsafe_allow_html=True)
        if not monthly_orders.empty:
            fig_mo = go.Figure()
            fig_mo.add_trace(go.Scatter(
                x=monthly_orders["year_month"], y=monthly_orders["orders"],
                mode="lines+markers",
                line=dict(color="#E78644", width=2.5),
                marker=dict(size=6, color="#E78644"),
                fill="tozeroy",
                fillcolor="rgba(231,134,68,0.2)"
            ))
            fig_mo.update_xaxes(title="Month")
            fig_mo.update_yaxes(title="Number of Orders")
            fig_mo.update_layout(title_text="")
            theme(fig_mo)
            st.plotly_chart(fig_mo, use_container_width=True)

    st.markdown('<div class="section-header">Top Performing Months</div>', unsafe_allow_html=True)
    if not monthly_sales.empty:
        top_months = monthly_sales.sort_values("year_month").copy()
        top_months["month_label"] = top_months["year_month"].dt.strftime("%b %Y")
        fig_tm = px.bar(
            top_months, x="month_label", y="price",
            color="price",
            color_continuous_scale=[[0, "#E1C699"], [0.5, "#E78644"], [1, "#C04000"]]
        )
        fig_tm.update_traces(marker_line_width=0)
        fig_tm.update_xaxes(title="Month", tickangle=-30)
        fig_tm.update_yaxes(title="Sales (R$)")
        fig_tm.update_layout(title_text="")
        theme(fig_tm)
        st.plotly_chart(fig_tm, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — COMPARATIVE
# ══════════════════════════════════════════════════════════════
with tab3:
    st.write("")

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown('<div class="section-header">Category-wise Sales Distribution</div>', unsafe_allow_html=True)
        cat_sales = (df.groupby("category", as_index=False)["price"].sum().sort_values("price", ascending=False).head(10))
        if not cat_sales.empty:
            fig_cat = px.bar(
                cat_sales, x="category", y="price",
                labels={"category": "Category", "price": "Total Sales"},
                color="price",
                color_continuous_scale=[[0, "#85b378"], [0.5, "#3d7a88"], [1, "#1e4a56"]],
                range_color=[0, cat_sales["price"].max()]
            )
            fig_cat.update_traces(marker_line_width=0)
            fig_cat.update_layout(xaxis_tickangle=-45, showlegend=False,coloraxis_colorbar=dict(yanchor="bottom", y=0, len=1))
            fig_cat.update_yaxes(title="Total Sales", tickformat=".2s", rangemode="tozero")
            fig_cat.update_xaxes(title="Category")
            theme(fig_cat)
            st.plotly_chart(fig_cat, use_container_width=True)

    with col_c2:
        st.markdown('<div class="section-header">Payment Method Distribution</div>', unsafe_allow_html=True)
        pay_ct = df["payment_type"].value_counts().reset_index()
        pay_ct.columns = ["payment_type", "count"]
        if not pay_ct.empty:
            fig_pay = px.pie(pay_ct, names="payment_type", values="count",color_discrete_sequence=["#E78644","#2E6F40","#C04000","#E1C699","#b85c2e"])
            fig_pay.update_traces(textposition="inside", textinfo="percent+label", hole=0.38,marker=dict(line=dict(color="rgba(0,0,0,0)", width=0)))
            fig_pay.update_layout(title_text="", showlegend=True)
            theme(fig_pay)
            st.plotly_chart(fig_pay, use_container_width=True)

    st.markdown('<div class="section-header">Top 10 Cities by Sales</div>', unsafe_allow_html=True)
    city_sales = (df.groupby("city", as_index=False)["price"].sum().sort_values("price", ascending=False).head(10))
    if not city_sales.empty:
        fig_city = px.bar(
            city_sales, x="city", y="price",
            labels={"city": "City", "price": "Total Sales"},
            color="price",
            color_continuous_scale=[[0, "#85b378"], [0.5, "#3d7a88"], [1, "#1e4a56"]],
            range_color=[0, city_sales["price"].max()]
        )
        fig_city.update_traces(marker_line_width=0)
        fig_city.update_layout(xaxis_tickangle=-45, showlegend=False,coloraxis_colorbar=dict(yanchor="bottom", y=0, len=1))
        fig_city.update_yaxes(title="Total Sales", tickformat=".2s", rangemode="tozero")
        fig_city.update_xaxes(title="City")
        theme(fig_city)
        st.plotly_chart(fig_city, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — ADVANCED
# ══════════════════════════════════════════════════════════════
with tab4:
    st.write("")

    # Sales Prediction
    st.markdown(
        '<div class="section-header">Sales Prediction — Linear Regression (6-Month Forecast)</div>',
        unsafe_allow_html=True
    )

    monthly_pred = (
        df.groupby("year_month", as_index=False)["price"]
        .sum()
        .sort_values("year_month")
        .dropna()
    )
    monthly_pred = monthly_pred[monthly_pred["price"] > 0].iloc[:-1].reset_index(drop=True)

    if len(monthly_pred) >= 3:
        n = len(monthly_pred)
        X = np.arange(n).reshape(-1, 1)
        Y = monthly_pred["price"].values

        model = LinearRegression().fit(X, Y)

        future_X = np.arange(n, n + 6).reshape(-1, 1)
        predicted_Y = model.predict(future_X)

        last_date = monthly_pred["year_month"].iloc[-1]
        future_dates = pd.Series([last_date + pd.DateOffset(months=i) for i in range(1, 7)])

        fig_pred = go.Figure()
        fig_pred.update_layout(title_text="")
        fig_pred.add_trace(go.Scatter(
            x=monthly_pred["year_month"], y=Y,
            mode="lines+markers",
            name="Actual Sales",
            line=dict(color="#2E6F40", width=2.5),
            marker=dict(size=6)
        ))
        conn_x = pd.concat([pd.Series([last_date]), future_dates])
        conn_y = np.concatenate(([Y[-1]], predicted_Y))
        fig_pred.add_trace(go.Scatter(
            x=conn_x, y=conn_y,
            mode="lines",
            name="Predicted Trend",
            line=dict(color="#E78644", width=2.5, dash="dash")
        ))
        theme(fig_pred)
        st.markdown('<div class="chart-box">', unsafe_allow_html=True)
        st.plotly_chart(fig_pred, use_container_width=True, key="sales_pred_chart")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Not enough data for prediction. Adjust filters.")

    # City Clustering
    run_clustering_analysis()

    # Review Score & Delivery
    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown('<div class="section-header">Review Score Distribution</div>', unsafe_allow_html=True)
        if "review_score" in df.columns:
            rev_ct = df["review_score"].dropna().value_counts().sort_index().reset_index()
            rev_ct.columns = ["score", "count"]
            fig_rev = px.bar(
                rev_ct, x="score", y="count",
                color="count",
                color_continuous_scale=[[0, "#E1C699"], [1, "#2E6F40"]]
            )
            fig_rev.update_xaxes(title="Review Score (1–5)")
            fig_rev.update_yaxes(title="Number of Reviews")
            fig_rev.update_layout(title_text="")
            theme(fig_rev)
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            st.plotly_chart(fig_rev, use_container_width=True, key="review_chart")
            st.markdown('</div>', unsafe_allow_html=True)

    with col_a2:
        st.markdown('<div class="section-header">Delivery Performance</div>', unsafe_allow_html=True)
        if "days_to_delivery" in df.columns:
            df["delivery_speed"] = pd.cut(
                df["days_to_delivery"],
                bins=[0, 7, 20, 999],
                labels=["Fast (0–7 days)", "Normal (8–20 days)", "Late (21+ days)"]
            )
            speed_ct = (
                df["delivery_speed"]
                .value_counts()
                .reindex(["Fast (0–7 days)", "Normal (8–20 days)", "Late (21+ days)"])
                .reset_index()
            )
            speed_ct.columns = ["delivery_speed", "count"]
            fig_del = px.bar(
                speed_ct, x="delivery_speed", y="count",
                color="delivery_speed",
                color_discrete_map={
                    "Fast (0–7 days)"   : "#2E6F40",
                    "Normal (8–20 days)": "#E78644",
                    "Late (21+ days)"   : "#e05a5a"
                }
            )
            fig_del.update_xaxes(title="Delivery Speed")
            fig_del.update_yaxes(title="Order Count")
            fig_del.update_layout(title_text="", showlegend=False)
            theme(fig_del)
            st.markdown('<div class="chart-box">', unsafe_allow_html=True)
            st.plotly_chart(fig_del, use_container_width=True, key="delivery_chart")
            st.markdown('</div>', unsafe_allow_html=True)

    # Numeric Summary
    st.markdown('<div class="section-header">Numeric Summary</div>', unsafe_allow_html=True)
    num_cols = [c for c in [
        "price", "total_price", "total_freight", "total_order_value",
        "payment_value", "review_score", "days_to_delivery",
        "delivery_delay_days", "n_items", "max_installments",
        "product_weight_g"
    ] if c in df.columns]
    if num_cols:
        summary = df[num_cols].describe().round(2)
        st.dataframe(style_table(summary), use_container_width=True)