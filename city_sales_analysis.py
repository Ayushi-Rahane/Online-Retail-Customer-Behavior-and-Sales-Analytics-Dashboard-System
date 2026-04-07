import pandas as pd
import plotly.express as px
import sys

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    # 1. Load the CSV file using pandas
    df = pd.read_csv(file_path)
    
    # Rename columns based on requirements
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
    
    # Handle missing or "unknown" categories by replacing them with "Others"
    df["category"] = df["category"].replace("unknown", "Others")
    df["category"] = df["category"].fillna("Others")
    
    # Process payment_type to keep only the first value separated by "|"
    # Use str.split to separate values and extract the first element
    df["payment_type"] = df["payment_type"].astype(str).str.split("|").str[0]
    
    return df

def aggregate_city_sales(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    # 2. Group the dataset by "city" and calculate total sales (sum of "price")
    city_sales = df.groupby("city", as_index=False)["price"].sum()
    
    # Sort results in descending order of sales
    city_sales = city_sales.sort_values(by="price", ascending=False)
    
    # 3. Select only the top 10 cities by sales
    if top_n is not None:
        city_sales = city_sales.head(top_n)
        
    return city_sales

def create_city_sales_chart(city_sales: pd.DataFrame):
    # 4. Create a bar chart using Plotly Express
    # Use a professional color palette (Teal) to denote total sales volume
    fig = px.bar(
        city_sales,
        x="city",
        y="price",
        title="Top 10 Cities by Sales",
        labels={"city": "City", "price": "Total Sales"},
        color="price",
        color_continuous_scale=px.colors.sequential.Teal
    )
    
    # Rotate x-axis labels for readability and apply clean template
    fig.update_layout(
        xaxis_tickangle=-45,
        template="plotly_white",
        showlegend=False
    )
    
    return fig

# Streamlit application setup (7. Optional Streamlit wrapper)
def run_streamlit_app(file_path: str):
    import streamlit as st
    
    st.set_page_config(page_title="City Sales Analytics", layout="wide")
    st.title("Top 10 Cities by Sales Analysis")
    
    try:
        # Run preprocessing and aggregation
        df = load_and_preprocess_data(file_path)
        top_city_sales = aggregate_city_sales(df, top_n=10)
        
        # Create and display visualization
        fig = create_city_sales_chart(top_city_sales)
        st.plotly_chart(fig, use_container_width=True)
        
        # Optional: Show the underlying data table
        with st.expander("View Underlying Data"):
            st.dataframe(top_city_sales)
            
    except FileNotFoundError:
        st.error(f"Error: Dataset file '{file_path}' not found.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Standard python script execution
def run_standard_script(file_path: str):
    try:
        print("Loading and processing data for city analysis...")
        df = load_and_preprocess_data(file_path)
        top_city_sales = aggregate_city_sales(df, top_n=10)
        
        print("Generating visualization...")
        fig = create_city_sales_chart(top_city_sales)
        
        # 5. Display the chart
        fig.show()
        
    except FileNotFoundError:
        print(f"Error: Dataset file '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    dataset_path = "final_cleaned_data.csv"
    
    # Detect if script is being run by Streamlit
    is_streamlit = "streamlit" in sys.modules
    
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            is_streamlit = True
    except ImportError:
        pass
    
    if is_streamlit:
        run_streamlit_app(dataset_path)
    else:
        run_standard_script(dataset_path)
