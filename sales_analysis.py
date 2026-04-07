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

def aggregate_sales_data(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    # 2. Group the dataset by 'category' and calculate total sales (sum of 'price')
    category_sales = df.groupby("category", as_index=False)["price"].sum()
    
    # Sort results in descending order of sales
    category_sales = category_sales.sort_values(by="price", ascending=False)
    
    # 4. Show only top N categories for better clarity
    if top_n is not None:
        category_sales = category_sales.head(top_n)
        
    return category_sales

def create_sales_chart(category_sales: pd.DataFrame):
    # 3. Create a bar chart using Plotly Express
    # Use a professional color palette (Blues) to denote total sales volume
    fig = px.bar(
        category_sales,
        x="category",
        y="price",
        title="Category-wise Sales Distribution",
        labels={"category": "Category", "price": "Total Sales"},
        color="price",
        color_continuous_scale=px.colors.sequential.Blues
    )
    
    # Rotate x-axis labels for readability and apply clean template
    fig.update_layout(
        xaxis_tickangle=-45,
        template="plotly_white",
        showlegend=False
    )
    
    return fig

# Streamlit application setup
def run_streamlit_app(file_path: str):
    import streamlit as st
    
    st.set_page_config(page_title="Sales Analytics", layout="wide")
    st.title("Category-wise Sales Analysis")
    
    try:
        # Run preprocessing and aggregation
        df = load_and_preprocess_data(file_path)
        top_category_sales = aggregate_sales_data(df, top_n=10)
        
        # Create and display visualization
        fig = create_sales_chart(top_category_sales)
        st.plotly_chart(fig, use_container_width=True)
        
        # Optional: Show the underlying data table
        with st.expander("View Underlying Data"):
            st.dataframe(top_category_sales)
            
    except FileNotFoundError:
        st.error(f"Error: Dataset file '{file_path}' not found.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Standard python script execution
def run_standard_script(file_path: str):
    try:
        print("Loading and processing data...")
        df = load_and_preprocess_data(file_path)
        top_category_sales = aggregate_sales_data(df, top_n=10)
        
        print("Generating visualization...")
        fig = create_sales_chart(top_category_sales)
        
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
    
    # Streamlit execution wrapper
    try:
        # Check standard Streamlit runtime execution context
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            is_streamlit = True
    except ImportError:
        pass
    
    if is_streamlit:
        run_streamlit_app(dataset_path)
    else:
        run_standard_script(dataset_path)
