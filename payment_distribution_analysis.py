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

def aggregate_payment_data(df: pd.DataFrame) -> pd.DataFrame:
    # 2. Count the number of occurrences of each payment_type
    payment_counts = df["payment_type"].value_counts().reset_index()
    payment_counts.columns = ["payment_type", "count"]
    
    return payment_counts

def create_payment_chart(payment_counts: pd.DataFrame):
    # 3. Create a pie chart using Plotly Express
    # Display percentage labels and use a professional color palette
    fig = px.pie(
        payment_counts,
        names="payment_type",
        values="count",
        title="Payment Method Distribution",
        color_discrete_sequence=px.colors.sequential.Teal
    )
    
    # Update traces to format percentage labels cleanly into the pie slices
    fig.update_traces(textposition="inside", textinfo="percent+label")
    
    return fig

# Streamlit application setup (6. Optional Streamlit wrapper)
def run_streamlit_app(file_path: str):
    import streamlit as st
    
    st.set_page_config(page_title="Payment Distribution Analytics", layout="wide")
    st.title("Payment Method Distribution")
    
    try:
        # Run preprocessing and aggregation
        df = load_and_preprocess_data(file_path)
        payment_counts = aggregate_payment_data(df)
        
        # Create and display visualization
        fig = create_payment_chart(payment_counts)
        st.plotly_chart(fig, use_container_width=True)
        
        # Optional: Show the underlying data table
        with st.expander("View Underlying Data"):
            st.dataframe(payment_counts)
            
    except FileNotFoundError:
        st.error(f"Error: Dataset file '{file_path}' not found.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Standard python script execution
def run_standard_script(file_path: str):
    try:
        print("Loading and processing data for payment method analysis...")
        df = load_and_preprocess_data(file_path)
        payment_counts = aggregate_payment_data(df)
        
        print("Generating visualization...")
        fig = create_payment_chart(payment_counts)
        
        # 4. Display the chart
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
