import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    # 1. Load the CSV file using pandas
    df = pd.read_csv(file_path)
    
    # Rename columns based on requirements
    column_mapping = {
        "order_purchase_timestamp": "order_date",
        "total_order_value": "price"
    }
    df = df.rename(columns=column_mapping)
    
    # Convert order_date to datetime format
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"])
    
    # Extract month and year from order_date into a new string column "year_month"
    df["year_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
    
    return df

def aggregate_monthly_sales(df: pd.DataFrame) -> pd.DataFrame:
    # 2. Group data by month (year-month timestamp) and calculate total monthly sales
    monthly_sales = df.groupby("year_month", as_index=False)["price"].sum()
    monthly_sales = monthly_sales.sort_values(by="year_month")
    
    # Remove rows where total sales is 0 or missing before plotting actual data
    monthly_sales = monthly_sales.dropna(subset=["price"])
    monthly_sales = monthly_sales[monthly_sales["price"] > 0]

    # Remove the last row to exclude incomplete or partially recorded last month data
    # The last month in the dataset may not have full data, causing an abnormal drop at the end
    monthly_sales = monthly_sales.iloc[:-1]

    # Reset index after filtering to ensure clean sequential indexing
    monthly_sales = monthly_sales.reset_index(drop=True)

    return monthly_sales

def train_and_predict(monthly_sales: pd.DataFrame, predict_months: int = 6):
    from sklearn.linear_model import LinearRegression
    
    # 3. Prepare Data for Model
    n_records = len(monthly_sales)
    
    # Month index (0, 1, 2, ...) mapped safely avoiding gaps if any
    X = np.arange(n_records).reshape(-1, 1)
    Y = monthly_sales["price"].values
    
    # 4. Model Building
    model = LinearRegression()
    model.fit(X, Y)
    
    # 5. Prediction
    # Extend predictions only AFTER the last actual data point (n_records to n_records+predict_months)
    future_X = np.arange(n_records, n_records + predict_months).reshape(-1, 1)
    predicted_Y = model.predict(future_X)
    
    # Generate future dates incrementally from the last known date
    last_date = monthly_sales["year_month"].iloc[-1]
    future_dates = pd.Series([last_date + pd.DateOffset(months=i) for i in range(1, predict_months + 1)])
    actual_dates = monthly_sales["year_month"].reset_index(drop=True)
    
    return actual_dates, Y, future_dates, predicted_Y

def create_prediction_chart(actual_dates, actual_Y, future_dates, predicted_Y):
    # 6. Visualization
    fig = go.Figure()
    
    # One line: Actual sales
    fig.add_trace(go.Scatter(
        x=actual_dates,
        y=actual_Y,
        mode="lines+markers",
        name="Actual Sales",
        line=dict(color="blue", width=2)
    ))
    
    # Extrapolate visual connection from the last actual point to clearly bridge the trajectory difference
    connection_x = pd.concat([pd.Series([actual_dates.iloc[-1]]), future_dates])
    connection_y = np.concatenate(([actual_Y[-1]], predicted_Y))
    
    # One line: Predicted sales starting strictly AFTER actual timeline with a dashed line
    fig.add_trace(go.Scatter(
        x=connection_x,
        y=connection_y,
        mode="lines",
        name="Predicted Sales Trend",
        line=dict(color="red", width=2, dash="dash")
    ))
    
    # Title and aesthetics
    fig.update_layout(
        title="Sales Prediction (Actual vs Predicted)",
        xaxis_title="Timeline (Month)",
        yaxis_title="Total Sales",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

# Streamlit application setup (9. Optional Streamlit wrapper)
def run_streamlit_app(file_path: str):
    import streamlit as st
    
    st.set_page_config(page_title="Sales Prediction Model", layout="wide")
    st.title("Monthly Sales Prediction (Regression)")
    
    try:
        # Data pipeline
        df = load_and_preprocess_data(file_path)
        monthly_sales = aggregate_monthly_sales(df)
        
        # Model, actual extraction and predictive indexing
        actual_dates, actual_Y, future_dates, predicted_Y = train_and_predict(monthly_sales, predict_months=6)
        
        # Plotting mechanics
        fig = create_prediction_chart(actual_dates, actual_Y, future_dates, predicted_Y)
        st.plotly_chart(fig, use_container_width=True)
        
    except ImportError as e:
         st.error(f"Missing dependency: {str(e)}. Please install it using: pip install scikit-learn")
    except FileNotFoundError:
        st.error(f"Error: Dataset file '{file_path}' not found.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Standard python script execution
def run_standard_script(file_path: str):
    try:
        print("Loading and processing data for prediction model...")
        df = load_and_preprocess_data(file_path)
        
        print("Aggregating normalized monthly sales...")
        monthly_sales = aggregate_monthly_sales(df)
        
        print("Training model and generating 6 month strictly future predictions...")
        actual_dates, actual_Y, future_dates, predicted_Y = train_and_predict(monthly_sales, predict_months=6)
        
        print("Generating separate-metric visualization...")
        fig = create_prediction_chart(actual_dates, actual_Y, future_dates, predicted_Y)
        
        # 7. Display the chart
        fig.show()
        
    except ImportError as e:
         print(f"Missing dependency: {str(e)}.\nRun 'pip install scikit-learn' to install it.")
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
