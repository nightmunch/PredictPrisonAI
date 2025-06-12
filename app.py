import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utility modules
from utils.data_utils import load_or_generate_data, load_models
from utils.visualization import plot_overview_metrics

# Configure the page
st.set_page_config(
    page_title="Malaysia Prison Predictive Planning",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🏢 Malaysia Prison Predictive Planning System")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select a page:",
        [
            "Dashboard Overview",
            "Population Forecast",
            "Staffing Forecast",
            "Resource Forecast",
            "Model Performance"
        ]
    )
    
    # Load data and models
    try:
        data = load_or_generate_data()
        models = load_models()
        
        if data is None:
            st.error("Failed to load data. Please check the data generation process.")
            return
            
    except Exception as e:
        st.error(f"Error loading data or models: {str(e)}")
        st.info("Please ensure the data generation notebook has been run and models are trained.")
        return
    
    # Route to different pages
    if page == "Dashboard Overview":
        show_dashboard_overview(data, models)
    elif page == "Population Forecast":
        from modules.population_forecast import show_population_forecast
        show_population_forecast(data, models)
    elif page == "Staffing Forecast":
        from modules.staffing_forecast import show_staffing_forecast
        show_staffing_forecast(data, models)
    elif page == "Resource Forecast":
        from modules.resource_forecast import show_resource_forecast
        show_resource_forecast(data, models)
    elif page == "Model Performance":
        from modules.model_performance import show_model_performance
        show_model_performance(data, models)

def show_dashboard_overview(data, models):
    """Display the main dashboard overview"""
    st.header("📊 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if 'population_data' in data:
        current_population = data['population_data']['total_prisoners'].iloc[-1]
        col1.metric("Current Prison Population", f"{current_population:,.0f}")
    
    if 'staffing_data' in data:
        current_staff = data['staffing_data']['total_staff'].iloc[-1]
        col2.metric("Current Staff Count", f"{current_staff:,.0f}")
    
    if 'resource_data' in data:
        capacity_utilization = data['resource_data']['capacity_utilization'].iloc[-1]
        col3.metric("Capacity Utilization", f"{capacity_utilization:.1f}%")
    
    if 'population_data' in data:
        avg_sentence = data['population_data']['avg_sentence_months'].iloc[-1]
        col4.metric("Avg Sentence Length", f"{avg_sentence:.1f} months")
    
    st.markdown("---")
    
    # Overview charts
    st.subheader("📈 Historical Trends")
    
    if data:
        plot_overview_metrics(data)
    
    # System information
    st.markdown("---")
    st.subheader("ℹ️ System Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.info("""
        **Data Coverage:**
        - Historical data: 5 years
        - Forecast horizon: 2 years
        - Update frequency: Monthly
        """)
    
    with info_col2:
        if models:
            model_status = "✅ Models loaded and ready"
            st.success(model_status)
        else:
            model_status = "❌ Models not available"
            st.error(model_status)
    
    # Quick actions
    st.markdown("---")
    st.subheader("🚀 Quick Actions")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if st.button("Generate New Forecast"):
            st.info("Navigate to specific forecast pages to generate predictions")
    
    with action_col2:
        if st.button("Export Current Data"):
            if data:
                st.download_button(
                    label="Download Population Data",
                    data=data['population_data'].to_csv(index=False),
                    file_name=f"prison_population_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    with action_col3:
        if st.button("Model Status Check"):
            if models:
                st.success("All models are operational")
                for model_name in models.keys():
                    st.write(f"✅ {model_name.replace('_', ' ').title()}")
            else:
                st.warning("Some models may need retraining")

if __name__ == "__main__":
    main()
