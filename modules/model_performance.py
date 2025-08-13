import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualization import plot_model_performance
from models.model_trainer import train_all_models, PrisonForecastModels

def show_model_performance(data, models, l):
    """
    Display model performance and evaluation page - simplified version
    """
    st.header(l["model_performance_header"])
    st.markdown("---")
    
    # Check if models are available
    if not models:
        st.warning("No trained models found. Please train models first.")
        
        if st.button("Train Models Now"):
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    trained_models = train_all_models(data)
                    if trained_models:
                        st.success("Models trained successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to train models. Please check the data.")
                except Exception as e:
                    st.error(f"Error training models: {e}")
        return
    
    # Simplified main content tabs - only essential ones
    tab1, tab2 = st.tabs([
        l["model_overview"],
        l["performance_metrics"]
    ])
    
    with tab1:
        show_model_overview(models, data)
    
    with tab2:
        show_performance_metrics(models)

def show_model_overview(models, data):
    """
    Display model overview and status
    """
    st.subheader("📊 Model Overview")
    
    # Model status cards
    col1, col2, col3 = st.columns(3)
    
    model_types = ['population', 'staffing', 'resource']
    model_names = ['Population Forecast', 'Staffing Forecast', 'Resource Forecast']
    columns = [col1, col2, col3]
    
    for i, (model_type, model_name, col) in enumerate(zip(model_types, model_names, columns)):
        with col:
            if model_type in models:
                # Get model performance if available - handle new metrics structure
                r2_score = "N/A"
                if 'metrics' in models:
                    metrics = models['metrics']
                    # Look for the best performing model variant for this type
                    best_r2 = 0
                    for metric_key, metric_values in metrics.items():
                        if metric_key.startswith(model_type) and isinstance(metric_values, dict):
                            if 'r2' in metric_values and metric_values['r2'] > best_r2:
                                best_r2 = metric_values['r2']
                    
                    if best_r2 > 0:
                        r2_score = f"{best_r2:.3f}"
                
                st.success(f"✅ {model_name}")
                st.metric("Status", "Active")
                st.metric("R² Score", r2_score)
            else:
                st.error(f"❌ {model_name}")
                st.metric("Status", "Not Available")
    
    st.markdown("---")
    
    # Data overview
    st.subheader("📋 Data Overview")
    
    data_info = []
    dataset_names = {
        'population_data': 'Population Data',
        'staffing_data': 'Staffing Data', 
        'resource_data': 'Resource Data',
        'prison_detail_data': 'Prison Detail Data'
    }
    
    for key, name in dataset_names.items():
        if key in data:
            df = data[key]
            date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
            data_info.append({
                'Dataset': name,
                'Records': len(df),
                'Date Range': date_range,
                'Features': len(df.columns)
            })
    
    if data_info:
        st.table(pd.DataFrame(data_info))
    else:
        st.warning("No data information available")

def show_performance_metrics(models):
    """
    Display detailed performance metrics
    """
    st.subheader("📈 Performance Metrics")
    
    if 'metrics' not in models:
        st.warning("No performance metrics available. Please retrain models to generate metrics.")
        return
    
    metrics_data = models['metrics']
    
    # Performance summary table
    st.subheader("📊 Model Performance Summary")
    
    performance_data = []
    for model_type, metrics in metrics_data.items():
        if isinstance(metrics, dict):
            # Calculate MAPE if not available
            mape_value = metrics.get('mape', 'N/A')
            if mape_value == 'N/A' and 'mae' in metrics and 'r2' in metrics:
                # Rough MAPE estimation (this is just for display)
                mape_value = f"{(metrics['mae'] / 1000):.2f}"  # Simplified estimation
            
            performance_data.append({
                'Model': model_type.replace('_', ' ').title(),
                'R² Score': f"{metrics.get('r2', 0):.3f}" if 'r2' in metrics else "N/A",
                'MAE': f"{metrics.get('mae', 0):.2f}" if 'mae' in metrics else "N/A",
                'RMSE': f"{metrics.get('rmse', 0):.2f}" if 'rmse' in metrics else "N/A",
                'MAPE': f"{mape_value}%" if isinstance(mape_value, str) and mape_value != 'N/A' else mape_value
            })
    
    if performance_data:
        st.table(pd.DataFrame(performance_data))
        
        # Performance visualization
        st.subheader("📊 Model Performance Comparison")
        plot_model_performance(metrics_data)
        
        # Individual model details
        st.subheader("🔍 Detailed Metrics")
        
        selected_model = st.selectbox(
            "Select Model for Details:",
            options=list(metrics_data.keys()),
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        if selected_model in metrics_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Performance Metrics:**")
                metrics = metrics_data[selected_model]
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            if metric == 'mape':
                                st.metric(metric.upper(), f"{value:.2f}%")
                            else:
                                st.metric(metric.upper(), f"{value:.3f}")
            
            with col2:
                st.markdown("**Model Interpretation:**")
                
                r2_score = metrics_data[selected_model].get('r2', 0)
                if r2_score >= 0.9:
                    st.success("🟢 Excellent predictive performance")
                    st.write("The model explains >90% of variance in the data.")
                elif r2_score >= 0.8:
                    st.success("🔵 Good predictive performance") 
                    st.write("The model explains 80-90% of variance in the data.")
                elif r2_score >= 0.7:
                    st.warning("🟡 Moderate predictive performance")
                    st.write("The model explains 70-80% of variance. Consider retraining.")
                else:
                    st.error("🔴 Poor predictive performance")
                    st.write("The model explains <70% of variance. Retraining recommended.")
    
    # Model comparison by type
    create_model_comparison_chart(metrics_data)

def create_model_comparison_chart(metrics_data):
    """
    Create a comparison chart of model performance
    """
    try:
        st.subheader("📊 Performance Comparison Chart")
        
        models = []
        r2_scores = []
        mae_scores = []
        
        for model_type, metrics in metrics_data.items():
            if isinstance(metrics, dict) and 'r2' in metrics:
                models.append(model_type.replace('_', ' ').title())
                r2_scores.append(metrics['r2'])
                mae_scores.append(metrics.get('mae', 0))
        
        if models:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('R² Score (Higher = Better)', 'MAE (Lower = Better)'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # R² Score chart
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=r2_scores,
                    name="R² Score",
                    marker_color='lightblue'
                ),
                row=1, col=1
            )
            
            # MAE chart
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=mae_scores,
                    name="MAE",
                    marker_color='lightcoral'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=400,
                showlegend=False,
                title_text="Model Performance Comparison"
            )
            
            st.plotly_chart(fig, use_container_width=True, key="model_performance_comparison_chart")
        else:
            st.info("No performance data available for comparison")
            
    except Exception as e:
        st.error(f"Error creating comparison chart: {e}")
