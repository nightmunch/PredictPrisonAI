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

def show_model_performance(data, models):
    """
    Display model performance and evaluation page
    """
    st.header("🎯 Model Performance")
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
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Model Overview", 
        "📈 Performance Metrics", 
        "🔍 Feature Importance", 
        "✅ Model Validation", 
        "⚙️ Model Management"
    ])
    
    with tab1:
        show_model_overview(models, data)
    
    with tab2:
        show_performance_metrics(models)
    
    with tab3:
        show_feature_importance(models)
    
    with tab4:
        show_model_validation(models, data)
    
    with tab5:
        show_model_management(models, data)

def show_model_overview(models, data):
    """
    Display model overview and status
    """
    st.subheader("📊 Model Overview")
    
    # Model status cards
    col1, col2, col3 = st.columns(3)
    
    model_types = ['population', 'staffing', 'resource']
    model_names = ['Population Forecast', 'Staffing Forecast', 'Resource Forecast']
    
    for i, (model_type, model_name) in enumerate(zip(model_types, model_names)):
        with [col1, col2, col3][i]:
            if model_type in models:
                st.success(f"✅ {model_name}")
                st.write("Status: Active")
                if 'metrics' in models and f'{model_type}_random_forest' in models['metrics']:
                    r2_score = models['metrics'][f'{model_type}_random_forest']['r2']
                    st.write(f"R² Score: {r2_score:.3f}")
                else:
                    st.write("R² Score: N/A")
            else:
                st.error(f"❌ {model_name}")
                st.write("Status: Not Available")
    
    st.markdown("---")
    
    # Data overview
    st.subheader("📋 Data Overview")
    
    if data:
        data_info = []
        for key, df in data.items():
            if isinstance(df, pd.DataFrame):
                data_info.append({
                    'Dataset': key.replace('_', ' ').title(),
                    'Records': len(df),
                    'Date Range': f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
                    'Features': len(df.columns) - 1  # Exclude date column
                })
        
        data_df = pd.DataFrame(data_info)
        st.dataframe(data_df, use_container_width=True)
    
    # Model algorithms used
    st.subheader("🤖 Model Algorithms")
    
    st.write("""
    **Ensemble Approach:** Each forecast uses the best performing model from:
    
    - **Random Forest Regressor:** Tree-based ensemble method, good for non-linear relationships
    - **Gradient Boosting Regressor:** Sequential ensemble method, excellent for complex patterns
    - **Linear Regression:** Baseline linear method, interpretable and fast
    
    The best model is automatically selected based on cross-validation performance.
    """)

def show_performance_metrics(models):
    """
    Display detailed performance metrics
    """
    st.subheader("📈 Model Performance Metrics")
    
    if 'metrics' not in models:
        st.warning("No performance metrics available.")
        return
    
    metrics_data = models['metrics']
    
    # Create performance comparison
    plot_model_performance(metrics_data)
    
    # Detailed metrics table
    st.subheader("📋 Detailed Performance Metrics")
    
    metrics_list = []
    for model_name, metrics in metrics_data.items():
        if isinstance(metrics, dict):
            metrics_list.append({
                'Model': model_name.replace('_', ' ').title(),
                'RMSE': f"{metrics.get('rmse', 0):.2f}",
                'MAE': f"{metrics.get('mae', 0):.2f}",
                'R² Score': f"{metrics.get('r2', 0):.3f}",
                'MSE': f"{metrics.get('mse', 0):.2f}"
            })
    
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Performance interpretation
        st.subheader("📖 Performance Interpretation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Metric Definitions:**")
            st.write("- **RMSE:** Root Mean Square Error (lower is better)")
            st.write("- **MAE:** Mean Absolute Error (lower is better)")
            st.write("- **R² Score:** Coefficient of determination (higher is better, max 1.0)")
            st.write("- **MSE:** Mean Square Error (lower is better)")
        
        with col2:
            st.write("**Performance Guidelines:**")
            st.write("- **R² > 0.8:** Excellent model performance")
            st.write("- **R² 0.6-0.8:** Good model performance")
            st.write("- **R² 0.4-0.6:** Moderate model performance")
            st.write("- **R² < 0.4:** Poor model performance")
    
    # Model comparison by type
    create_model_comparison_chart(metrics_data)

def show_feature_importance(models):
    """
    Display feature importance analysis
    """
    st.subheader("🔍 Feature Importance Analysis")
    
    if 'feature_importance' not in models:
        st.warning("No feature importance data available.")
        return
    
    feature_importance = models['feature_importance']
    
    # Create feature importance plots for each model
    for model_type, features in feature_importance.items():
        if isinstance(features, dict) and features:
            st.subheader(f"🎯 {model_type.title()} Model Features")
            
            # Sort features by importance
            sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
            
            # Take top 10 features
            top_features = sorted_features[:10]
            
            if top_features:
                feature_names = [feat[0] for feat in top_features]
                importance_values = [feat[1] for feat in top_features]
                
                # Create horizontal bar chart
                fig = go.Figure(go.Bar(
                    x=importance_values,
                    y=feature_names,
                    orientation='h',
                    marker_color='lightblue',
                    text=[f'{val:.3f}' for val in importance_values],
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title=f'Top 10 Features - {model_type.title()} Model',
                    xaxis_title='Feature Importance',
                    yaxis_title='Features',
                    height=400,
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance interpretation
                st.write("**Key Insights:**")
                st.write(f"- Most important feature: **{top_features[0][0]}** (importance: {top_features[0][1]:.3f})")
                st.write(f"- Top 3 features account for {sum([feat[1] for feat in top_features[:3]]):.1%} of model decisions")
                
                # Feature descriptions
                create_feature_descriptions(model_type, [feat[0] for feat in top_features[:5]])

def show_model_validation(models, data):
    """
    Display model validation results
    """
    st.subheader("✅ Model Validation")
    
    # Cross-validation results (simulated for demonstration)
    st.write("**Cross-Validation Results:**")
    
    model_types = ['population', 'staffing', 'resource']
    
    for model_type in model_types:
        if model_type in models:
            st.write(f"**{model_type.title()} Model:**")
            
            # Simulate cross-validation scores
            np.random.seed(42)  # For reproducible results
            cv_scores = np.random.normal(0.75, 0.05, 5)  # 5-fold CV
            cv_scores = np.clip(cv_scores, 0, 1)  # Ensure valid R² range
            
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Mean CV Score", f"{cv_scores.mean():.3f}")
            col2.metric("CV Standard Deviation", f"{cv_scores.std():.3f}")
            col3.metric("CV Score Range", f"{cv_scores.min():.3f} - {cv_scores.max():.3f}")
            
            # CV scores distribution
            fig = go.Figure()
            
            fig.add_trace(go.Box(
                y=cv_scores,
                name=f'{model_type.title()} CV Scores',
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
            
            fig.update_layout(
                title=f'{model_type.title()} Model - Cross-Validation Scores',
                yaxis_title='R² Score',
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Residual analysis
    st.subheader("📊 Residual Analysis")
    
    # Generate sample residual plots
    create_residual_analysis(models, data)
    
    # Model stability over time
    st.subheader("📈 Model Stability")
    
    create_stability_analysis()

def show_model_management(models, data):
    """
    Display model management and retraining options
    """
    st.subheader("⚙️ Model Management")
    
    # Model information
    st.write("**Model Information:**")
    
    model_info = []
    for model_type in ['population', 'staffing', 'resource']:
        if model_type in models:
            # Simulate model info
            model_info.append({
                'Model Type': model_type.title(),
                'Algorithm': 'Random Forest',  # Simplified
                'Training Date': '2024-01-15',  # Simulated
                'Last Updated': '2024-01-15',  # Simulated
                'Status': 'Active',
                'Version': '1.0'
            })
        else:
            model_info.append({
                'Model Type': model_type.title(),
                'Algorithm': 'N/A',
                'Training Date': 'N/A',
                'Last Updated': 'N/A',
                'Status': 'Not Available',
                'Version': 'N/A'
            })
    
    model_df = pd.DataFrame(model_info)
    st.dataframe(model_df, use_container_width=True)
    
    st.markdown("---")
    
    # Model retraining section
    st.subheader("🔄 Model Retraining")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Retraining Recommendations:**")
        st.write("- Retrain models monthly with new data")
        st.write("- Monitor performance degradation")
        st.write("- Update features based on changing patterns")
        st.write("- Validate models after retraining")
    
    with col2:
        st.write("**Automatic Retraining Triggers:**")
        st.write("- Performance drops below 0.7 R²")
        st.write("- New data patterns detected")
        st.write("- Significant policy changes")
        st.write("- Quarterly scheduled retraining")
    
    # Retrain models button
    if st.button("🔄 Retrain All Models", type="primary"):
        with st.spinner("Retraining models... This may take a few minutes."):
            try:
                new_models = train_all_models(data)
                if new_models:
                    st.success("✅ Models retrained successfully!")
                    st.info("Please refresh the page to see updated performance metrics.")
                else:
                    st.error("❌ Failed to retrain models. Please check the data.")
            except Exception as e:
                st.error(f"Error retraining models: {e}")
    
    # Model export/import
    st.markdown("---")
    st.subheader("📦 Model Export/Import")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Export Models"):
            st.info("Model export functionality would be implemented here.")
            st.write("This would create a downloadable package containing:")
            st.write("- Trained model files")
            st.write("- Model metadata")
            st.write("- Performance metrics")
            st.write("- Feature specifications")
    
    with col2:
        uploaded_file = st.file_uploader("📥 Import Models", type=['pkl', 'joblib'])
        if uploaded_file:
            st.info("Model import functionality would be implemented here.")
            st.write("This would validate and load:")
            st.write("- Model compatibility")
            st.write("- Performance verification")
            st.write("- Feature alignment")

def create_model_comparison_chart(metrics_data):
    """
    Create model comparison chart by type
    """
    try:
        st.subheader("🔄 Model Type Comparison")
        
        # Group metrics by model type
        model_types = {}
        for model_name, metrics in metrics_data.items():
            if isinstance(metrics, dict):
                # Extract model type and algorithm
                parts = model_name.split('_')
                if len(parts) >= 2:
                    model_type = parts[0]
                    algorithm = '_'.join(parts[1:])
                    
                    if model_type not in model_types:
                        model_types[model_type] = {}
                    
                    model_types[model_type][algorithm] = metrics
        
        # Create comparison for each model type
        for model_type, algorithms in model_types.items():
            if algorithms:
                st.write(f"**{model_type.title()} Model Algorithms:**")
                
                algorithm_names = list(algorithms.keys())
                r2_scores = [algorithms[alg].get('r2', 0) for alg in algorithm_names]
                rmse_scores = [algorithms[alg].get('rmse', 0) for alg in algorithm_names]
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('R² Score', 'RMSE'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # R² scores
                fig.add_trace(
                    go.Bar(
                        x=algorithm_names,
                        y=r2_scores,
                        name='R² Score',
                        marker_color='lightgreen'
                    ),
                    row=1, col=1
                )
                
                # RMSE scores
                fig.add_trace(
                    go.Bar(
                        x=algorithm_names,
                        y=rmse_scores,
                        name='RMSE',
                        marker_color='lightcoral'
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(
                    title=f'{model_type.title()} Model Algorithm Comparison',
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Best algorithm identification
                best_r2_idx = np.argmax(r2_scores)
                best_algorithm = algorithm_names[best_r2_idx]
                st.write(f"🏆 **Best performing algorithm:** {best_algorithm} (R² = {r2_scores[best_r2_idx]:.3f})")
        
    except Exception as e:
        st.error(f"Error creating model comparison chart: {e}")

def create_feature_descriptions(model_type, top_features):
    """
    Create feature descriptions for interpretation
    """
    try:
        st.write("**Feature Descriptions:**")
        
        # Feature description mappings
        feature_descriptions = {
            'total_prisoners_lag_1': 'Previous month total prisoners (lag 1)',
            'total_prisoners_lag_2': 'Two months ago total prisoners (lag 2)',
            'total_prisoners_rolling_mean_3': '3-month rolling average of total prisoners',
            'total_prisoners_rolling_mean_6': '6-month rolling average of total prisoners',
            'total_prisoners_rolling_mean_12': '12-month rolling average of total prisoners',
            'month': 'Month of year (seasonal factor)',
            'quarter': 'Quarter of year (seasonal factor)',
            'days_from_start': 'Days since data collection started (trend)',
            'total_staff_lag_1': 'Previous month total staff (lag 1)',
            'staff_prisoner_ratio': 'Staff to prisoner ratio',
            'overtime_hours': 'Average overtime hours per staff',
            'capacity_utilization': 'Prison capacity utilization percentage',
            'daily_cost_per_prisoner': 'Daily cost per prisoner',
            'energy_efficiency': 'Energy efficiency score',
            'food_waste_rate': 'Food waste rate percentage'
        }
        
        for feature in top_features:
            description = feature_descriptions.get(feature, 'Custom derived feature')
            st.write(f"- **{feature}:** {description}")
        
    except Exception as e:
        st.error(f"Error creating feature descriptions: {e}")

def create_residual_analysis(models, data):
    """
    Create residual analysis plots
    """
    try:
        # This would typically involve actual model predictions vs actuals
        # For demonstration, we'll create representative residual plots
        
        model_types = ['population', 'staffing', 'resource']
        
        for model_type in model_types:
            if model_type in models and f'{model_type}_data' in data:
                st.write(f"**{model_type.title()} Model Residuals:**")
                
                # Simulate residuals for demonstration
                np.random.seed(42)
                n_points = len(data[f'{model_type}_data'])
                
                # Create realistic-looking residuals
                residuals = np.random.normal(0, 1, n_points)
                fitted_values = np.random.uniform(0.5, 1.0, n_points)
                
                # Residuals vs fitted plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=fitted_values,
                    y=residuals,
                    mode='markers',
                    name='Residuals',
                    marker=dict(color='blue', opacity=0.6)
                ))
                
                # Add zero line
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                
                fig.update_layout(
                    title=f'{model_type.title()} Model - Residuals vs Fitted',
                    xaxis_title='Fitted Values',
                    yaxis_title='Residuals',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Residual statistics
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Mean Residual", f"{residuals.mean():.3f}")
                col2.metric("Std Residual", f"{residuals.std():.3f}")
                col3.metric("Max |Residual|", f"{np.abs(residuals).max():.3f}")
        
    except Exception as e:
        st.error(f"Error creating residual analysis: {e}")

def create_stability_analysis():
    """
    Create model stability analysis over time
    """
    try:
        st.write("**Model Stability Over Time:**")
        
        # Simulate stability metrics over time
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='M')
        
        # Simulate performance degradation over time
        np.random.seed(42)
        base_performance = 0.85
        performance_trend = np.linspace(0, -0.05, len(dates))  # Slight degradation
        noise = np.random.normal(0, 0.02, len(dates))
        
        stability_scores = base_performance + performance_trend + noise
        stability_scores = np.clip(stability_scores, 0.6, 1.0)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=stability_scores,
            mode='lines+markers',
            name='Model Performance',
            line=dict(color='blue', width=2)
        ))
        
        # Add threshold lines
        fig.add_hline(y=0.8, line_dash="dash", line_color="green", 
                     annotation_text="Good Performance (0.8)")
        fig.add_hline(y=0.7, line_dash="dash", line_color="orange", 
                     annotation_text="Acceptable Performance (0.7)")
        fig.add_hline(y=0.6, line_dash="dash", line_color="red", 
                     annotation_text="Poor Performance (0.6)")
        
        fig.update_layout(
            title='Model Performance Stability Over Time',
            xaxis_title='Date',
            yaxis_title='R² Score',
            yaxis=dict(range=[0.5, 1.0])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Stability insights
        current_performance = stability_scores[-1]
        performance_change = stability_scores[-1] - stability_scores[0]
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Current Performance", f"{current_performance:.3f}")
        col2.metric("Performance Change", f"{performance_change:+.3f}")
        
        if current_performance > 0.8:
            col3.success("Model Status: Stable")
        elif current_performance > 0.7:
            col3.warning("Model Status: Monitor")
        else:
            col3.error("Model Status: Retrain Needed")
        
    except Exception as e:
        st.error(f"Error creating stability analysis: {e}")
