import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_overview_metrics(data):
    """
    Create overview dashboard with key metrics
    """
    try:
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Prison Population Trend', 'Capacity Utilization', 
                          'Staff Levels', 'Monthly Costs'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Population trend
        if 'population_data' in data:
            pop_data = data['population_data']
            fig.add_trace(
                go.Scatter(
                    x=pop_data['date'],
                    y=pop_data['total_prisoners'],
                    mode='lines',
                    name='Total Prisoners',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
        
        # Capacity utilization
        if 'resource_data' in data:
            resource_data = data['resource_data']
            fig.add_trace(
                go.Scatter(
                    x=resource_data['date'],
                    y=resource_data['capacity_utilization'],
                    mode='lines',
                    name='Capacity %',
                    line=dict(color='red', width=2)
                ),
                row=1, col=2
            )
            
            # Add 100% capacity line
            fig.add_hline(
                y=100, line_dash="dash", line_color="red",
                annotation_text="Full Capacity",
                row=1, col=2
            )
        
        # Staff levels
        if 'staffing_data' in data:
            staff_data = data['staffing_data']
            fig.add_trace(
                go.Scatter(
                    x=staff_data['date'],
                    y=staff_data['total_staff'],
                    mode='lines',
                    name='Total Staff',
                    line=dict(color='green', width=2)
                ),
                row=2, col=1
            )
        
        # Monthly costs
        if 'resource_data' in data:
            fig.add_trace(
                go.Scatter(
                    x=resource_data['date'],
                    y=resource_data['total_monthly_cost'] / 1000000,  # Convert to millions
                    mode='lines',
                    name='Cost (Million MYR)',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Prison System Overview Dashboard"
        )
        
        # Update x-axis labels
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Number of Prisoners", row=1, col=1)
        fig.update_yaxes(title_text="Utilization %", row=1, col=2)
        fig.update_yaxes(title_text="Number of Staff", row=2, col=1)
        fig.update_yaxes(title_text="Cost (Million MYR)", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating overview plot: {e}")

def plot_population_forecast(historical_data, forecast_data, forecast_dates):
    """
    Create population forecast visualization
    """
    try:
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['total_prisoners'],
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Forecast data
        if forecast_data is not None and len(forecast_data) > 0:
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_data,
                mode='lines',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title='Prison Population Forecast',
            xaxis_title='Date',
            yaxis_title='Number of Prisoners',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating population forecast plot: {e}")

def plot_staffing_breakdown(staffing_data):
    """
    Create staffing breakdown visualization
    """
    try:
        # Get latest data
        latest_data = staffing_data.iloc[-1]
        
        # Create pie chart for staff breakdown
        fig = go.Figure(data=[go.Pie(
            labels=['Security Staff', 'Admin Staff', 'Medical Staff', 'Other Staff'],
            values=[
                latest_data['security_staff'],
                latest_data['admin_staff'],
                latest_data['medical_staff'],
                latest_data['other_staff']
            ],
            hole=0.3
        )])
        
        fig.update_layout(
            title='Current Staff Distribution',
            annotations=[dict(text='Staff', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating staffing breakdown plot: {e}")

def plot_cost_breakdown(resource_data):
    """
    Create cost breakdown visualization
    """
    try:
        # Get latest data
        latest_data = resource_data.iloc[-1]
        
        # Create pie chart for cost breakdown
        fig = go.Figure(data=[go.Pie(
            labels=['Food Cost', 'Medical Cost', 'Utility Cost', 'Other Cost'],
            values=[
                latest_data['monthly_food_cost'],
                latest_data['monthly_medical_cost'],
                latest_data['monthly_utility_cost'],
                latest_data['monthly_other_cost']
            ],
            hole=0.3
        )])
        
        fig.update_layout(
            title='Monthly Cost Distribution (MYR)',
            annotations=[dict(text='Costs', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating cost breakdown plot: {e}")

def plot_trend_analysis(data, column, title):
    """
    Create trend analysis plot
    """
    try:
        fig = go.Figure()
        
        # Add actual data
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=data[column],
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        
        # Add trend line
        x_numeric = np.arange(len(data))
        z = np.polyfit(x_numeric, data[column], 1)
        p = np.poly1d(z)
        
        fig.add_trace(go.Scatter(
            x=data['date'],
            y=p(x_numeric),
            mode='lines',
            name='Trend',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title=column.replace('_', ' ').title(),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating trend analysis plot: {e}")

def plot_correlation_matrix(data, columns):
    """
    Create correlation matrix heatmap
    """
    try:
        # Calculate correlation matrix
        corr_matrix = data[columns].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Correlation Matrix',
            xaxis_title='Variables',
            yaxis_title='Variables'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating correlation matrix: {e}")

def plot_seasonal_decomposition(data, column):
    """
    Create seasonal decomposition plot
    """
    try:
        # Simple seasonal decomposition visualization
        data_copy = data.copy()
        data_copy['year'] = data_copy['date'].dt.year
        data_copy['month'] = data_copy['date'].dt.month
        
        # Calculate monthly averages
        monthly_avg = data_copy.groupby('month')[column].mean()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=monthly_avg.index,
            y=monthly_avg.values,
            name='Monthly Average',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title=f'Seasonal Pattern - {column.replace("_", " ").title()}',
            xaxis_title='Month',
            yaxis_title=column.replace('_', ' ').title(),
            xaxis=dict(tickmode='array', tickvals=list(range(1, 13)))
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating seasonal decomposition plot: {e}")

def plot_model_performance(metrics_data):
    """
    Create model performance visualization
    """
    try:
        if not metrics_data:
            st.warning("No model performance data available")
            return
        
        # Extract model names and metrics
        models = []
        rmse_values = []
        r2_values = []
        
        for model_name, metrics in metrics_data.items():
            if isinstance(metrics, dict) and 'rmse' in metrics:
                models.append(model_name.replace('_', ' ').title())
                rmse_values.append(metrics['rmse'])
                r2_values.append(metrics.get('r2', 0))
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('RMSE (Lower is Better)', 'R² Score (Higher is Better)')
        )
        
        # RMSE plot
        fig.add_trace(
            go.Bar(x=models, y=rmse_values, name='RMSE', marker_color='red'),
            row=1, col=1
        )
        
        # R² plot
        fig.add_trace(
            go.Bar(x=models, y=r2_values, name='R²', marker_color='green'),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Model Performance Comparison',
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating model performance plot: {e}")

def plot_forecast_comparison(historical_data, forecasts, forecast_dates, metric_name):
    """
    Create forecast comparison plot with confidence intervals
    """
    try:
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data[metric_name],
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Forecast
        if forecasts is not None and len(forecasts) > 0:
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecasts,
                mode='lines',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Add confidence intervals (simple approximation)
            upper_bound = np.array(forecasts) * 1.1
            lower_bound = np.array(forecasts) * 0.9
            
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=upper_bound,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=lower_bound,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                name='Confidence Interval',
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=f'{metric_name.replace("_", " ").title()} Forecast',
            xaxis_title='Date',
            yaxis_title=metric_name.replace('_', ' ').title(),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating forecast comparison plot: {e}")
