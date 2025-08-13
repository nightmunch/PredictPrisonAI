import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_overview_metrics(data, chart_key="executive_dashboard_overview", lang=None):
    """
    Create professional overview dashboard with enhanced metrics
    """
    try:
        # Set default language if not provided
        if lang is None:
            lang = {
                "prison_population_by_state": "🗺️ Prison Population by State",
                "date": "Date",
                "number_of_prisoners": "Number of Prisoners",
                "current_staff_distribution": "Current Staff Distribution",
                "monthly_cost_distribution": "Monthly Cost Distribution (MYR)"
            }
        
        # Professional color scheme
        colors = {
            'primary': '#1e3a8a',
            'secondary': '#3b82f6', 
            'accent': '#06b6d4',
            'success': '#10b981',
            'warning': '#f59e0b',
            'danger': '#ef4444'
        }
        
        # Create subplots with enhanced styling and translated titles
        subplot_titles = [
            '<b>Prison Population Trend</b>' if lang.get('date') == 'Date' else '<b>Trend Populasi Penjara</b>',
            '<b>Capacity Utilization Analysis</b>' if lang.get('date') == 'Date' else '<b>Analisis Penggunaan Kapasiti</b>',
            '<b>Staff Deployment Levels</b>' if lang.get('date') == 'Date' else '<b>Tahap Penempatan Kakitangan</b>',
            '<b>Monthly Operational Costs</b>' if lang.get('date') == 'Date' else '<b>Kos Operasi Bulanan</b>'
        ]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        # Population trend with enhanced styling
        if 'population_data' in data:
            pop_data = data['population_data']
            fig.add_trace(
                go.Scatter(
                    x=pop_data['date'],
                    y=pop_data['total_prisoners'],
                    mode='lines+markers',
                    name='Total Prisoners',
                    line=dict(color=colors['primary'], width=3, shape='spline'),
                    marker=dict(size=4, color=colors['primary']),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Prisoners:</b> %{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add trend line
            x_numeric = np.arange(len(pop_data))
            z = np.polyfit(x_numeric, pop_data['total_prisoners'], 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=pop_data['date'],
                    y=p(x_numeric),
                    mode='lines',
                    name='Trend',
                    line=dict(color=colors['secondary'], width=2, dash='dot'),
                    opacity=0.7,
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Capacity utilization with zones
        if 'resource_data' in data:
            resource_data = data['resource_data']
            
            # Add capacity zones background
            fig.add_hrect(
                y0=0, y1=70, 
                fillcolor=colors['success'], opacity=0.1,
                line_width=0, row=1, col=2
            )
            fig.add_hrect(
                y0=70, y1=85, 
                fillcolor=colors['warning'], opacity=0.1,
                line_width=0, row=1, col=2
            )
            fig.add_hrect(
                y0=85, y1=100, 
                fillcolor=colors['danger'], opacity=0.1,
                line_width=0, row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=resource_data['date'],
                    y=resource_data['capacity_utilization'],
                    mode='lines+markers',
                    name='Capacity %',
                    line=dict(color=colors['accent'], width=3),
                    marker=dict(size=4, color=colors['accent']),
                    fill='tonexty',
                    fillcolor=f'rgba(6, 182, 212, 0.1)',
                    hovertemplate='<b>Date:</b> %{x}<br><b>Utilization:</b> %{y:.1f}%<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Add capacity reference lines
            fig.add_hline(
                y=100, line_dash="dash", line_color=colors['danger'], line_width=2,
                annotation_text="Maximum Capacity", annotation_position="top right",
                row=1, col=2
            )
            fig.add_hline(
                y=85, line_dash="dot", line_color=colors['warning'], line_width=1,
                annotation_text="Overcrowding Alert", annotation_position="bottom right",
                row=1, col=2
            )
        
        # Staff levels with breakdown
        if 'staffing_data' in data:
            staff_data = data['staffing_data']
            fig.add_trace(
                go.Scatter(
                    x=staff_data['date'],
                    y=staff_data['total_staff'],
                    mode='lines+markers',
                    name='Total Staff',
                    line=dict(color=colors['success'], width=3),
                    marker=dict(size=4, color=colors['success']),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Staff:</b> %{y:,.0f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add available staff overlay
            fig.add_trace(
                go.Scatter(
                    x=staff_data['date'],
                    y=staff_data['available_staff'],
                    mode='lines',
                    name='Available Staff',
                    line=dict(color=colors['warning'], width=2, dash='dash'),
                    opacity=0.8,
                    showlegend=False,
                    hovertemplate='<b>Available:</b> %{y:,.0f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Monthly costs with trend analysis
        if 'resource_data' in data:
            cost_millions = resource_data['total_monthly_cost'] / 1000000
            fig.add_trace(
                go.Scatter(
                    x=resource_data['date'],
                    y=cost_millions,
                    mode='lines+markers',
                    name='Cost (Million MYR)',
                    line=dict(color=colors['primary'], width=3),
                    marker=dict(size=4, color=colors['primary']),
                    fill='tonexty',
                    fillcolor=f'rgba(30, 58, 138, 0.1)',
                    hovertemplate='<b>Date:</b> %{x}<br><b>Cost:</b> RM %{y:.1f}M<extra></extra>'
                ),
                row=2, col=2
            )
            
            # Add average cost line
            avg_cost = cost_millions.mean()
            fig.add_hline(
                y=avg_cost, line_dash="dot", line_color=colors['secondary'],
                annotation_text=f"Average: RM {avg_cost:.1f}M", 
                annotation_position="top left",
                row=2, col=2
            )
        
        # Enhanced layout with professional styling and translated title
        dashboard_title = "<b>Malaysian Prison System Executive Dashboard</b>" if lang.get('date') == 'Date' else "<b>Papan Pemuka Eksekutif Sistem Penjara Malaysia</b>"
        
        fig.update_layout(
            height=700,
            showlegend=False,
            title=dict(
                text=dashboard_title,
                font=dict(size=20, color=colors['primary']),
                x=0.5,
                y=0.98
            ),
            font=dict(size=12, family="Arial, sans-serif"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=80, b=50, l=50, r=50)
        )
        
        # Enhanced subplot titles
        for i, subplot_title in enumerate(['Prison Population Trend', 'Capacity Utilization Analysis', 
                                         'Staff Deployment Levels', 'Monthly Operational Costs']):
            fig.layout.annotations[i].update(
                font=dict(size=14, color=colors['primary'], family="Arial, sans-serif"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=colors['primary'],
                borderwidth=1
            )
        
        # Professional axis formatting
        fig.update_xaxes(
            showgrid=True, gridcolor='rgba(0,0,0,0.1)', gridwidth=1,
            showline=True, linecolor='rgba(0,0,0,0.3)',
            title_font=dict(size=12, color=colors['primary'])
        )
        fig.update_yaxes(
            showgrid=True, gridcolor='rgba(0,0,0,0.1)', gridwidth=1,
            showline=True, linecolor='rgba(0,0,0,0.3)',
            title_font=dict(size=12, color=colors['primary'])
        )
        
        # Update axis labels with professional formatting
        fig.update_xaxes(title_text="<b>Timeline</b>", row=2, col=1)
        fig.update_xaxes(title_text="<b>Timeline</b>", row=2, col=2)
        
        fig.update_yaxes(title_text="<b>Population Count</b>", row=1, col=1)
        fig.update_yaxes(title_text="<b>Utilization (%)</b>", row=1, col=2)
        fig.update_yaxes(title_text="<b>Staff Count</b>", row=2, col=1)
        fig.update_yaxes(title_text="<b>Cost (Million MYR)</b>", row=2, col=2)
        
        # Format y-axis with proper number formatting
        fig.update_yaxes(tickformat=',.0f', row=1, col=1)
        fig.update_yaxes(tickformat='.1f', row=1, col=2)
        fig.update_yaxes(tickformat=',.0f', row=2, col=1)
        fig.update_yaxes(tickformat='.1f', row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True, key=chart_key)
        
        # Add Malaysian context information
        if 'resource_data' in data and len(resource_data) > 0:
            min_date = resource_data['date'].min()
            max_date = resource_data['date'].max()
            if min_date.year <= 2020 and max_date.year >= 2020:
                st.info("📊 **Data Context:** Charts include COVID-19 pandemic impact period (2020-2021) with MCO effects on operations.")
        
    except Exception as e:
        st.error(f"Error creating overview dashboard: {e}")
        st.write("Debug info:", str(e))

def plot_population_forecast(historical_data, forecast_data, forecast_dates):
    """
    Create enhanced population forecast visualization with Malaysian context
    """
    try:
        fig = go.Figure()
        
        # Historical data with enhanced styling
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['total_prisoners'],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=4, color='#1f77b4'),
            hovertemplate='<b>Date:</b> %{x}<br><b>Prisoners:</b> %{y:,.0f}<extra></extra>'
        ))
        
        # Forecast data with confidence styling
        if forecast_data is not None and len(forecast_data) > 0:
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_data,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#ff7f0e', width=3, dash='dash'),
                marker=dict(size=5, color='#ff7f0e', symbol='diamond'),
                hovertemplate='<b>Date:</b> %{x}<br><b>Forecast:</b> %{y:,.0f}<extra></extra>'
            ))
            
            # Add confidence bands (±5% uncertainty)
            uncertainty = np.array(forecast_data) * 0.05
            upper_bound = np.array(forecast_data) + uncertainty
            lower_bound = np.array(forecast_data) - uncertainty
            
            fig.add_trace(go.Scatter(
                x=list(forecast_dates) + list(forecast_dates[::-1]),
                y=list(upper_bound) + list(lower_bound[::-1]),
                fill='toself',
                fillcolor='rgba(255,127,14,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='Confidence Interval (±5%)',
                hoverinfo='skip'
            ))
        
        # Add Malaysian prison system capacity line
        max_capacity = 97000
        fig.add_hline(
            y=max_capacity, 
            line_dash="dot", 
            line_color="red",
            annotation_text="System Capacity (97,000)",
            annotation_position="top right"
        )
        
        # Add overcrowding warning line (90% capacity)
        warning_capacity = max_capacity * 0.9
        fig.add_hline(
            y=warning_capacity, 
            line_dash="dot", 
            line_color="orange",
            annotation_text="Overcrowding Alert (90%)",
            annotation_position="bottom right"
        )
        
        # Enhanced layout with Malaysian styling
        fig.update_layout(
            title=dict(
                text='Malaysian Prison Population Forecast',
                font=dict(size=18, color='#2c3e50')
            ),
            xaxis_title='Date',
            yaxis_title='Number of Prisoners',
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500
        )
        
        # Format y-axis with thousands separator
        fig.update_yaxes(tickformat=',.')
        
        st.plotly_chart(fig, use_container_width=True, key="population_forecast_chart")
        
    except Exception as e:
        st.error(f"Error creating population forecast plot: {e}")
        st.write("Debug info:", str(e))

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
        
        st.plotly_chart(fig, use_container_width=True, key="staffing_breakdown_chart")
        
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
        
        st.plotly_chart(fig, use_container_width=True, key="cost_breakdown_chart")
        
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
        
        st.plotly_chart(fig, use_container_width=True, key="trend_analysis_chart")
        
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
        
        st.plotly_chart(fig, use_container_width=True, key="correlation_matrix_chart")
        
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
        
        st.plotly_chart(fig, use_container_width=True, key="seasonal_decomposition_chart")
        
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
        
        st.plotly_chart(fig, use_container_width=True, key="model_performance_chart")
        
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
        
        st.plotly_chart(fig, use_container_width=True, key="forecast_comparison_chart")
        
    except Exception as e:
        st.error(f"Error creating forecast comparison plot: {e}")
