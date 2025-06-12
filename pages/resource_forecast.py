import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualization import plot_cost_breakdown, plot_trend_analysis
from utils.data_utils import calculate_growth_rate

def show_resource_forecast(data, models):
    """
    Display resource forecasting page
    """
    st.header("💰 Resource Forecast")
    st.markdown("---")
    
    if 'resource_data' not in data or 'population_data' not in data:
        st.error("Required data not available")
        return
    
    resource_data = data['resource_data']
    population_data = data['population_data']
    
    # Sidebar controls
    st.sidebar.subheader("Resource Parameters")
    
    forecast_months = st.sidebar.slider(
        "Forecast Period (months)",
        min_value=6,
        max_value=36,
        value=24,
        step=6
    )
    
    cost_adjustment = st.sidebar.slider(
        "Cost Inflation Rate (%/year)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.5
    )
    
    efficiency_target = st.sidebar.slider(
        "Target Efficiency Improvement (%)",
        min_value=0,
        max_value=25,
        value=5,
        step=1
    )
    
    capacity_expansion = st.sidebar.slider(
        "Planned Capacity Expansion (%)",
        min_value=0,
        max_value=30,
        value=0,
        step=5
    )
    
    # Current statistics
    col1, col2, col3, col4 = st.columns(4)
    
    current_utilization = resource_data['capacity_utilization'].iloc[-1]
    current_cost = resource_data['total_monthly_cost'].iloc[-1]
    current_efficiency = resource_data['energy_efficiency'].iloc[-1]
    food_waste = resource_data['food_waste_rate'].iloc[-1]
    
    col1.metric("Capacity Utilization", f"{current_utilization:.1f}%")
    col2.metric("Monthly Cost", f"MYR {current_cost/1000000:.1f}M")
    col3.metric("Energy Efficiency", f"{current_efficiency:.1%}")
    col4.metric("Food Waste Rate", f"{food_waste:.1%}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Cost Forecast", "🏗️ Capacity Planning", "⚡ Efficiency Analysis", "📊 Resource Optimization"])
    
    with tab1:
        st.subheader("Cost Forecast and Budget Planning")
        
        # Generate cost forecast
        forecast_data, forecast_dates = generate_resource_forecast(
            resource_data, population_data, forecast_months, cost_adjustment, efficiency_target
        )
        
        if forecast_data is not None:
            # Plot cost forecast
            plot_cost_forecast_chart(resource_data, forecast_data, forecast_dates)
            
            # Cost projections
            st.subheader("Cost Projections")
            
            col1, col2, col3 = st.columns(3)
            
            projected_cost = forecast_data['total_monthly_cost'][-1]
            cost_change = ((projected_cost - current_cost) / current_cost) * 100
            annual_cost = projected_cost * 12
            
            with col1:
                st.metric(
                    "Projected Monthly Cost",
                    f"MYR {projected_cost/1000000:.2f}M",
                    f"{cost_change:+.1f}%"
                )
            
            with col2:
                st.metric(
                    "Projected Annual Cost",
                    f"MYR {annual_cost/1000000:.1f}M"
                )
            
            with col3:
                savings_potential = calculate_efficiency_savings(forecast_data, efficiency_target)
                st.metric(
                    "Efficiency Savings",
                    f"MYR {savings_potential/1000000:.2f}M/year"
                )
            
            # Cost breakdown forecast
            create_cost_breakdown_forecast(forecast_data, forecast_dates)
        
        else:
            st.warning("Unable to generate resource forecast.")
    
    with tab2:
        st.subheader("Capacity Planning and Infrastructure")
        
        # Current capacity analysis
        create_capacity_analysis(resource_data, population_data, capacity_expansion)
        
        # Infrastructure requirements
        create_infrastructure_planning(resource_data, forecast_data if 'forecast_data' in locals() else None)
        
        # Maintenance scheduling
        create_maintenance_analysis(resource_data)
    
    with tab3:
        st.subheader("Efficiency Analysis and Optimization")
        
        # Energy efficiency trends
        create_efficiency_analysis(resource_data)
        
        # Waste reduction analysis
        create_waste_analysis(resource_data)
        
        # Operational efficiency metrics
        create_operational_efficiency(resource_data, population_data)
    
    with tab4:
        st.subheader("Resource Optimization Strategies")
        
        # Resource allocation optimization
        create_resource_optimization(resource_data, population_data)
        
        # Procurement planning
        create_procurement_analysis(resource_data)
        
        # Sustainability initiatives
        create_sustainability_analysis(resource_data)

def generate_resource_forecast(resource_data, population_data, months, cost_inflation, efficiency_improvement):
    """
    Generate resource forecast based on population projections and efficiency targets
    """
    try:
        # Get population forecast (simplified trend-based projection)
        current_population = population_data['total_prisoners'].iloc[-1]
        population_growth_rate = calculate_growth_rate(population_data, 'total_prisoners', 12) / 100
        
        # Generate population projections
        population_forecast = []
        for i in range(months):
            projected_pop = current_population * (1 + population_growth_rate/12) ** (i + 1)
            population_forecast.append(projected_pop)
        
        # Base costs and metrics
        current_daily_cost = resource_data['daily_cost_per_prisoner'].iloc[-1]
        current_capacity = resource_data['total_capacity'].iloc[0]
        current_efficiency = resource_data['energy_efficiency'].iloc[-1]
        
        # Generate resource forecasts
        forecast_data = {
            'total_monthly_cost': [],
            'monthly_food_cost': [],
            'monthly_medical_cost': [],
            'monthly_utility_cost': [],
            'monthly_other_cost': [],
            'capacity_utilization': [],
            'daily_cost_per_prisoner': [],
            'energy_efficiency': [],
            'food_waste_rate': [],
            'maintenance_cost': []
        }
        
        for i, pop in enumerate(population_forecast):
            # Apply cost inflation
            month_inflation = (cost_inflation / 100) / 12
            inflated_daily_cost = current_daily_cost * (1 + month_inflation) ** (i + 1)
            
            # Apply efficiency improvements
            month_efficiency = current_efficiency + (efficiency_improvement / 100) * (i / months)
            efficiency_factor = current_efficiency / month_efficiency if month_efficiency > 0 else 1
            
            # Calculate costs with efficiency adjustments
            effective_daily_cost = inflated_daily_cost * efficiency_factor
            
            # Cost breakdown (30 days per month)
            monthly_food_cost = pop * effective_daily_cost * 30 * 0.4
            monthly_medical_cost = pop * effective_daily_cost * 30 * 0.15
            monthly_utility_cost = pop * effective_daily_cost * 30 * 0.20 * efficiency_factor
            monthly_other_cost = pop * effective_daily_cost * 30 * 0.25
            
            total_monthly_cost = monthly_food_cost + monthly_medical_cost + monthly_utility_cost + monthly_other_cost
            
            # Capacity utilization
            capacity_utilization = (pop / current_capacity) * 100
            
            # Maintenance costs (increase with utilization and age)
            base_maintenance = resource_data['maintenance_cost'].mean()
            utilization_factor = max(1, capacity_utilization / 80)  # Increase if over 80%
            maintenance_cost = base_maintenance * utilization_factor
            
            # Food waste (improving with efficiency)
            current_waste_rate = resource_data['food_waste_rate'].mean()
            improved_waste_rate = current_waste_rate * (1 - (efficiency_improvement / 100) * 0.3)
            
            # Store forecasted values
            forecast_data['total_monthly_cost'].append(total_monthly_cost)
            forecast_data['monthly_food_cost'].append(monthly_food_cost)
            forecast_data['monthly_medical_cost'].append(monthly_medical_cost)
            forecast_data['monthly_utility_cost'].append(monthly_utility_cost)
            forecast_data['monthly_other_cost'].append(monthly_other_cost)
            forecast_data['capacity_utilization'].append(capacity_utilization)
            forecast_data['daily_cost_per_prisoner'].append(effective_daily_cost)
            forecast_data['energy_efficiency'].append(month_efficiency)
            forecast_data['food_waste_rate'].append(improved_waste_rate)
            forecast_data['maintenance_cost'].append(maintenance_cost)
        
        # Generate forecast dates
        last_date = resource_data['date'].max()
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=months,
            freq='M'
        )
        
        return forecast_data, forecast_dates
        
    except Exception as e:
        st.error(f"Error generating resource forecast: {e}")
        return None, None

def plot_cost_forecast_chart(historical_data, forecast_data, forecast_dates):
    """
    Plot cost forecast chart
    """
    try:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total Monthly Cost', 'Cost per Prisoner', 'Capacity Utilization', 'Energy Efficiency'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Total monthly cost
        fig.add_trace(
            go.Scatter(
                x=historical_data['date'],
                y=historical_data['total_monthly_cost'] / 1000000,
                mode='lines',
                name='Historical Cost',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=np.array(forecast_data['total_monthly_cost']) / 1000000,
                mode='lines',
                name='Forecast Cost',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Cost per prisoner
        fig.add_trace(
            go.Scatter(
                x=historical_data['date'],
                y=historical_data['daily_cost_per_prisoner'],
                mode='lines',
                name='Historical Cost/Prisoner',
                line=dict(color='blue', width=2),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast_data['daily_cost_per_prisoner'],
                mode='lines',
                name='Forecast Cost/Prisoner',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Capacity utilization
        fig.add_trace(
            go.Scatter(
                x=historical_data['date'],
                y=historical_data['capacity_utilization'],
                mode='lines',
                name='Historical Utilization',
                line=dict(color='blue', width=2),
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast_data['capacity_utilization'],
                mode='lines',
                name='Forecast Utilization',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add 100% capacity line
        fig.add_hline(y=100, line_dash="dot", line_color="orange", row=2, col=1)
        
        # Energy efficiency
        fig.add_trace(
            go.Scatter(
                x=historical_data['date'],
                y=historical_data['energy_efficiency'] * 100,
                mode='lines',
                name='Historical Efficiency',
                line=dict(color='blue', width=2),
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=np.array(forecast_data['energy_efficiency']) * 100,
                mode='lines',
                name='Forecast Efficiency',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Resource Forecast Dashboard',
            height=700,
            hovermode='x unified'
        )
        
        # Update axis labels
        fig.update_yaxes(title_text="Cost (Million MYR)", row=1, col=1)
        fig.update_yaxes(title_text="MYR per Day", row=1, col=2)
        fig.update_yaxes(title_text="Utilization %", row=2, col=1)
        fig.update_yaxes(title_text="Efficiency %", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating cost forecast chart: {e}")

def create_cost_breakdown_forecast(forecast_data, forecast_dates):
    """
    Create cost breakdown forecast visualization
    """
    try:
        st.subheader("Cost Breakdown Forecast")
        
        fig = go.Figure()
        
        # Stacked area chart for cost breakdown
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=np.array(forecast_data['monthly_food_cost']) / 1000000,
            mode='lines',
            name='Food Costs',
            stackgroup='one',
            line=dict(color='orange')
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=np.array(forecast_data['monthly_medical_cost']) / 1000000,
            mode='lines',
            name='Medical Costs',
            stackgroup='one',
            line=dict(color='red')
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=np.array(forecast_data['monthly_utility_cost']) / 1000000,
            mode='lines',
            name='Utility Costs',
            stackgroup='one',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=np.array(forecast_data['monthly_other_cost']) / 1000000,
            mode='lines',
            name='Other Costs',
            stackgroup='one',
            line=dict(color='gray')
        ))
        
        fig.update_layout(
            title='Monthly Cost Breakdown Forecast',
            xaxis_title='Date',
            yaxis_title='Cost (Million MYR)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating cost breakdown forecast: {e}")

def calculate_efficiency_savings(forecast_data, efficiency_target):
    """
    Calculate potential savings from efficiency improvements
    """
    try:
        # Compare costs with and without efficiency improvements
        total_forecast_cost = sum(forecast_data['total_monthly_cost']) * 12  # Annual
        
        # Calculate cost without efficiency improvements
        baseline_cost = total_forecast_cost / (1 - efficiency_target / 100)
        
        savings = baseline_cost - total_forecast_cost
        return max(0, savings)
        
    except Exception as e:
        return 0

def create_capacity_analysis(resource_data, population_data, capacity_expansion):
    """
    Create capacity analysis and planning
    """
    try:
        st.subheader("Capacity Analysis")
        
        # Current capacity metrics
        current_capacity = resource_data['total_capacity'].iloc[0]
        current_utilization = resource_data['capacity_utilization'].iloc[-1]
        current_population = population_data['total_prisoners'].iloc[-1]
        
        # With expansion
        expanded_capacity = current_capacity * (1 + capacity_expansion / 100)
        new_utilization = (current_population / expanded_capacity) * 100
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Current Capacity", f"{current_capacity:,.0f}")
        col2.metric("Current Utilization", f"{current_utilization:.1f}%")
        
        if capacity_expansion > 0:
            col3.metric(
                "With Expansion",
                f"{new_utilization:.1f}%",
                f"{new_utilization - current_utilization:+.1f}%"
            )
        else:
            col3.metric("Expansion Planned", "None")
        
        # Capacity trends
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=resource_data['date'],
            y=resource_data['capacity_utilization'],
            mode='lines+markers',
            name='Capacity Utilization',
            line=dict(color='blue', width=2)
        ))
        
        # Add critical thresholds
        fig.add_hline(y=90, line_dash="dash", line_color="orange", annotation_text="Critical Level (90%)")
        fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Full Capacity")
        
        if capacity_expansion > 0:
            # Show impact of expansion
            fig.add_hline(
                y=new_utilization,
                line_dash="dot",
                line_color="green",
                annotation_text=f"With {capacity_expansion}% Expansion"
            )
        
        fig.update_layout(
            title='Capacity Utilization Trends',
            xaxis_title='Date',
            yaxis_title='Utilization %',
            yaxis=dict(range=[0, 120])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Capacity recommendations
        if current_utilization > 90:
            st.error("⚠️ Critical: Capacity utilization above 90%. Immediate expansion needed.")
        elif current_utilization > 80:
            st.warning("⚠️ Warning: Capacity utilization above 80%. Plan for expansion.")
        else:
            st.success("✅ Capacity utilization within acceptable range.")
            
    except Exception as e:
        st.error(f"Error creating capacity analysis: {e}")

def create_infrastructure_planning(resource_data, forecast_data):
    """
    Create infrastructure planning analysis
    """
    try:
        st.subheader("Infrastructure Planning")
        
        # Current infrastructure costs
        current_maintenance = resource_data['maintenance_cost'].iloc[-1]
        avg_maintenance = resource_data['maintenance_cost'].mean()
        
        col1, col2 = st.columns(2)
        
        col1.metric("Current Monthly Maintenance", f"MYR {current_maintenance:,.0f}")
        col2.metric("Average Monthly Maintenance", f"MYR {avg_maintenance:,.0f}")
        
        # Infrastructure investment needs
        st.write("**Infrastructure Investment Priorities:**")
        
        # Calculate infrastructure scores based on utilization and age
        current_utilization = resource_data['capacity_utilization'].iloc[-1]
        
        if current_utilization > 85:
            st.write("🔴 **High Priority:**")
            st.write("- Additional housing blocks")
            st.write("- Kitchen and dining facility expansion")
            st.write("- Medical facility upgrades")
        
        if current_utilization > 75:
            st.write("🟡 **Medium Priority:**")
            st.write("- Security system upgrades")
            st.write("- Utility infrastructure improvements")
            st.write("- Recreation facility expansion")
        
        st.write("🟢 **Routine Maintenance:**")
        st.write("- Regular building maintenance")
        st.write("- Equipment replacement")
        st.write("- Technology updates")
        
        # Maintenance cost trends
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=resource_data['date'],
            y=resource_data['maintenance_cost'],
            mode='lines+markers',
            name='Monthly Maintenance Cost',
            line=dict(color='brown', width=2)
        ))
        
        fig.add_hline(
            y=avg_maintenance,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Average: MYR {avg_maintenance:,.0f}"
        )
        
        fig.update_layout(
            title='Infrastructure Maintenance Costs',
            xaxis_title='Date',
            yaxis_title='Cost (MYR)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating infrastructure planning: {e}")

def create_maintenance_analysis(resource_data):
    """
    Create maintenance scheduling analysis
    """
    try:
        st.subheader("Maintenance Scheduling")
        
        # Maintenance statistics
        maintenance_costs = resource_data['maintenance_cost']
        maintenance_trend = calculate_growth_rate(resource_data, 'maintenance_cost', 12)
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Annual Maintenance", f"MYR {maintenance_costs.sum():,.0f}")
        col2.metric("Monthly Average", f"MYR {maintenance_costs.mean():,.0f}")
        col3.metric("Growth Rate", f"{maintenance_trend:.1f}%" if maintenance_trend else "N/A")
        
        # Maintenance scheduling recommendations
        st.write("**Recommended Maintenance Schedule:**")
        
        st.write("**Monthly:**")
        st.write("- HVAC system inspection and cleaning")
        st.write("- Security equipment testing")
        st.write("- Emergency system checks")
        
        st.write("**Quarterly:**")
        st.write("- Electrical system inspection")
        st.write("- Plumbing system maintenance")
        st.write("- Kitchen equipment servicing")
        
        st.write("**Annually:**")
        st.write("- Building structure inspection")
        st.write("- Fire safety system overhaul")
        st.write("- Technology infrastructure upgrade")
        
        # Seasonal maintenance planning
        st.write("**Seasonal Considerations:**")
        st.write("- **Monsoon Season:** Roof and drainage maintenance")
        st.write("- **Hot Season:** Air conditioning system preparation")
        st.write("- **Year-end:** Budget planning for next year's maintenance")
        
    except Exception as e:
        st.error(f"Error creating maintenance analysis: {e}")

def create_efficiency_analysis(resource_data):
    """
    Create energy efficiency analysis
    """
    try:
        st.subheader("Energy Efficiency Analysis")
        
        # Efficiency metrics
        current_efficiency = resource_data['energy_efficiency'].iloc[-1]
        avg_efficiency = resource_data['energy_efficiency'].mean()
        efficiency_trend = calculate_growth_rate(resource_data, 'energy_efficiency', 12)
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Current Efficiency", f"{current_efficiency:.1%}")
        col2.metric("Average Efficiency", f"{avg_efficiency:.1%}")
        col3.metric("Efficiency Trend", f"{efficiency_trend:.1f}%" if efficiency_trend else "N/A")
        
        # Efficiency trends
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=resource_data['date'],
            y=resource_data['energy_efficiency'] * 100,
            mode='lines+markers',
            name='Energy Efficiency',
            line=dict(color='green', width=2)
        ))
        
        # Add target line
        target_efficiency = 85  # 85% target
        fig.add_hline(
            y=target_efficiency,
            line_dash="dash",
            line_color="blue",
            annotation_text=f"Target: {target_efficiency}%"
        )
        
        fig.update_layout(
            title='Energy Efficiency Trends',
            xaxis_title='Date',
            yaxis_title='Efficiency %',
            yaxis=dict(range=[60, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Efficiency improvement recommendations
        st.write("**Energy Efficiency Improvement Strategies:**")
        
        if current_efficiency < 0.75:
            st.write("🔴 **Urgent Actions Needed:**")
            st.write("- LED lighting conversion")
            st.write("- HVAC system upgrade")
            st.write("- Building insulation improvement")
        
        st.write("🟡 **Medium-term Improvements:**")
        st.write("- Solar panel installation")
        st.write("- Smart energy management systems")
        st.write("- Energy-efficient appliances")
        
        st.write("🟢 **Long-term Initiatives:**")
        st.write("- Green building certification")
        st.write("- Renewable energy integration")
        st.write("- Energy storage systems")
        
    except Exception as e:
        st.error(f"Error creating efficiency analysis: {e}")

def create_waste_analysis(resource_data):
    """
    Create waste reduction analysis
    """
    try:
        st.subheader("Waste Reduction Analysis")
        
        # Waste metrics
        current_waste = resource_data['food_waste_rate'].iloc[-1]
        avg_waste = resource_data['food_waste_rate'].mean()
        
        col1, col2 = st.columns(2)
        
        col1.metric("Current Food Waste", f"{current_waste:.1%}")
        col2.metric("Average Food Waste", f"{avg_waste:.1%}")
        
        # Waste trends
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=resource_data['date'],
            y=resource_data['food_waste_rate'] * 100,
            mode='lines+markers',
            name='Food Waste Rate',
            line=dict(color='red', width=2)
        ))
        
        # Add target line
        target_waste = 8  # 8% target
        fig.add_hline(
            y=target_waste,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Target: {target_waste}%"
        )
        
        fig.update_layout(
            title='Food Waste Rate Trends',
            xaxis_title='Date',
            yaxis_title='Waste Rate %'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Waste reduction strategies
        st.write("**Waste Reduction Strategies:**")
        
        if current_waste > 0.15:
            st.write("🔴 **Immediate Actions:**")
            st.write("- Improve meal planning accuracy")
            st.write("- Implement portion control measures")
            st.write("- Staff training on waste reduction")
        
        st.write("🟡 **Operational Improvements:**")
        st.write("- Digital meal planning systems")
        st.write("- Inventory management optimization")
        st.write("- Food donation programs")
        
        st.write("🟢 **Sustainability Initiatives:**")
        st.write("- Composting programs")
        st.write("- Prison garden/farming projects")
        st.write("- Waste-to-energy systems")
        
    except Exception as e:
        st.error(f"Error creating waste analysis: {e}")

def create_operational_efficiency(resource_data, population_data):
    """
    Create operational efficiency metrics
    """
    try:
        st.subheader("Operational Efficiency Metrics")
        
        # Calculate efficiency metrics
        merged_data = pd.merge(resource_data, population_data[['date', 'total_prisoners']], on='date')
        merged_data['cost_per_prisoner_per_day'] = merged_data['total_monthly_cost'] / merged_data['total_prisoners'] / 30
        merged_data['efficiency_score'] = merged_data['energy_efficiency'] / merged_data['food_waste_rate']
        
        # Current metrics
        current_cost_per_prisoner = merged_data['cost_per_prisoner_per_day'].iloc[-1]
        avg_cost_per_prisoner = merged_data['cost_per_prisoner_per_day'].mean()
        efficiency_score = merged_data['efficiency_score'].iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Cost per Prisoner/Day", f"MYR {current_cost_per_prisoner:.2f}")
        col2.metric("Average Historical", f"MYR {avg_cost_per_prisoner:.2f}")
        col3.metric("Efficiency Score", f"{efficiency_score:.1f}")
        
        # Efficiency trends
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Cost per Prisoner per Day', 'Operational Efficiency Score')
        )
        
        fig.add_trace(
            go.Scatter(
                x=merged_data['date'],
                y=merged_data['cost_per_prisoner_per_day'],
                mode='lines',
                name='Cost/Prisoner/Day',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=merged_data['date'],
                y=merged_data['efficiency_score'],
                mode='lines',
                name='Efficiency Score',
                line=dict(color='green', width=2)
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Operational Efficiency Trends',
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating operational efficiency metrics: {e}")

def create_resource_optimization(resource_data, population_data):
    """
    Create resource optimization recommendations
    """
    try:
        st.subheader("Resource Optimization Recommendations")
        
        # Analysis current resource allocation
        latest_data = resource_data.iloc[-1]
        total_cost = latest_data['total_monthly_cost']
        
        # Cost distribution analysis
        food_pct = (latest_data['monthly_food_cost'] / total_cost) * 100
        medical_pct = (latest_data['monthly_medical_cost'] / total_cost) * 100
        utility_pct = (latest_data['monthly_utility_cost'] / total_cost) * 100
        other_pct = (latest_data['monthly_other_cost'] / total_cost) * 100
        
        st.write("**Current Resource Allocation:**")
        st.write(f"- Food: {food_pct:.1f}% (MYR {latest_data['monthly_food_cost']:,.0f})")
        st.write(f"- Medical: {medical_pct:.1f}% (MYR {latest_data['monthly_medical_cost']:,.0f})")
        st.write(f"- Utilities: {utility_pct:.1f}% (MYR {latest_data['monthly_utility_cost']:,.0f})")
        st.write(f"- Other: {other_pct:.1f}% (MYR {latest_data['monthly_other_cost']:,.0f})")
        
        # Optimization recommendations
        st.write("**Optimization Opportunities:**")
        
        if food_pct > 45:
            st.write("🔴 **Food Costs (High):**")
            st.write("- Negotiate bulk purchasing contracts")
            st.write("- Implement central kitchen model")
            st.write("- Reduce food waste through better planning")
        
        if utility_pct > 25:
            st.write("🟡 **Utility Costs (Above Target):**")
            st.write("- Energy efficiency improvements")
            st.write("- Time-of-use electricity management")
            st.write("- Water conservation measures")
        
        if medical_pct < 12:
            st.write("🟢 **Medical Costs (Efficient):**")
            st.write("- Maintain current preventive care programs")
            st.write("- Consider expanding mental health services")
        
        # Resource reallocation scenarios
        st.write("**Resource Reallocation Scenarios:**")
        
        scenario_col1, scenario_col2 = st.columns(2)
        
        with scenario_col1:
            st.write("**Scenario A: Cost Reduction Focus**")
            st.write("- Reduce food costs by 10%")
            st.write("- Reduce utility costs by 15%")
            st.write("- Maintain medical spending")
            
            savings_a = (latest_data['monthly_food_cost'] * 0.1 + 
                        latest_data['monthly_utility_cost'] * 0.15)
            st.write(f"**Potential monthly savings: MYR {savings_a:,.0f}**")
        
        with scenario_col2:
            st.write("**Scenario B: Quality Enhancement**")
            st.write("- Increase medical spending by 20%")
            st.write("- Improve food quality (5% increase)")
            st.write("- Reduce utility costs by 10%")
            
            net_change_b = (latest_data['monthly_medical_cost'] * 0.2 + 
                           latest_data['monthly_food_cost'] * 0.05 - 
                           latest_data['monthly_utility_cost'] * 0.1)
            st.write(f"**Net monthly cost change: MYR {net_change_b:+,.0f}**")
        
    except Exception as e:
        st.error(f"Error creating resource optimization: {e}")

def create_procurement_analysis(resource_data):
    """
    Create procurement planning analysis
    """
    try:
        st.subheader("Procurement Planning")
        
        # Current procurement costs
        latest_data = resource_data.iloc[-1]
        medical_supplies = latest_data['medical_supplies_cost']
        security_equipment = latest_data['security_equipment_cost']
        
        col1, col2 = st.columns(2)
        
        col1.metric("Monthly Medical Supplies", f"MYR {medical_supplies:,.0f}")
        col2.metric("Monthly Security Equipment", f"MYR {security_equipment:,.0f}")
        
        # Procurement trends
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Medical Supplies Cost', 'Security Equipment Cost')
        )
        
        fig.add_trace(
            go.Scatter(
                x=resource_data['date'],
                y=resource_data['medical_supplies_cost'],
                mode='lines',
                name='Medical Supplies',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=resource_data['date'],
                y=resource_data['security_equipment_cost'],
                mode='lines',
                name='Security Equipment',
                line=dict(color='blue', width=2)
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Procurement Cost Trends',
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Procurement recommendations
        st.write("**Procurement Optimization Strategies:**")
        
        st.write("🟢 **Strategic Sourcing:**")
        st.write("- Consolidate suppliers for better pricing")
        st.write("- Implement competitive bidding processes")
        st.write("- Negotiate long-term contracts with volume discounts")
        
        st.write("🟡 **Inventory Management:**")
        st.write("- Implement just-in-time inventory for non-critical items")
        st.write("- Maintain strategic stockpiles for essential supplies")
        st.write("- Use inventory management software")
        
        st.write("🔵 **Quality Assurance:**")
        st.write("- Establish supplier quality standards")
        st.write("- Regular supplier performance reviews")
        st.write("- Backup supplier arrangements")
        
    except Exception as e:
        st.error(f"Error creating procurement analysis: {e}")

def create_sustainability_analysis(resource_data):
    """
    Create sustainability initiatives analysis
    """
    try:
        st.subheader("Sustainability Initiatives")
        
        # Current sustainability metrics
        current_efficiency = resource_data['energy_efficiency'].iloc[-1]
        current_waste = resource_data['food_waste_rate'].iloc[-1]
        
        # Sustainability score (simplified)
        sustainability_score = (current_efficiency * 0.6 + (1 - current_waste) * 0.4) * 100
        
        st.metric("Overall Sustainability Score", f"{sustainability_score:.1f}/100")
        
        # Sustainability initiatives
        st.write("**Current Sustainability Initiatives:**")
        
        if current_efficiency > 0.8:
            st.write("✅ Energy efficiency programs")
        else:
            st.write("❌ Energy efficiency programs (needs improvement)")
            
        if current_waste < 0.1:
            st.write("✅ Waste reduction programs")
        else:
            st.write("❌ Waste reduction programs (needs improvement)")
        
        # Future sustainability projects
        st.write("**Recommended Sustainability Projects:**")
        
        st.write("🌱 **Environmental:**")
        st.write("- Solar panel installation (ROI: 7-10 years)")
        st.write("- Rainwater harvesting systems")
        st.write("- Green building upgrades")
        
        st.write("💰 **Economic:**")
        st.write("- Energy management systems")
        st.write("- Waste-to-energy conversion")
        st.write("- Local sourcing initiatives")
        
        st.write("👥 **Social:**")
        st.write("- Prisoner training in green skills")
        st.write("- Community garden programs")
        st.write("- Environmental education initiatives")
        
        # Sustainability ROI calculation
        st.write("**Sustainability Investment ROI:**")
        
        # Simple ROI calculation for solar panels
        solar_investment = 2000000  # MYR 2M investment
        annual_energy_savings = current_efficiency * 500000  # Estimated savings
        solar_roi_years = solar_investment / annual_energy_savings
        
        st.write(f"- Solar panel investment: MYR {solar_investment:,.0f}")
        st.write(f"- Estimated annual savings: MYR {annual_energy_savings:,.0f}")
        st.write(f"- Payback period: {solar_roi_years:.1f} years")
        
    except Exception as e:
        st.error(f"Error creating sustainability analysis: {e}")
