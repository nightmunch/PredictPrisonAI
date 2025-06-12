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

from utils.visualization import plot_staffing_breakdown, plot_trend_analysis
from utils.data_utils import calculate_growth_rate

def show_staffing_forecast(data, models):
    """
    Display staffing forecasting page
    """
    st.header("👮 Staffing Forecast")
    st.markdown("---")
    
    if 'staffing_data' not in data or 'population_data' not in data:
        st.error("Required data not available")
        return
    
    staffing_data = data['staffing_data']
    population_data = data['population_data']
    
    # Sidebar controls
    st.sidebar.subheader("Staffing Parameters")
    
    forecast_months = st.sidebar.slider(
        "Forecast Period (months)",
        min_value=6,
        max_value=36,
        value=24,
        step=6
    )
    
    target_ratio = st.sidebar.slider(
        "Target Staff-to-Prisoner Ratio",
        min_value=0.20,
        max_value=0.40,
        value=0.28,
        step=0.01,
        format="%.2f"
    )
    
    efficiency_improvement = st.sidebar.slider(
        "Efficiency Improvement (%)",
        min_value=0,
        max_value=20,
        value=0,
        step=1
    )
    
    # Current statistics
    col1, col2, col3, col4 = st.columns(4)
    
    current_staff = staffing_data['total_staff'].iloc[-1]
    current_ratio = staffing_data['staff_prisoner_ratio'].iloc[-1]
    current_overtime = staffing_data['overtime_hours'].iloc[-1]
    available_staff = staffing_data['available_staff'].iloc[-1]
    
    col1.metric("Current Staff", f"{current_staff:,.0f}")
    col2.metric("Staff-Prisoner Ratio", f"{current_ratio:.3f}")
    col3.metric("Overtime Hours/Month", f"{current_overtime:.0f}")
    col4.metric("Available Staff", f"{available_staff:,.0f}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "👥 Staff Distribution", "⏰ Schedule Analysis", "📋 Planning"])
    
    with tab1:
        st.subheader("Staff Requirements Forecast")
        
        # Generate staffing forecast
        forecast_data, forecast_dates = generate_staffing_forecast(
            staffing_data, population_data, forecast_months, target_ratio, efficiency_improvement
        )
        
        if forecast_data is not None:
            # Plot staffing forecast
            plot_staffing_forecast_chart(staffing_data, forecast_data, forecast_dates)
            
            # Forecast statistics
            st.subheader("Staffing Projections")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                projected_staff = forecast_data['total_staff'][-1]
                staff_change = projected_staff - current_staff
                st.metric(
                    "Projected Staff Needs",
                    f"{projected_staff:,.0f}",
                    f"{staff_change:+.0f}"
                )
            
            with col2:
                recruitment_needed = max(0, staff_change)
                st.metric(
                    "Additional Recruitment",
                    f"{recruitment_needed:.0f}",
                    "New positions" if recruitment_needed > 0 else "No additional staff needed"
                )
            
            with col3:
                projected_ratio = forecast_data['staff_ratio'][-1]
                st.metric(
                    "Projected Ratio",
                    f"{projected_ratio:.3f}",
                    f"{((projected_ratio - current_ratio) / current_ratio * 100):+.1f}%"
                )
            
            # Cost implications
            st.subheader("Cost Implications")
            calculate_staffing_costs(forecast_data, current_staff)
        
        else:
            st.warning("Unable to generate staffing forecast.")
    
    with tab2:
        st.subheader("Staff Distribution Analysis")
        
        # Current staff breakdown
        plot_staffing_breakdown(staffing_data)
        
        # Staff category trends
        create_staff_category_trends(staffing_data)
        
        # Shift distribution analysis
        create_shift_analysis(staffing_data)
    
    with tab3:
        st.subheader("Schedule and Utilization Analysis")
        
        # Overtime analysis
        create_overtime_analysis(staffing_data)
        
        # Availability analysis
        create_availability_analysis(staffing_data)
        
        # Workload distribution
        create_workload_analysis(staffing_data, population_data)
    
    with tab4:
        st.subheader("Strategic Staffing Planning")
        
        # Recruitment planning
        create_recruitment_plan(staffing_data, forecast_data if 'forecast_data' in locals() else None)
        
        # Training requirements
        create_training_analysis(staffing_data)
        
        # Budget planning
        create_budget_planning(staffing_data)

def generate_staffing_forecast(staffing_data, population_data, months, target_ratio, efficiency_improvement):
    """
    Generate staffing forecast based on population projections and efficiency targets
    """
    try:
        # Get population forecast (simplified)
        current_population = population_data['total_prisoners'].iloc[-1]
        population_growth_rate = calculate_growth_rate(population_data, 'total_prisoners', 12) / 100
        
        # Generate population projections
        population_forecast = []
        for i in range(months):
            projected_pop = current_population * (1 + population_growth_rate/12) ** (i + 1)
            population_forecast.append(projected_pop)
        
        # Generate staffing requirements
        forecast_data = {
            'total_staff': [],
            'security_staff': [],
            'admin_staff': [],
            'medical_staff': [],
            'other_staff': [],
            'staff_ratio': [],
            'overtime_hours': [],
            'recruitment_needed': []
        }
        
        current_efficiency = 1.0
        target_efficiency = 1.0 + (efficiency_improvement / 100)
        
        for i, pop in enumerate(population_forecast):
            # Apply efficiency improvements gradually
            month_efficiency = current_efficiency + (target_efficiency - current_efficiency) * (i / months)
            
            # Calculate required staff
            effective_ratio = target_ratio / month_efficiency
            required_staff = int(pop * effective_ratio)
            
            # Staff distribution (based on current patterns)
            security_staff = int(required_staff * 0.65)
            admin_staff = int(required_staff * 0.15)
            medical_staff = int(required_staff * 0.08)
            other_staff = required_staff - security_staff - admin_staff - medical_staff
            
            # Overtime calculation (inverse relationship with efficiency)
            base_overtime = staffing_data['overtime_hours'].mean()
            overtime_hours = base_overtime / month_efficiency
            
            # Recruitment needs
            current_total = staffing_data['total_staff'].iloc[-1]
            recruitment_needed = max(0, required_staff - current_total)
            
            forecast_data['total_staff'].append(required_staff)
            forecast_data['security_staff'].append(security_staff)
            forecast_data['admin_staff'].append(admin_staff)
            forecast_data['medical_staff'].append(medical_staff)
            forecast_data['other_staff'].append(other_staff)
            forecast_data['staff_ratio'].append(effective_ratio)
            forecast_data['overtime_hours'].append(overtime_hours)
            forecast_data['recruitment_needed'].append(recruitment_needed)
        
        # Generate forecast dates
        last_date = staffing_data['date'].max()
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=months,
            freq='M'
        )
        
        return forecast_data, forecast_dates
        
    except Exception as e:
        st.error(f"Error generating staffing forecast: {e}")
        return None, None

def plot_staffing_forecast_chart(historical_data, forecast_data, forecast_dates):
    """
    Plot staffing forecast chart
    """
    try:
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Total Staff Forecast', 'Staff-to-Prisoner Ratio'),
            vertical_spacing=0.1
        )
        
        # Historical total staff
        fig.add_trace(
            go.Scatter(
                x=historical_data['date'],
                y=historical_data['total_staff'],
                mode='lines',
                name='Historical Staff',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Forecast total staff
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast_data['total_staff'],
                mode='lines',
                name='Forecast Staff',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Historical ratio
        fig.add_trace(
            go.Scatter(
                x=historical_data['date'],
                y=historical_data['staff_prisoner_ratio'],
                mode='lines',
                name='Historical Ratio',
                line=dict(color='blue', width=2),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Forecast ratio
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast_data['staff_ratio'],
                mode='lines',
                name='Forecast Ratio',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='Staffing Requirements Forecast',
            height=600,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Number of Staff", row=1, col=1)
        fig.update_yaxes(title_text="Staff-to-Prisoner Ratio", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating staffing forecast chart: {e}")

def create_staff_category_trends(staffing_data):
    """
    Create staff category trends visualization
    """
    try:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=staffing_data['date'],
            y=staffing_data['security_staff'],
            mode='lines',
            name='Security Staff',
            line=dict(color='red', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=staffing_data['date'],
            y=staffing_data['admin_staff'],
            mode='lines',
            name='Administrative Staff',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=staffing_data['date'],
            y=staffing_data['medical_staff'],
            mode='lines',
            name='Medical Staff',
            line=dict(color='green', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=staffing_data['date'],
            y=staffing_data['other_staff'],
            mode='lines',
            name='Other Staff',
            line=dict(color='orange', width=2)
        ))
        
        fig.update_layout(
            title='Staff Category Trends',
            xaxis_title='Date',
            yaxis_title='Number of Staff',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating staff category trends: {e}")

def create_shift_analysis(staffing_data):
    """
    Create shift distribution analysis
    """
    try:
        # Get latest shift data
        latest_data = staffing_data.iloc[-1]
        
        # Create shift distribution pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Day Shift', 'Evening Shift', 'Night Shift'],
            values=[
                latest_data['day_shift_staff'],
                latest_data['evening_shift_staff'],
                latest_data['night_shift_staff']
            ],
            hole=0.3
        )])
        
        fig.update_layout(
            title='Current Shift Distribution',
            annotations=[dict(text='Shifts', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Shift trends over time
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=staffing_data['date'],
            y=staffing_data['day_shift_staff'],
            mode='lines',
            name='Day Shift',
            stackgroup='one'
        ))
        
        fig2.add_trace(go.Scatter(
            x=staffing_data['date'],
            y=staffing_data['evening_shift_staff'],
            mode='lines',
            name='Evening Shift',
            stackgroup='one'
        ))
        
        fig2.add_trace(go.Scatter(
            x=staffing_data['date'],
            y=staffing_data['night_shift_staff'],
            mode='lines',
            name='Night Shift',
            stackgroup='one'
        ))
        
        fig2.update_layout(
            title='Shift Distribution Trends',
            xaxis_title='Date',
            yaxis_title='Number of Staff',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating shift analysis: {e}")

def create_overtime_analysis(staffing_data):
    """
    Create overtime analysis
    """
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            # Overtime trend
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=staffing_data['date'],
                y=staffing_data['overtime_hours'],
                mode='lines+markers',
                name='Monthly Overtime Hours',
                line=dict(color='red', width=2)
            ))
            
            # Add average line
            avg_overtime = staffing_data['overtime_hours'].mean()
            fig.add_hline(
                y=avg_overtime,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Average: {avg_overtime:.0f} hours"
            )
            
            fig.update_layout(
                title='Overtime Hours Trend',
                xaxis_title='Date',
                yaxis_title='Hours per Staff per Month'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Overtime statistics
            current_overtime = staffing_data['overtime_hours'].iloc[-1]
            max_overtime = staffing_data['overtime_hours'].max()
            min_overtime = staffing_data['overtime_hours'].min()
            
            st.metric("Current Overtime", f"{current_overtime:.0f} hours")
            st.metric("Maximum Recorded", f"{max_overtime:.0f} hours")
            st.metric("Minimum Recorded", f"{min_overtime:.0f} hours")
            
            # Overtime cost estimation
            avg_hourly_rate = 25  # MYR per hour (estimated)
            monthly_overtime_cost = current_overtime * staffing_data['total_staff'].iloc[-1] * avg_hourly_rate
            
            st.metric("Est. Monthly Overtime Cost", f"MYR {monthly_overtime_cost:,.0f}")
        
    except Exception as e:
        st.error(f"Error creating overtime analysis: {e}")

def create_availability_analysis(staffing_data):
    """
    Create staff availability analysis
    """
    try:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Sick Leave Rate', 'Vacation Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Sick leave rate
        fig.add_trace(
            go.Scatter(
                x=staffing_data['date'],
                y=staffing_data['sick_leave_rate'] * 100,
                mode='lines',
                name='Sick Leave %',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # Vacation rate
        fig.add_trace(
            go.Scatter(
                x=staffing_data['date'],
                y=staffing_data['vacation_rate'] * 100,
                mode='lines',
                name='Vacation %',
                line=dict(color='blue', width=2)
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Staff Availability Trends',
            showlegend=False,
            height=400
        )
        
        fig.update_yaxes(title_text="Percentage (%)", row=1, col=1)
        fig.update_yaxes(title_text="Percentage (%)", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Availability metrics
        col1, col2, col3 = st.columns(3)
        
        current_sick_rate = staffing_data['sick_leave_rate'].iloc[-1] * 100
        current_vacation_rate = staffing_data['vacation_rate'].iloc[-1] * 100
        overall_availability = (1 - staffing_data['sick_leave_rate'].iloc[-1] - staffing_data['vacation_rate'].iloc[-1]) * 100
        
        col1.metric("Current Sick Leave Rate", f"{current_sick_rate:.1f}%")
        col2.metric("Current Vacation Rate", f"{current_vacation_rate:.1f}%")
        col3.metric("Overall Availability", f"{overall_availability:.1f}%")
        
    except Exception as e:
        st.error(f"Error creating availability analysis: {e}")

def create_workload_analysis(staffing_data, population_data):
    """
    Create workload analysis
    """
    try:
        # Merge data for analysis
        merged_data = pd.merge(staffing_data, population_data[['date', 'total_prisoners']], on='date')
        merged_data['prisoners_per_staff'] = merged_data['total_prisoners'] / merged_data['available_staff']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=merged_data['date'],
            y=merged_data['prisoners_per_staff'],
            mode='lines+markers',
            name='Prisoners per Available Staff',
            line=dict(color='purple', width=2)
        ))
        
        # Add target line
        target_workload = 3.5  # Target prisoners per staff
        fig.add_hline(
            y=target_workload,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Target: {target_workload} prisoners/staff"
        )
        
        fig.update_layout(
            title='Staff Workload Analysis',
            xaxis_title='Date',
            yaxis_title='Prisoners per Available Staff'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Workload statistics
        current_workload = merged_data['prisoners_per_staff'].iloc[-1]
        avg_workload = merged_data['prisoners_per_staff'].mean()
        
        col1, col2 = st.columns(2)
        col1.metric("Current Workload", f"{current_workload:.1f} prisoners/staff")
        col2.metric("Average Workload", f"{avg_workload:.1f} prisoners/staff")
        
    except Exception as e:
        st.error(f"Error creating workload analysis: {e}")

def calculate_staffing_costs(forecast_data, current_staff):
    """
    Calculate staffing cost implications
    """
    try:
        # Cost assumptions (in MYR)
        avg_annual_salary = 45000  # Average annual salary
        recruitment_cost = 8000    # Cost per new hire
        training_cost = 5000       # Training cost per new hire
        
        # Calculate costs
        additional_staff = max(0, forecast_data['total_staff'][-1] - current_staff)
        
        # Annual salary costs
        additional_salary_cost = additional_staff * avg_annual_salary
        
        # One-time costs
        recruitment_costs = additional_staff * recruitment_cost
        training_costs = additional_staff * training_cost
        
        # Total first-year cost
        total_first_year = additional_salary_cost + recruitment_costs + training_costs
        
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Additional Staff", f"{additional_staff:.0f}")
        col2.metric("Annual Salary Cost", f"MYR {additional_salary_cost:,.0f}")
        col3.metric("Recruitment Cost", f"MYR {recruitment_costs:,.0f}")
        col4.metric("Total First Year", f"MYR {total_first_year:,.0f}")
        
    except Exception as e:
        st.error(f"Error calculating staffing costs: {e}")

def create_recruitment_plan(staffing_data, forecast_data):
    """
    Create recruitment planning analysis
    """
    try:
        st.subheader("Recruitment Planning")
        
        if forecast_data:
            # Calculate recruitment timeline
            total_recruitment = sum(forecast_data['recruitment_needed'])
            
            st.write(f"**Total recruitment needed over forecast period: {total_recruitment:.0f} staff**")
            
            # Recruitment by category (based on current distribution)
            latest_staffing = staffing_data.iloc[-1]
            total_current = latest_staffing['total_staff']
            
            security_ratio = latest_staffing['security_staff'] / total_current
            admin_ratio = latest_staffing['admin_staff'] / total_current
            medical_ratio = latest_staffing['medical_staff'] / total_current
            
            recruitment_breakdown = {
                'Security Staff': int(total_recruitment * security_ratio),
                'Administrative Staff': int(total_recruitment * admin_ratio),
                'Medical Staff': int(total_recruitment * medical_ratio),
                'Other Staff': int(total_recruitment * (1 - security_ratio - admin_ratio - medical_ratio))
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Recruitment by Category:**")
                for category, count in recruitment_breakdown.items():
                    st.write(f"- {category}: {count}")
            
            with col2:
                # Timeline
                st.write("**Suggested Timeline:**")
                quarterly_recruitment = total_recruitment / (len(forecast_data['total_staff']) / 3)
                st.write(f"- Quarterly recruitment target: {quarterly_recruitment:.0f} staff")
                st.write(f"- Lead time for recruitment: 3-4 months")
                st.write(f"- Training period: 2-3 months")
        else:
            st.info("Generate forecast to see recruitment planning details.")
            
    except Exception as e:
        st.error(f"Error creating recruitment plan: {e}")

def create_training_analysis(staffing_data):
    """
    Create training requirements analysis
    """
    try:
        st.subheader("Training Requirements")
        
        # Current staff numbers
        current_staff = staffing_data['total_staff'].iloc[-1]
        
        # Training assumptions
        annual_training_hours = 40  # Hours per staff per year
        refresher_training_rate = 0.2  # 20% need refresher training annually
        
        total_training_hours = current_staff * annual_training_hours
        refresher_staff = int(current_staff * refresher_training_rate)
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Annual Training Hours", f"{total_training_hours:,.0f}")
        col2.metric("Staff Needing Refresher", f"{refresher_staff:,.0f}")
        col3.metric("Training Cost Estimate", f"MYR {total_training_hours * 50:,.0f}")
        
        # Training categories
        st.write("**Key Training Areas:**")
        st.write("- Security protocols and procedures")
        st.write("- Emergency response and crisis management")
        st.write("- Human rights and prisoner welfare")
        st.write("- Technology and equipment updates")
        st.write("- Leadership and management development")
        
    except Exception as e:
        st.error(f"Error creating training analysis: {e}")

def create_budget_planning(staffing_data):
    """
    Create budget planning analysis
    """
    try:
        st.subheader("Budget Planning")
        
        # Current costs
        current_staff = staffing_data['total_staff'].iloc[-1]
        current_overtime = staffing_data['overtime_hours'].iloc[-1]
        
        # Cost calculations
        avg_annual_salary = 45000
        overtime_rate = 25  # MYR per hour
        
        annual_salary_cost = current_staff * avg_annual_salary
        monthly_overtime_cost = current_staff * current_overtime * overtime_rate
        annual_overtime_cost = monthly_overtime_cost * 12
        
        total_annual_cost = annual_salary_cost + annual_overtime_cost
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Annual Salary Cost", f"MYR {annual_salary_cost:,.0f}")
        col2.metric("Annual Overtime Cost", f"MYR {annual_overtime_cost:,.0f}")
        col3.metric("Total Annual Cost", f"MYR {total_annual_cost:,.0f}")
        
        # Budget breakdown
        st.write("**Budget Breakdown:**")
        salary_percentage = (annual_salary_cost / total_annual_cost) * 100
        overtime_percentage = (annual_overtime_cost / total_annual_cost) * 100
        
        st.write(f"- Base Salaries: {salary_percentage:.1f}% (MYR {annual_salary_cost:,.0f})")
        st.write(f"- Overtime: {overtime_percentage:.1f}% (MYR {annual_overtime_cost:,.0f})")
        
        # Recommendations
        st.write("**Budget Optimization Recommendations:**")
        if overtime_percentage > 15:
            st.warning("High overtime costs detected. Consider hiring additional staff.")
        if current_overtime > 150:
            st.warning("Excessive overtime hours may lead to staff burnout.")
        
    except Exception as e:
        st.error(f"Error creating budget planning: {e}")
