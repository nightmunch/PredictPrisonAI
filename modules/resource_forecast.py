import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta


def show_resource_forecast(data, models, l):
    """
    Display resource forecasting page with proper state/prison filtering
    """
    st.title(l["resource_forecast_header"])

    # Sidebar parameters
    st.sidebar.header(l["forecast_parameters"])

    # Forecast level selection
    forecast_level = st.sidebar.selectbox(
        l["forecast_level"], [l["overall_malaysia"], l["by_state"], l["by_prison"]]
    )

    selected_state = None
    selected_prison = None

    if forecast_level in [l["by_state"], l["by_prison"]]:
        available_states = sorted(data["prison_detail_data"]["state"].unique())
        selected_state = st.sidebar.selectbox(l["select_state"], available_states)

        if forecast_level == l["by_prison"] and selected_state:
            prisons_in_state = data["prison_detail_data"][
                data["prison_detail_data"]["state"] == selected_state
            ]["prison_name"].unique()
            selected_prison = st.sidebar.selectbox(
                l["select_prison"], sorted(prisons_in_state)
            )

    # Other parameters
    forecast_months = st.sidebar.slider(l["forecast_period"], 6, 36, 24)
    cost_adjustment = st.sidebar.slider(l["cost_inflation_rate"], -10, 20, 3)
    efficiency_target = st.sidebar.slider(l["efficiency_improvement_target"], 0, 25, 5)

    # Get base data
    resource_data = data["resource_data"]
    population_data = data["population_data"]

    # Calculate filtered data and title
    title_suffix, filtered_resource_data, population_ratio = calculate_filtered_data(
        data,
        forecast_level,
        selected_state,
        selected_prison,
        resource_data,
        population_data,
    )

    # Display header with filtered title
    st.header(f"{l['resource_forecast_header']}{title_suffix}")

    # Show population ratio info
    if population_ratio < 1.0:
        st.info(l["showing_data_for"].format(population_ratio))

    # Display current metrics with filtered data
    display_current_metrics(filtered_resource_data, resource_data)

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            l["cost_forecast_tab"],
            l["capacity_planning_tab"],
            l["efficiency_analysis_tab"],
            l["resource_optimization_tab"],
        ]
    )

    with tab1:
        show_cost_forecast_tab(
            filtered_resource_data,
            population_data,
            forecast_months,
            cost_adjustment,
            efficiency_target,
            title_suffix,
            population_ratio,
            l,
        )

    with tab2:
        show_capacity_planning_tab(
            filtered_resource_data, population_data, title_suffix, l
        )

    with tab3:
        show_efficiency_analysis_tab(filtered_resource_data, title_suffix, l)

    with tab4:
        show_resource_optimization_tab(
            filtered_resource_data, population_data, title_suffix, l
        )


def calculate_filtered_data(
    data,
    forecast_level,
    selected_state,
    selected_prison,
    resource_data,
    population_data,
):
    """
    Calculate filtered resource data based on selection
    """
    title_suffix = ""
    population_ratio = 1.0
    filtered_resource_data = resource_data.copy()

    if forecast_level == "By State" and selected_state:
        # Get state population data
        state_data = data["prison_detail_data"][
            data["prison_detail_data"]["state"] == selected_state
        ]
        latest_state_data = state_data[state_data["date"] == state_data["date"].max()]
        state_population = latest_state_data["prison_population"].sum()

        if state_population > 0:
            total_population = population_data["total_prisoners"].iloc[-1]
            population_ratio = state_population / total_population

            # Apply proportional scaling
            cost_columns = [
                "total_monthly_cost",
                "monthly_food_cost",
                "monthly_medical_cost",
                "monthly_utility_cost",
                "monthly_other_cost",
                "maintenance_cost",
            ]

            for col in cost_columns:
                if col in filtered_resource_data.columns:
                    filtered_resource_data[col] = (
                        filtered_resource_data[col] * population_ratio
                    )

            title_suffix = f" - {selected_state} State"
        else:
            title_suffix = " - Malaysia Overall"

    elif forecast_level == "By Prison" and selected_prison and selected_state:
        # Get prison population data
        prison_data = data["prison_detail_data"][
            (data["prison_detail_data"]["state"] == selected_state)
            & (data["prison_detail_data"]["prison_name"] == selected_prison)
        ]
        latest_prison_data = prison_data[
            prison_data["date"] == prison_data["date"].max()
        ]

        if not latest_prison_data.empty:
            prison_population = latest_prison_data["prison_population"].iloc[0]
            total_population = population_data["total_prisoners"].iloc[-1]
            population_ratio = prison_population / total_population

            # Apply proportional scaling
            cost_columns = [
                "total_monthly_cost",
                "monthly_food_cost",
                "monthly_medical_cost",
                "monthly_utility_cost",
                "monthly_other_cost",
                "maintenance_cost",
            ]

            for col in cost_columns:
                if col in filtered_resource_data.columns:
                    filtered_resource_data[col] = (
                        filtered_resource_data[col] * population_ratio
                    )

            title_suffix = f" - {selected_prison}"
        else:
            title_suffix = " - Malaysia Overall"
    else:
        title_suffix = " - Malaysia Overall"

    return title_suffix, filtered_resource_data, population_ratio


def display_current_metrics(filtered_resource_data, original_resource_data):
    """
    Display current resource metrics using filtered data
    """
    col1, col2, col3, col4 = st.columns(4)

    # Use filtered data for cost, original for percentages
    current_utilization = original_resource_data["capacity_utilization"].iloc[-1]
    current_cost = filtered_resource_data["total_monthly_cost"].iloc[-1]
    current_efficiency = original_resource_data["energy_efficiency"].iloc[-1]
    food_waste = original_resource_data["food_waste_rate"].iloc[-1]

    col1.metric("Capacity Utilization", f"{current_utilization:.1f}%")
    col2.metric("Monthly Cost", f"MYR {current_cost/1000000:.1f}M")
    col3.metric("Energy Efficiency", f"{current_efficiency:.1%}")
    col4.metric("Food Waste Rate", f"{food_waste:.1%}")


def show_cost_forecast_tab(
    filtered_resource_data,
    population_data,
    forecast_months,
    cost_adjustment,
    efficiency_target,
    title_suffix,
    population_ratio=1.0,
    l=None,
):
    """
    Show cost forecast tab with filtered data
    """
    st.subheader(f"{l['cost_forecast_planning']}{title_suffix}")

    # Generate forecast using filtered data
    forecast_data, forecast_dates = generate_filtered_resource_forecast(
        filtered_resource_data,
        population_data,
        forecast_months,
        cost_adjustment,
        efficiency_target,
        population_ratio,
    )

    if forecast_data is not None:
        # Create cost forecast chart
        create_cost_forecast_chart(
            filtered_resource_data, forecast_data, forecast_dates
        )

        # Show cost projections
        show_cost_projections(filtered_resource_data, forecast_data, l)

        # Show cost breakdown
        show_cost_breakdown(forecast_data, forecast_dates, l)


def generate_filtered_resource_forecast(
    resource_data,
    population_data,
    months,
    cost_inflation,
    efficiency_improvement,
    population_ratio=1.0,
):
    """
    Generate resource forecast based on filtered data
    """
    try:
        # Get current values from filtered data
        current_cost = resource_data["total_monthly_cost"].iloc[-1]
        current_population = (
            population_data["total_prisoners"].iloc[-1] * population_ratio
            if population_ratio < 1.0
            else population_data["total_prisoners"].iloc[-1]
        )
        current_daily_cost = (
            current_cost / 30 / current_population if current_population > 0 else 0
        )

        # Generate population forecast (simplified)
        population_growth_rate = 0.02 / 12  # 2% annual growth, monthly
        population_forecast = []
        for i in range(months):
            future_pop = current_population * (1 + population_growth_rate) ** i
            population_forecast.append(future_pop)

        # Generate cost forecasts
        forecast_data = {
            "total_monthly_cost": [],
            "monthly_food_cost": [],
            "monthly_medical_cost": [],
            "monthly_utility_cost": [],
            "monthly_other_cost": [],
            "capacity_utilization": [],
            "daily_cost_per_prisoner": [],
            "energy_efficiency": [],
        }

        current_efficiency = resource_data["energy_efficiency"].iloc[-1]
        current_capacity = (
            resource_data["total_capacity"].iloc[-1] * population_ratio
            if population_ratio < 1.0
            else resource_data["total_capacity"].iloc[-1]
        )

        for i, pop in enumerate(population_forecast):
            # Apply cost inflation
            month_inflation = (cost_inflation / 100) / 12
            inflated_daily_cost = current_daily_cost * (1 + month_inflation) ** (i + 1)

            # Apply efficiency improvements
            month_efficiency = current_efficiency + (efficiency_improvement / 100) * (
                i / months
            )
            efficiency_factor = (
                current_efficiency / month_efficiency if month_efficiency > 0 else 1
            )

            # Calculate costs
            effective_daily_cost = inflated_daily_cost * efficiency_factor
            total_monthly_cost = pop * effective_daily_cost * 30

            # Cost breakdown
            monthly_food_cost = total_monthly_cost * 0.4
            monthly_medical_cost = total_monthly_cost * 0.15
            monthly_utility_cost = total_monthly_cost * 0.20
            monthly_other_cost = total_monthly_cost * 0.25

            # Store results
            forecast_data["total_monthly_cost"].append(total_monthly_cost)
            forecast_data["monthly_food_cost"].append(monthly_food_cost)
            forecast_data["monthly_medical_cost"].append(monthly_medical_cost)
            forecast_data["monthly_utility_cost"].append(monthly_utility_cost)
            forecast_data["monthly_other_cost"].append(monthly_other_cost)
            forecast_data["capacity_utilization"].append(
                min(100, (pop / current_capacity) * 100)
            )
            forecast_data["daily_cost_per_prisoner"].append(effective_daily_cost)
            forecast_data["energy_efficiency"].append(month_efficiency)

        # Generate forecast dates
        start_date = resource_data["date"].iloc[-1]
        forecast_dates = [
            start_date + timedelta(days=30 * i) for i in range(1, months + 1)
        ]

        return forecast_data, forecast_dates

    except Exception as e:
        st.error(f"Error generating forecast: {e}")
        return None, None


def create_cost_forecast_chart(historical_data, forecast_data, forecast_dates):
    """
    Create cost forecast chart with proper scaling
    """
    try:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Total Monthly Cost",
                "Cost per Prisoner",
                "Capacity Utilization",
                "Energy Efficiency",
            ),
        )

        # Historical and forecast costs
        historical_cost = historical_data["total_monthly_cost"] / 1000000
        forecast_cost = np.array(forecast_data["total_monthly_cost"]) / 1000000

        # Total monthly cost
        fig.add_trace(
            go.Scatter(
                x=historical_data["date"],
                y=historical_cost,
                mode="lines",
                name="Historical Cost",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast_cost,
                mode="lines",
                name="Forecast Cost",
                line=dict(color="red", width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )

        # Cost per prisoner
        fig.add_trace(
            go.Scatter(
                x=historical_data["date"],
                y=historical_data["daily_cost_per_prisoner"],
                mode="lines",
                name="Historical Cost/Prisoner",
                line=dict(color="blue", width=2),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast_data["daily_cost_per_prisoner"],
                mode="lines",
                name="Forecast Cost/Prisoner",
                line=dict(color="red", width=2, dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # Capacity utilization
        fig.add_trace(
            go.Scatter(
                x=historical_data["date"],
                y=historical_data["capacity_utilization"],
                mode="lines",
                name="Historical Utilization",
                line=dict(color="blue", width=2),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast_data["capacity_utilization"],
                mode="lines",
                name="Forecast Utilization",
                line=dict(color="red", width=2, dash="dash"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # Energy efficiency
        fig.add_trace(
            go.Scatter(
                x=historical_data["date"],
                y=historical_data["energy_efficiency"] * 100,
                mode="lines",
                name="Historical Efficiency",
                line=dict(color="blue", width=2),
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=np.array(forecast_data["energy_efficiency"]) * 100,
                mode="lines",
                name="Forecast Efficiency",
                line=dict(color="red", width=2, dash="dash"),
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title="Resource Forecast Dashboard", height=700, hovermode="x unified"
        )

        # Update axis labels
        fig.update_yaxes(title_text="Cost (Million MYR)", row=1, col=1)
        fig.update_yaxes(title_text="MYR per Day", row=1, col=2)
        fig.update_yaxes(title_text="Utilization %", row=2, col=1)
        fig.update_yaxes(title_text="Efficiency %", row=2, col=2)

        st.plotly_chart(fig, use_container_width=True, key="resource_analysis_chart")

    except Exception as e:
        st.error(f"Error creating chart: {e}")


def show_cost_projections(resource_data, forecast_data, l):
    """
    Show cost projection summary
    """
    st.subheader(l["cost_projections"])

    col1, col2, col3 = st.columns(3)

    current_cost = resource_data["total_monthly_cost"].iloc[-1]
    projected_cost = forecast_data["total_monthly_cost"][-1]
    cost_change = (
        ((projected_cost - current_cost) / current_cost) * 100
        if current_cost > 0
        else 0
    )
    annual_cost = projected_cost * 12

    with col1:
        st.metric(
            "Projected Monthly Cost",
            f"MYR {projected_cost/1000000:.1f}M",
            f"{cost_change:+.1f}%",
        )

    with col2:
        st.metric("Annual Cost Projection", f"MYR {annual_cost/1000000:.1f}M")

    with col3:
        efficiency_savings = current_cost * 0.05  # 5% efficiency target
        st.metric(
            "Potential Efficiency Savings",
            f"MYR {efficiency_savings/1000000:.1f}M/month",
        )


def show_cost_breakdown(forecast_data, forecast_dates, l):
    """
    Show cost breakdown over time
    """
    st.subheader(l["cost_breakdown_forecast"])

    # Create stacked area chart
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast_data["monthly_food_cost"],
            fill="tonexty",
            name="Food Costs",
            line=dict(color="green"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=np.array(forecast_data["monthly_food_cost"])
            + np.array(forecast_data["monthly_medical_cost"]),
            fill="tonexty",
            name="Medical Costs",
            line=dict(color="red"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=np.array(forecast_data["monthly_food_cost"])
            + np.array(forecast_data["monthly_medical_cost"])
            + np.array(forecast_data["monthly_utility_cost"]),
            fill="tonexty",
            name="Utility Costs",
            line=dict(color="blue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast_data["total_monthly_cost"],
            fill="tonexty",
            name="Other Costs",
            line=dict(color="orange"),
        )
    )

    fig.update_layout(
        title="Cost Breakdown Over Time",
        xaxis_title="Date",
        yaxis_title="Cost (MYR)",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True, key="cost_projections_chart")


def show_capacity_planning_tab(resource_data, population_data, title_suffix, l):
    """
    Show capacity planning tab
    """
    st.subheader(f"{l['capacity_planning_infrastructure']}{title_suffix}")
    st.info(l["capacity_planning_analysis"])

    # Simple capacity metrics
    col1, col2 = st.columns(2)

    with col1:
        current_capacity = resource_data["total_capacity"].iloc[-1]
        current_population = population_data["total_prisoners"].iloc[-1]
        utilization = (current_population / current_capacity) * 100

        st.metric("Current Capacity", f"{current_capacity:,}")
        st.metric("Current Utilization", f"{utilization:.1f}%")

    with col2:
        available_capacity = current_capacity - current_population
        st.metric("Available Capacity", f"{available_capacity:,}")

        if utilization > 90:
            st.warning("⚠️ High utilization - consider capacity expansion")
        elif utilization < 60:
            st.success("✅ Good capacity availability")


def show_efficiency_analysis_tab(resource_data, title_suffix, l):
    """
    Show efficiency analysis tab
    """
    st.subheader(f"{l['efficiency_analysis_optimization']}{title_suffix}")
    st.info(l["energy_efficiency_analysis"])

    # Simple efficiency metrics
    current_efficiency = resource_data["energy_efficiency"].iloc[-1]
    food_waste = resource_data["food_waste_rate"].iloc[-1]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Energy Efficiency", f"{current_efficiency:.1%}")
        if current_efficiency < 0.7:
            st.warning("⚠️ Low energy efficiency - improvement needed")
        else:
            st.success("✅ Good energy efficiency")

    with col2:
        st.metric("Food Waste Rate", f"{food_waste:.1%}")
        if food_waste > 0.15:
            st.warning("⚠️ High food waste - reduction opportunities")
        else:
            st.success("✅ Acceptable food waste levels")


def show_resource_optimization_tab(resource_data, population_data, title_suffix, l):
    """
    Show resource optimization tab
    """
    st.subheader(f"{l['resource_optimization_planning']}{title_suffix}")
    st.info(l["resource_optimization_recommendations"])

    # Simple optimization recommendations
    st.write(l["optimization_opportunities"])

    current_cost = resource_data["total_monthly_cost"].iloc[-1]
    cost_per_prisoner = current_cost / population_data["total_prisoners"].iloc[-1]

    recommendations = []

    if cost_per_prisoner > 1500:  # If cost per prisoner per month > MYR 1500
        recommendations.append(
            "💡 High per-prisoner costs - review operational efficiency"
        )

    if resource_data["food_waste_rate"].iloc[-1] > 0.12:
        recommendations.append("🍽️ Food waste reduction programs recommended")

    if resource_data["energy_efficiency"].iloc[-1] < 0.75:
        recommendations.append("⚡ Energy efficiency upgrades needed")

    if not recommendations:
        recommendations.append("✅ Current operations are well-optimized")

    for rec in recommendations:
        st.write(f"• {rec}")

    # Cost summary
    st.write(l["cost_summary"])
    st.write(l["monthly_cost_per_prisoner"].format(cost_per_prisoner))
    st.write(l["annual_cost_projection"].format(current_cost * 12 / 1000000))
