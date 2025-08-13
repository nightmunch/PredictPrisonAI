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

from utils.visualization import (
    plot_population_forecast,
    plot_trend_analysis,
    plot_seasonal_decomposition,
)
from utils.data_utils import prepare_forecast_data, calculate_growth_rate


def show_population_forecast(data, models, l):
    """
    Display population forecasting page
    """
    st.header(l["population_forecast_header"])
    st.markdown("---")

    if "population_data" not in data:
        st.error("Population data not available")
        return

    population_data = data["population_data"]

    # Sidebar controls
    st.sidebar.subheader(l["forecast_parameters"])

    # Geographic selection
    forecast_level = st.sidebar.selectbox(
        l["forecast_level"], [l["overall_malaysia"], l["by_state"], l["by_prison"]]
    )

    selected_state = None
    selected_prison = None

    if forecast_level == l["by_state"]:
        states = list(data["malaysia_prisons"].keys())
        selected_state = st.sidebar.selectbox(l["select_state"], states)
    elif forecast_level == l["by_prison"]:
        states = list(data["malaysia_prisons"].keys())
        selected_state = st.sidebar.selectbox(l["select_state"], states)
        prisons = data["malaysia_prisons"][selected_state]
        prison_names = []
        for p in prisons:
            if isinstance(p, dict):
                prison_names.append(p["name"])
            else:
                prison_names.append(p)
        prison_options = [l["all_prisons_in_state"]] + prison_names
        selected_prison = st.sidebar.selectbox(l["select_prison"], prison_options)
        if selected_prison == l["all_prisons_in_state"]:
            selected_prison = None

    forecast_months = st.sidebar.slider(
        l["forecast_period"], min_value=6, max_value=36, value=24, step=6
    )

    scenario = st.sidebar.selectbox(
        l["scenario_analysis"], [l["base_case"], l["enhanced_rehabilitation"], l["economic_impact"]]
    )

    # Filter data based on selection
    if forecast_level == "By State" and selected_state:
        filtered_data = data["prison_detail_data"][
            data["prison_detail_data"]["state"] == selected_state
        ]
        display_data = (
            filtered_data.groupby("date")
            .agg(
                {
                    "prison_population": "sum",
                    "male_prisoners": "sum",
                    "female_prisoners": "sum",
                    "drug_crimes": "sum",
                }
            )
            .reset_index()
        )
        title_suffix = f" - {selected_state} State"
    elif (
        forecast_level == "By Prison"
        and selected_prison
        and selected_prison != "All Prisons in State"
    ):
        filtered_data = data["prison_detail_data"][
            (data["prison_detail_data"]["state"] == selected_state)
            & (data["prison_detail_data"]["prison_name"] == selected_prison)
        ]
        display_data = filtered_data.rename(
            columns={"prison_population": "prison_population"}
        )
        title_suffix = f" - {selected_prison}"
    else:
        display_data = population_data
        title_suffix = " - Malaysia Overall"

    # Current statistics
    col1, col2, col3, col4 = st.columns(4)

    if forecast_level == "By State" and selected_state:
        current_population = (
            display_data["prison_population"].iloc[-1] if not display_data.empty else 0
        )
        growth_rate = (
            calculate_growth_rate(display_data, "prison_population", 12)
            if not display_data.empty
            else 0
        )
        male_percentage = (
            (display_data["male_prisoners"].iloc[-1] / current_population * 100)
            if current_population > 0
            else 0
        )
        female_percentage = (
            (display_data["female_prisoners"].iloc[-1] / current_population * 100)
            if current_population > 0
            else 0
        )
    elif (
        forecast_level == "By Prison"
        and selected_prison
        and selected_prison != "All Prisons in State"
    ):
        current_population = (
            display_data["prison_population"].iloc[-1] if not display_data.empty else 0
        )
        growth_rate = (
            calculate_growth_rate(display_data, "prison_population", 12)
            if not display_data.empty
            else 0
        )
        male_percentage = (
            (display_data["male_prisoners"].iloc[-1] / current_population * 100)
            if current_population > 0
            else 0
        )
        female_percentage = (
            (display_data["female_prisoners"].iloc[-1] / current_population * 100)
            if current_population > 0
            else 0
        )
    else:
        current_population = population_data["total_prisoners"].iloc[-1]
        growth_rate = calculate_growth_rate(population_data, "total_prisoners", 12)
        male_percentage = (
            population_data["male_prisoners"].iloc[-1] / current_population
        ) * 100
        female_percentage = (
            population_data["female_prisoners"].iloc[-1] / current_population
        ) * 100
    avg_sentence = population_data["avg_sentence_months"].iloc[-1]

    col1.metric("Current Population", f"{current_population:,.0f}")
    col2.metric("Annual Growth Rate", f"{growth_rate:.1f}%" if growth_rate else "N/A")
    if male_percentage is not None:
        col3.metric("Male Population", f"{male_percentage:.1f}%")
    if female_percentage is not None:
        col4.metric("Female Population", f"{female_percentage:.1f}%")
    else:
        col4.metric("Avg Sentence", f"{avg_sentence:.1f} months")

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Forecast", "🔍 Analysis", "📊 Demographics", "📋 Summary"]
    )

    with tab1:
        st.subheader(f"Population Forecast{title_suffix}")

        # Generate forecast using appropriate data structure
        if forecast_level == "By State" and selected_state:
            # Use aggregated state data but with correct column names
            forecast_input = display_data.rename(
                columns={"prison_population": "total_prisoners"}
            )
        elif (
            forecast_level == "By Prison"
            and selected_prison
            and selected_prison != "All Prisons in State"
        ):
            # Use individual prison data with correct column names
            forecast_input = display_data.rename(
                columns={"prison_population": "total_prisoners"}
            )
        else:
            # Use overall population data
            forecast_input = population_data

        forecast_data, forecast_dates = generate_population_forecast(
            forecast_input, forecast_months, scenario, models
        )

        if forecast_data is not None:
            # Plot forecast using the appropriate data
            plot_population_forecast(forecast_input, forecast_data, forecast_dates)

            # Forecast statistics
            st.subheader("Forecast Statistics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Projected Population (End)",
                    f"{forecast_data[-1]:,.0f}",
                    f"{((forecast_data[-1] - current_population) / current_population * 100):+.1f}%",
                )

            with col2:
                max_population = max(forecast_data)
                st.metric(
                    "Peak Population",
                    f"{max_population:,.0f}",
                    f"{((max_population - current_population) / current_population * 100):+.1f}%",
                )

            with col3:
                avg_forecast = np.mean(forecast_data)
                st.metric(
                    "Average (Forecast Period)",
                    f"{avg_forecast:,.0f}",
                    f"{((avg_forecast - current_population) / current_population * 100):+.1f}%",
                )

            # Scenario impact
            st.subheader("Scenario Impact")
            
            # Add Malaysian context explanations
            with st.expander("📊 Scenario Explanations", expanded=False):
                st.markdown("""
                **Base Case**: Current trend continuation based on historical patterns
                
                **Enhanced Rehabilitation**: Impact of improved rehabilitation programs reducing recidivism rates, aligned with Malaysia's prison reform initiatives
                
                **Economic Impact**: Population changes due to economic fluctuations affecting crime rates and sentencing patterns
                """)
            
            create_scenario_comparison(population_data, forecast_months, models)

        else:
            st.warning("Unable to generate forecast. Please check model availability.")

    with tab2:
        st.subheader("Population Analysis")

        # Trend analysis
        plot_trend_analysis(
            population_data, "total_prisoners", "Prison Population Trend Analysis"
        )

        # Seasonal patterns
        st.subheader("Seasonal Patterns")
        plot_seasonal_decomposition(population_data, "total_prisoners")

        # Admission and release analysis
        st.subheader("Admission vs Release Trends")
        create_flow_analysis(population_data)

    with tab3:
        st.subheader("Demographic Breakdown")

        # Gender distribution over time
        create_gender_analysis(population_data)

        # Age group analysis
        create_age_group_analysis(population_data)

        # Crime type analysis
        create_crime_type_analysis(population_data)

    with tab4:
        st.subheader("Summary Report")

        # Key insights
        create_summary_report(
            population_data, forecast_data if "forecast_data" in locals() else None
        )


def generate_population_forecast(data, months, scenario, models):
    """
    Generate population forecast based on scenario
    """
    try:
        # Prepare forecast dates
        last_date = data["date"].max()
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1), periods=months, freq="ME"
        )

        # Base forecast (using simple trend + seasonality)
        base_population = data["total_prisoners"].iloc[-1]

        # Calculate trend
        recent_data = data.tail(12)
        if len(recent_data) > 1:
            trend_slope = (
                recent_data["total_prisoners"].iloc[-1]
                - recent_data["total_prisoners"].iloc[0]
            ) / 12
        else:
            trend_slope = 0

        # Generate base forecast
        forecast = []
        for i in range(months):
            # Trend component
            trend_value = base_population + (trend_slope * (i + 1))

            # Seasonal component (simplified)
            seasonal_factor = 1 + 0.03 * np.sin(2 * np.pi * (i % 12) / 12)

            # Scenario adjustments
            scenario_factor = get_scenario_factor(scenario, i)

            forecasted_value = trend_value * seasonal_factor * scenario_factor
            forecast.append(max(0, forecasted_value))  # Ensure non-negative

        return forecast, forecast_dates

    except Exception as e:
        st.error(f"Error generating forecast: {e}")
        return None, None


def get_scenario_factor(scenario, month_index):
    """
    Get scenario adjustment factor for Malaysian prison context
    """
    if scenario == "Base Case":
        return 1.0
    elif scenario == "Enhanced Rehabilitation":
        # Gradual decrease due to improved rehabilitation programs reducing recidivism
        # More aggressive reduction as programs mature
        return 1.0 - (month_index * 0.004)
    elif scenario == "Economic Impact":
        # Economic fluctuations affecting crime rates
        # Initial increase then stabilization
        if month_index <= 12:
            return 1.0 + (month_index * 0.002)
        else:
            return 1.024 - ((month_index - 12) * 0.001)
    else:
        return 1.0


def create_scenario_comparison(data, months, models):
    """
    Create scenario comparison visualization for Malaysian prison context
    """
    try:
        scenarios = ["Base Case", "Enhanced Rehabilitation", "Economic Impact"]
        fig = go.Figure()

        # Historical data
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["total_prisoners"],
                mode="lines",
                name="Historical",
                line=dict(color="blue", width=2),
            )
        )

        colors = ["red", "green", "orange"]  # Base Case, Enhanced Rehabilitation, Economic Impact

        for i, scenario in enumerate(scenarios):
            forecast_data, forecast_dates = generate_population_forecast(
                data, months, scenario, models
            )

            if forecast_data:
                fig.add_trace(
                    go.Scatter(
                        x=forecast_dates,
                        y=forecast_data,
                        mode="lines",
                        name=scenario,
                        line=dict(color=colors[i], width=2, dash="dash"),
                    )
                )

        fig.update_layout(
            title="Population Forecast - Scenario Comparison",
            xaxis_title="Date",
            yaxis_title="Number of Prisoners",
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True, key="population_forecast_main_chart")

    except Exception as e:
        st.error(f"Error creating scenario comparison: {e}")


def create_flow_analysis(data):
    """
    Create admission and release flow analysis
    """
    try:
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("Monthly Admissions", "Monthly Releases")
        )

        # Admissions
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["monthly_admissions"],
                mode="lines",
                name="Admissions",
                line=dict(color="red", width=2),
            ),
            row=1,
            col=1,
        )

        # Releases
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["monthly_releases"],
                mode="lines",
                name="Releases",
                line=dict(color="green", width=2),
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title="Prison Population Flow Analysis", showlegend=False, height=400
        )

        st.plotly_chart(fig, use_container_width=True, key="capacity_analysis_chart")

        # Net flow calculation
        net_flow = data["monthly_admissions"] - data["monthly_releases"]
        avg_net_flow = net_flow.mean()

        st.metric("Average Net Monthly Flow", f"{avg_net_flow:+.0f} prisoners")

    except Exception as e:
        st.error(f"Error creating flow analysis: {e}")


def create_gender_analysis(data):
    """
    Create gender distribution analysis
    """
    try:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["male_prisoners"],
                mode="lines",
                name="Male Prisoners",
                line=dict(color="blue", width=2),
                stackgroup="one",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["female_prisoners"],
                mode="lines",
                name="Female Prisoners",
                line=dict(color="pink", width=2),
                stackgroup="one",
            )
        )

        fig.update_layout(
            title="Gender Distribution Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Prisoners",
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True, key="population_distribution_chart")

    except Exception as e:
        st.error(f"Error creating gender analysis: {e}")


def create_age_group_analysis(data):
    """
    Create age group distribution analysis
    """
    try:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["young_prisoners"],
                mode="lines",
                name="Young (18-30)",
                line=dict(color="green", width=2),
                stackgroup="one",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["middle_prisoners"],
                mode="lines",
                name="Middle (31-50)",
                line=dict(color="orange", width=2),
                stackgroup="one",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["old_prisoners"],
                mode="lines",
                name="Older (50+)",
                line=dict(color="red", width=2),
                stackgroup="one",
            )
        )

        fig.update_layout(
            title="Age Group Distribution Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Prisoners",
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating age group analysis: {e}")


def create_crime_type_analysis(data):
    """
    Create crime type distribution analysis
    """
    try:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["drug_crimes"],
                mode="lines",
                name="Drug Crimes",
                line=dict(color="red", width=2),
                stackgroup="one",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["violent_crimes"],
                mode="lines",
                name="Violent Crimes",
                line=dict(color="orange", width=2),
                stackgroup="one",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["property_crimes"],
                mode="lines",
                name="Property Crimes",
                line=dict(color="blue", width=2),
                stackgroup="one",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["other_crimes"],
                mode="lines",
                name="Other Crimes",
                line=dict(color="gray", width=2),
                stackgroup="one",
            )
        )

        fig.update_layout(
            title="Crime Type Distribution Over Time",
            xaxis_title="Date",
            yaxis_title="Number of Prisoners",
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating crime type analysis: {e}")


def create_summary_report(data, forecast_data):
    """
    Create summary report with key insights
    """
    try:
        # Current statistics
        current_pop = data["total_prisoners"].iloc[-1]
        growth_rate = calculate_growth_rate(data, "total_prisoners", 12)

        # Crime distribution
        latest_data = data.iloc[-1]
        drug_percentage = (
            latest_data["drug_crimes"] / latest_data["total_prisoners"]
        ) * 100
        violent_percentage = (
            latest_data["violent_crimes"] / latest_data["total_prisoners"]
        ) * 100

        # Summary text
        st.markdown(
            f"""
        ### Key Insights
        
        **Current Status:**
        - Total prison population: {current_pop:,.0f}
        - Annual growth rate: {growth_rate:.1f}% (last 12 months)
        - Average sentence length: {latest_data['avg_sentence_months']:.1f} months
        
        **Crime Profile:**
        - Drug-related crimes: {drug_percentage:.1f}% of population
        - Violent crimes: {violent_percentage:.1f}% of population
        - Property crimes: {(latest_data['property_crimes'] / latest_data['total_prisoners']) * 100:.1f}% of population
        
        **Population Flow:**
        - Average monthly admissions: {data['monthly_admissions'].mean():.0f}
        - Average monthly releases: {data['monthly_releases'].mean():.0f}
        - Net monthly change: {(data['monthly_admissions'] - data['monthly_releases']).mean():+.0f}
        """
        )

        if forecast_data:
            forecast_end = forecast_data[-1]
            forecast_change = ((forecast_end - current_pop) / current_pop) * 100

            st.markdown(
                f"""
            **Forecast Summary:**
            - Projected population: {forecast_end:,.0f}
            - Expected change: {forecast_change:+.1f}%
            - Peak forecast: {max(forecast_data):,.0f}
            """
            )

        # Export functionality
        st.subheader("Export Data")

        col1, col2 = st.columns(2)

        with col1:
            csv_data = data.to_csv(index=False)
            st.download_button(
                label="Download Historical Data",
                data=csv_data,
                file_name=f"population_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

        with col2:
            if forecast_data:
                forecast_df = pd.DataFrame(
                    {
                        "date": pd.date_range(
                            start=data["date"].max() + pd.DateOffset(months=1),
                            periods=len(forecast_data),
                            freq="ME",
                        ),
                        "forecasted_population": forecast_data,
                    }
                )

                forecast_csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="Download Forecast Data",
                    data=forecast_csv,
                    file_name=f"population_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.error(f"Error creating summary report: {e}")
