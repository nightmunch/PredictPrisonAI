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

# Language dictionaries
LANG = {
    "en": {
        "page_title": "Malaysia Prison Predictive Planning",
        "title": "\U0001f3ea Malaysia Prison Predictive Planning System",
        "navigation": "Navigation",
        "select_page": "Select a page:",
        "pages": [
            "Dashboard Overview",
            "Population Forecast",
            # "Staffing Forecast",  # Feature disabled
            "Resource Forecast",
            "Model Performance",
        ],
        "dashboard_header": "\U0001f4ca Dashboard Overview",
        "state_selector": "Select State:",
        "prison_selector": "Select Prison:",
        "all_states": "All States",
        "all_prisons": "All Prisons",
        "all_prisons_in_state": "All Prisons in State",
        "prison_system_by_state": "\U0001f5fa\ufe0f Prison System by State",
        "current_population": "Current Prison Population",
        "daily_cost": "Daily Cost per Prisoner",
        "capacity_utilization": "Capacity Utilization",
        "avg_sentence": "Avg Sentence Length",
        "months": "months",
        "male_prisoners": "Male Prisoners",
        "female_prisoners": "Female Prisoners",
        "drug_crimes": "Drug Crimes",
        "violent_crimes": "Violent Crimes",
        "prison_population_by_state": "\U0001f5fa\ufe0f Prison Population by State",
        "historical_trends": "\U0001f4c8 Historical Trends",
        "system_info": "\u2139\ufe0f System Information",
        "data_coverage": "**Data Coverage:**\n- Historical data: 5 years\n- Forecast horizon: 2 years\n- Update frequency: Monthly",
        "models_loaded": "\u2705 Models loaded and ready",
        "models_not_available": "\u274c Models not available",
        "quick_actions": "\U0001f680 Quick Actions",
        "generate_forecast": "Generate New Forecast",
        "export_data": "Export Current Data",
        "download_population_data": "Download Population Data",
        "model_status_check": "Model Status Check",
        "all_models_operational": "All models are operational",
        "some_models_need_retraining": "Some models may need retraining",
        "failed_load_data": "Failed to load data. Please check the data generation process.",
        "error_loading": "Error loading data or models: ",
        "ensure_data": "Please ensure the data generation notebook has been run and models are trained.",
        "language": "Language",
        "english": "English",
        "malay": "Malay",
        "no_historical_data": "No historical data available for the selected state/prison.",
        "no_model_status": "No model status information available.",
    },
    "ms": {
        "page_title": "Sistem Perancangan Prediktif Penjara Malaysia",
        "title": "\U0001f3ea Sistem Perancangan Prediktif Penjara Malaysia",
        "navigation": "Navigasi",
        "select_page": "Pilih halaman:",
        "pages": [
            "Papan Pemuka Utama",
            "Ramalan Populasi",
            # "Ramalan Kakitangan",  # Ciri dilumpuhkan
            "Ramalan Sumber",
            "Prestasi Model",
        ],
        "dashboard_header": "\U0001f4ca Papan Pemuka Utama",
        "state_selector": "Pilih Negeri:",
        "prison_selector": "Pilih Penjara:",
        "all_states": "Semua Negeri",
        "all_prisons": "Semua Penjara",
        "all_prisons_in_state": "Semua Penjara dalam Negeri",
        "prison_system_by_state": "\U0001f5fa\ufe0f Sistem Penjara mengikut Negeri",
        "current_population": "Populasi Penjara Semasa",
        "daily_cost": "Kos Harian per Banduan",
        "capacity_utilization": "Penggunaan Kapasiti",
        "avg_sentence": "Purata Tempoh Hukuman",
        "months": "bulan",
        "male_prisoners": "Banduan Lelaki",
        "female_prisoners": "Banduan Perempuan",
        "drug_crimes": "Jenayah Dadah",
        "violent_crimes": "Jenayah Kekerasan",
        "prison_population_by_state": "\U0001f5fa\ufe0f Populasi Penjara mengikut Negeri",
        "historical_trends": "\U0001f4c8 Trend Sejarah",
        "system_info": "\u2139\ufe0f Maklumat Sistem",
        "data_coverage": "**Liputan Data:**\n- Data sejarah: 5 tahun\n- Horizon ramalan: 2 tahun\n- Kekerapan kemas kini: Bulanan",
        "models_loaded": "\u2705 Model dimuatkan dan sedia",
        "models_not_available": "\u274c Model tidak tersedia",
        "quick_actions": "\U0001f680 Tindakan Pantas",
        "generate_forecast": "Jana Ramalan Baharu",
        "export_data": "Eksport Data Semasa",
        "download_population_data": "Muat Turun Data Populasi",
        "model_status_check": "Semak Status Model",
        "all_models_operational": "Semua model beroperasi",
        "some_models_need_retraining": "Sesetengah model mungkin perlu dilatih semula",
        "failed_load_data": "Gagal memuatkan data. Sila semak proses penjanaan data.",
        "error_loading": "Ralat memuatkan data atau model: ",
        "ensure_data": "Sila pastikan buku nota penjanaan data telah dijalankan dan model telah dilatih.",
        "language": "Bahasa",
        "english": "Inggeris",
        "malay": "Melayu",
        "no_historical_data": "Tiada data sejarah untuk negeri/penjara yang dipilih.",
        "no_model_status": "Tiada maklumat status model tersedia.",
    },
}

# Configure the page (default to English)
st.set_page_config(
    page_title=LANG["en"]["page_title"],
    page_icon="\U0001f3ea",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    # Language selection
    lang = st.sidebar.selectbox(
        LANG["en"]["language"] + " / " + LANG["ms"]["language"],
        options=["en", "ms"],
        format_func=lambda x: LANG[x]["english"] if x == "en" else LANG[x]["malay"],
    )
    l = LANG[lang]
    st.title(l["title"])
    st.markdown("---")

    # Sidebar navigation
    st.sidebar.title(l["navigation"])
    page = st.sidebar.selectbox(
        l["select_page"],
        l["pages"],
    )

    # Load data and models
    try:
        data = load_or_generate_data()
        models = load_models()
        if data is None:
            st.error(l["failed_load_data"])
            return
    except Exception as e:
        st.error(l["error_loading"] + str(e))
        st.info(l["ensure_data"])
        return

    # Route to different pages
    if page == l["pages"][0]:
        show_dashboard_overview(data, models, l)
    elif page == l["pages"][1]:
        from modules.population_forecast import show_population_forecast

        show_population_forecast(data, models)
    # elif page == l['pages'][2]:  # Staffing/Staffing Forecast (disabled)
    #     from modules.staffing_forecast import show_staffing_forecast
    #     show_staffing_forecast(data, models)
    elif page == l["pages"][2]:
        from modules.resource_forecast import show_resource_forecast

        show_resource_forecast(data, models)
    elif page == l["pages"][3]:
        from modules.model_performance import show_model_performance

        show_model_performance(data, models)


def show_dashboard_overview(data, models, l):
    st.header(l["dashboard_header"])

    # State and Prison Selector
    selected_state = l["all_states"]
    selected_prison = l["all_prisons"]

    if "prison_detail_data" in data and "malaysia_prisons" in data:
        st.subheader(l["prison_system_by_state"])
        col1, col2 = st.columns(2)
        with col1:
            selected_state = st.selectbox(
                l["state_selector"],
                [l["all_states"]] + list(data["malaysia_prisons"].keys()),
            )
        with col2:
            if selected_state != l["all_states"]:
                prisons_in_state = data["malaysia_prisons"][selected_state]
                prison_names = []
                for prison in prisons_in_state:
                    if isinstance(prison, dict):
                        prison_names.append(prison["name"])
                    else:
                        prison_names.append(prison)
                selected_prison = st.selectbox(
                    l["prison_selector"], [l["all_prisons_in_state"]] + prison_names
                )
            else:
                selected_prison = l["all_prisons"]
        st.markdown("---")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    if "population_data" in data:
        current_population = data["population_data"]["total_prisoners"].iloc[-1]
        col1.metric(l["current_population"], f"{current_population:,.0f}")
    if "resource_data" in data:
        daily_cost = data["resource_data"]["daily_cost_per_prisoner"].iloc[-1]
        col2.metric(
            l["daily_cost"],
            (
                f"MYR {daily_cost:.2f}"
                if l["daily_cost"].startswith("Daily")
                else f"RM {daily_cost:.2f}"
            ),
        )
    if "resource_data" in data:
        capacity_utilization = data["resource_data"]["capacity_utilization"].iloc[-1]
        col3.metric(l["capacity_utilization"], f"{capacity_utilization:.1f}%")
    if "population_data" in data:
        avg_sentence = data["population_data"]["avg_sentence_months"].iloc[-1]
        col4.metric(l["avg_sentence"], f"{avg_sentence:.1f} {l['months']}")
    st.markdown("---")

    # State/Prison specific data display
    if "prison_detail_data" in data and "malaysia_prisons" in data:
        if selected_state != l["all_states"]:
            if selected_prison != l["all_prisons_in_state"]:
                prison_data = data["prison_detail_data"][
                    (data["prison_detail_data"]["state"] == selected_state)
                    & (data["prison_detail_data"]["prison_name"] == selected_prison)
                ]
                if not prison_data.empty:
                    latest_data = prison_data.iloc[-1]
                    st.info(
                        f"**{selected_prison}** - {l['current_population'] if l['current_population'].startswith('Current') else 'Populasi Semasa'}: {latest_data['prison_population']:,} {'prisoners' if l['current_population'].startswith('Current') else 'banduan'}"
                    )
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric(
                        l["male_prisoners"], f"{latest_data['male_prisoners']:,}"
                    )
                    col2.metric(
                        l["female_prisoners"], f"{latest_data['female_prisoners']:,}"
                    )
                    col3.metric(l["drug_crimes"], f"{latest_data['drug_crimes']:,}")
                    col4.metric(
                        l["violent_crimes"], f"{latest_data['violent_crimes']:,}"
                    )
            else:
                state_data = data["prison_detail_data"][
                    data["prison_detail_data"]["state"] == selected_state
                ]
                if not state_data.empty:
                    latest_state = state_data[
                        state_data["date"] == state_data["date"].max()
                    ]
                    total_state_pop = latest_state["prison_population"].sum()
                    st.info(
                        f"**{selected_state}{' State' if l['all_states']=='All States' else ''}** - {'Total Population' if l['all_states']=='All States' else 'Jumlah Populasi'}: {total_state_pop:,} {'prisoners' if l['all_states']=='All States' else 'banduan'} {'across' if l['all_states']=='All States' else 'merentasi'} {len(data['malaysia_prisons'][selected_state])} {'prisons' if l['all_states']=='All States' else 'penjara'}"
                    )
                    st.write(
                        f"**{'Prisons in this state:' if l['all_states']=='All States' else 'Penjara dalam negeri ini:'}**"
                    )
                    for prison in data["malaysia_prisons"][selected_state]:
                        prison_name = (
                            prison["name"] if isinstance(prison, dict) else prison
                        )
                        prison_match = latest_state[
                            latest_state["prison_name"] == prison_name
                        ]
                        if not prison_match.empty:
                            prison_pop = prison_match["prison_population"].iloc[0]
                            st.write(
                                f"\u2022 {prison_name}: {prison_pop:,} {'prisoners' if l['all_states']=='All States' else 'banduan'}"
                            )
        # State comparison chart
        st.subheader(l["prison_population_by_state"])
        latest_detail = data["prison_detail_data"][
            data["prison_detail_data"]["date"]
            == data["prison_detail_data"]["date"].max()
        ]
        state_totals = (
            latest_detail.groupby("state")["prison_population"]
            .sum()
            .sort_values(ascending=True)
        )
        import plotly.express as px

        fig = px.bar(
            x=state_totals.values,
            y=state_totals.index,
            orientation="h",
            title=l["prison_population_by_state"],
            labels={"x": l["current_population"], "y": l["all_states"]},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Overview charts
    st.subheader(l["historical_trends"])
    if data:
        plot_overview_metrics(data)

    # System information
    st.markdown("---")
    st.subheader(l["system_info"])
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.info(l["data_coverage"])
    with info_col2:
        if models:
            st.success(l["models_loaded"])
        else:
            st.error(l["models_not_available"])

    # Quick actions
    st.markdown("---")
    st.subheader(l["quick_actions"])
    action_col1, action_col2, action_col3 = st.columns(3)
    with action_col1:
        if st.button(l["generate_forecast"]):
            st.info(
                "Navigate to specific forecast pages to generate predictions"
                if l["generate_forecast"].startswith("Generate")
                else "Navigasi ke halaman ramalan khusus untuk menjana prediksi"
            )
    with action_col2:
        if st.button(l["export_data"]):
            if data:
                st.download_button(
                    label=l["download_population_data"],
                    data=data["population_data"].to_csv(index=False),
                    file_name=(
                        f"prison_population_data_{datetime.now().strftime('%Y%m%d')}.csv"
                        if l["download_population_data"].startswith("Download")
                        else f"data_populasi_penjara_{datetime.now().strftime('%Y%m%d')}.csv"
                    ),
                    mime="text/csv",
                )
    with action_col3:
        if st.button("Export State/Prison Detail Data"):
            if data:
                # Filter by current state/prison selection if not 'All States'
                detail_df = data["prison_detail_data"]
                if (
                    "selected_state" in locals()
                    and selected_state != l["all_states"]
                    and selected_state in detail_df["state"].unique()
                ):
                    detail_df = detail_df[detail_df["state"] == selected_state]
                    if (
                        "selected_prison" in locals()
                        and selected_prison != l["all_prisons_in_state"]
                        and selected_prison in detail_df["prison_name"].unique()
                    ):
                        detail_df = detail_df[
                            detail_df["prison_name"] == selected_prison
                        ]
                st.download_button(
                    label="Download State/Prison Detail Data",
                    data=detail_df.to_csv(index=False),
                    file_name=f"prison_detail_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )

    # System information
    st.markdown("---")
    st.subheader(l["system_info"])
    st.markdown(l["data_coverage"])
    if "model_status" in data:
        model_status = data["model_status"]
        if model_status["all_models_operational"]:
            st.success(l["all_models_operational"])
        else:
            st.warning(l["some_models_need_retraining"])
            if "retraining_info" in model_status:
                st.info(model_status["retraining_info"])
    else:
        st.info(l["no_model_status"])


if __name__ == "__main__":
    main()
