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
    page_title="Sistem Perancangan Prediktif Penjara Malaysia",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.title("🏢 Sistem Perancangan Prediktif Penjara Malaysia")
    st.markdown("---")

    # Sidebar navigation
    st.sidebar.title("Navigasi")
    page = st.sidebar.selectbox(
        "Pilih halaman:",
        [
            "Papan Pemuka Utama",
            "Ramalan Populasi",
            # "Ramalan Kakitangan",  # Dikomen - ciri dilumpuhkan
            "Ramalan Sumber",
            "Prestasi Model",
        ],
    )

    # Load data and models
    try:
        data = load_or_generate_data()
        models = load_models()

        if data is None:
            st.error("Gagal memuatkan data. Sila semak proses penjanaan data.")
            return

    except Exception as e:
        st.error(f"Ralat memuatkan data atau model: {str(e)}")
        st.info(
            "Sila pastikan buku nota penjanaan data telah dijalankan dan model telah dilatih."
        )
        return

    # Route to different pages
    if page == "Papan Pemuka Utama":
        show_dashboard_overview(data, models)
    elif page == "Ramalan Populasi":
        from modules.population_forecast import show_population_forecast

        show_population_forecast(data, models)
    # elif page == "Ramalan Kakitangan":  # Dikomen - ciri dilumpuhkan
    #     from modules.staffing_forecast import show_staffing_forecast
    #     show_staffing_forecast(data, models)
    elif page == "Ramalan Sumber":
        from modules.resource_forecast import show_resource_forecast

        show_resource_forecast(data, models)
    elif page == "Prestasi Model":
        from modules.model_performance import show_model_performance

        show_model_performance(data, models)


def show_dashboard_overview(data, models):
    """Paparkan papan pemuka utama"""
    st.header("📊 Papan Pemuka Utama")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    if "population_data" in data:
        current_population = data["population_data"]["total_prisoners"].iloc[-1]
        col1.metric("Populasi Penjara Semasa", f"{current_population:,.0f}")

    # if 'staffing_data' in data:  # Dikomen - ciri kakitangan dilumpuhkan
    #     current_staff = data['staffing_data']['total_staff'].iloc[-1]
    #     col2.metric("Bilangan Kakitangan Semasa", f"{current_staff:,.0f}")

    if "resource_data" in data:
        daily_cost = data["resource_data"]["daily_cost_per_prisoner"].iloc[-1]
        col2.metric("Kos Harian per Banduan", f"RM {daily_cost:.2f}")

    if "resource_data" in data:
        capacity_utilization = data["resource_data"]["capacity_utilization"].iloc[-1]
        col3.metric("Penggunaan Kapasiti", f"{capacity_utilization:.1f}%")

    if "population_data" in data:
        avg_sentence = data["population_data"]["avg_sentence_months"].iloc[-1]
        col4.metric("Purata Tempoh Hukuman", f"{avg_sentence:.1f} bulan")

    st.markdown("---")

    # Overview charts
    st.subheader("📈 Trend Sejarah")

    if data:
        plot_overview_metrics(data)

    # System information
    st.markdown("---")
    st.subheader("ℹ️ Maklumat Sistem")

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.info(
            """
        **Liputan Data:**
        - Data sejarah: 5 tahun
        - Horizon ramalan: 2 tahun
        - Kekerapan kemas kini: Bulanan
        """
        )

    with info_col2:
        if models:
            model_status = "✅ Model dimuatkan dan sedia"
            st.success(model_status)
        else:
            model_status = "❌ Model tidak tersedia"
            st.error(model_status)

    # Quick actions
    st.markdown("---")
    st.subheader("🚀 Tindakan Pantas")

    action_col1, action_col2, action_col3 = st.columns(3)

    with action_col1:
        if st.button("Jana Ramalan Baharu"):
            st.info("Navigasi ke halaman ramalan khusus untuk menjana prediksi")

    with action_col2:
        if st.button("Eksport Data Semasa"):
            if data:
                st.download_button(
                    label="Muat Turun Data Populasi",
                    data=data["population_data"].to_csv(index=False),
                    file_name=f"data_populasi_penjara_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )

    with action_col3:
        if st.button("Semak Status Model"):
            if models:
                st.success("Semua model beroperasi")
                for model_name in models.keys():
                    st.write(f"✅ {model_name.replace('_', ' ').title()}")
            else:
                st.warning("Sesetengah model mungkin perlu dilatih semula")


if __name__ == "__main__":
    main()
