import streamlit as st

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Malaysia Prison Analytics - Powered By Credence AI & Analytics",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import warnings
import base64

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utility modules
from utils.data_utils import load_or_generate_data, load_models
from utils.visualization import plot_overview_metrics

def get_logo_base64():
    """Convert logo image to base64 for embedding in HTML"""
    try:
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "Penjara-logo.jpg")
        with open(logo_path, "rb") as f:
            img_bytes = f.read()
        return base64.b64encode(img_bytes).decode()
    except Exception as e:
        print(f"Error loading logo: {e}")
        return ""

# Language dictionaries
LANG = {
    "en": {
        "page_title": "Malaysia Prison Analytics Powered By Credence AI & Analytics",
        "title": "Malaysia Prison Analytics\nPowered By Credence AI & Analytics",
        "subtitle": "Advanced Predictive Analytics for Malaysian Prison Administration",
        "navigation": "Navigation",
        "select_page": "Select a page:",
        "pages": [
            "Dashboard Overview",
            "Population Forecast",
            # "Staffing Forecast",  # Feature disabled
            "Resource Forecast",
            "Model Performance",
        ],
        "dashboard_header": "📊 Executive Dashboard",
        "state_selector": "Select State:",
        "prison_selector": "Select Prison:",
        "all_states": "All States",
        "all_prisons": "All Prisons",
        "all_prisons_in_state": "All Prisons in State",
        "prison_system_by_state": "🗺️ Prison System by State",
        "current_population": "Current Prison Population",
        "daily_cost": "Daily Cost per Prisoner",
        "capacity_utilization": "Capacity Utilization",
        "avg_sentence": "Avg Sentence Length",
        "months": "months",
        "male_prisoners": "Male Prisoners",
        "female_prisoners": "Female Prisoners",
        "drug_crimes": "Drug Crimes",
        "violent_crimes": "Violent Crimes",
        "prison_population_by_state": "🗺️ Prison Population by State",
        "historical_trends": "📈 Historical Trends",
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
        
        # Population Forecast Page
        "population_forecast_header": "👥 Prison Population Forecast",
        "forecast_parameters": "Forecast Parameters",
        "forecast_level": "Forecast Level",
        "overall_malaysia": "Overall Malaysia",
        "by_state": "By State", 
        "by_prison": "By Prison",
        "select_state": "Select State:",
        "select_prison": "Select Prison:",
        "forecast_period": "Forecast Period (months):",
        "scenario_analysis": "Scenario Analysis",
        "enhanced_rehabilitation": "Enhanced Rehabilitation",
        "economic_impact": "Economic Impact",
        "base_case": "Base Case",
        "scenario_factor": "Scenario Factor:",
        "generate_forecast_btn": "Generate Forecast",
        "current_metrics": "Current Metrics",
        "forecast_results": "Forecast Results",
        "monthly_forecast": "Monthly Forecast",
        "demographic_analysis": "Demographic Analysis",
        "trend_analysis": "Trend Analysis",
        "seasonal_patterns": "Seasonal Patterns",
        
        # Additional Population Forecast translations
        "forecast_statistics": "Forecast Statistics",
        "projected_population_end": "Projected Population (End)",
        "peak_population": "Peak Population",
        "average_forecast_period": "Average (Forecast Period)",
        "scenario_impact": "Scenario Impact",
        "population_forecast_scenario_comparison": "Population Forecast - Scenario Comparison",
        "malaysian_prison_population_forecast": "Malaysian Prison Population Forecast",
        "population_forecast_malaysia_overall": "Population Forecast - Malaysia Overall",
        "historical_data": "Historical Data",
        "forecast": "Forecast",
        "confidence_interval": "Confidence Interval (±5%)",
        "number_of_prisoners": "Number of Prisoners",
        
        # Resource Forecast Page
        "resource_forecast_header": "💰 Resource Forecast",
        "cost_forecast_planning": "Cost Forecast and Budget Planning",
        "cost_projections": "Cost Projections",
        "cost_breakdown_forecast": "Cost Breakdown Forecast",
        "capacity_planning_infrastructure": "Capacity Planning and Infrastructure",
        "efficiency_analysis_optimization": "Efficiency Analysis and Optimization",
        "resource_optimization_planning": "Resource Optimization and Planning",
        "optimization_opportunities": "**Optimization Opportunities:**",
        "cost_summary": "**Cost Summary:**",
        "monthly_cost_per_prisoner": "• Monthly cost per prisoner: MYR {:.0f}",
        "annual_cost_projection": "• Annual cost projection: MYR {:.1f}M",
        
        # Resource Forecast Tabs
        "cost_forecast_tab": "📈 Cost Forecast",
        "capacity_planning_tab": "🏗️ Capacity Planning", 
        "efficiency_analysis_tab": "⚡ Efficiency Analysis",
        "resource_optimization_tab": "📊 Resource Optimization",
        
        # Resource Forecast Parameters
        "resource_forecast_level": "Resource Forecast Level",
        "malaysia_overall": "Malaysia Overall",
        "cost_inflation_rate": "Cost Inflation Rate (%/year)",
        "efficiency_improvement_target": "Efficiency Improvement Target (%)",
        
        # Resource Forecast Info Messages
        "showing_data_for": "📊 Showing data for {:.1%} of national population",
        "capacity_planning_analysis": "📋 Capacity planning analysis based on filtered data",
        "energy_efficiency_analysis": "⚡ Energy and operational efficiency analysis", 
        "resource_optimization_recommendations": "📊 Resource optimization recommendations",
        
        # Model Performance Page
        "model_performance_header": "🤖 Model Performance",
        "model_overview": "Model Overview",
        "performance_metrics": "Performance Metrics",
        
        # Chart Titles and Labels
        "date": "Date",
        "number_of_prisoners": "Number of Prisoners",
        "current_staff_distribution": "Current Staff Distribution",
        "monthly_cost_distribution": "Monthly Cost Distribution (MYR)",
        "correlation_matrix": "Correlation Matrix",
        "variables": "Variables",
        "seasonal_pattern": "Seasonal Pattern",
        "month": "Month",
        "model_performance_comparison": "Model Performance Comparison",
        "forecast": "Forecast",
        
        # Common Terms
        "prisoners": "prisoners",
        "staff": "staff",
        "myr": "MYR",
        "months_unit": "months",
        "years": "years",
        "total": "Total",
        "average": "Average",
        "maximum": "Maximum",
        "minimum": "Minimum",
    },
    "ms": {
        "page_title": "Analitik Penjara Malaysia Dikuasakan Oleh Credence AI & Analytics",
        "title": "Analitik Penjara Malaysia\nDikuasakan Oleh Credence AI & Analytics",
        "subtitle": "Analitik Prediktif Termaju untuk Pentadbiran Penjara Malaysia",
        "navigation": "Navigasi",
        "select_page": "Pilih halaman:",
        "pages": [
            "Papan Pemuka Eksekutif",
            "Ramalan Populasi",
            # "Ramalan Kakitangan",  # Ciri dilumpuhkan
            "Ramalan Sumber",
            "Prestasi Model",
        ],
        "dashboard_header": "📊 Papan Pemuka Eksekutif",
        "state_selector": "Pilih Negeri:",
        "prison_selector": "Pilih Penjara:",
        "all_states": "Semua Negeri",
        "all_prisons": "Semua Penjara",
        "all_prisons_in_state": "Semua Penjara dalam Negeri",
        "prison_system_by_state": "🗺️ Sistem Penjara mengikut Negeri",
        "current_population": "Populasi Penjara Semasa",
        "daily_cost": "Kos Harian per Banduan",
        "capacity_utilization": "Penggunaan Kapasiti",
        "avg_sentence": "Purata Tempoh Hukuman",
        "months": "bulan",
        "male_prisoners": "Banduan Lelaki",
        "female_prisoners": "Banduan Perempuan",
        "drug_crimes": "Jenayah Dadah",
        "violent_crimes": "Jenayah Kekerasan",
        "prison_population_by_state": "🗺️ Populasi Penjara mengikut Negeri",
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
        "malay": "Bahasa Malaysia",
        "no_historical_data": "Tiada data sejarah untuk negeri/penjara yang dipilih.",
        "no_model_status": "Tiada maklumat status model tersedia.",
        
        # Population Forecast Page
        "population_forecast_header": "👥 Ramalan Populasi Penjara",
        "forecast_parameters": "Parameter Ramalan",
        "forecast_level": "Tahap Ramalan",
        "overall_malaysia": "Keseluruhan Malaysia",
        "by_state": "Mengikut Negeri", 
        "by_prison": "Mengikut Penjara",
        "select_state": "Pilih Negeri:",
        "select_prison": "Pilih Penjara:",
        "forecast_period": "Tempoh Ramalan (bulan):",
        "scenario_analysis": "Analisis Senario",
        "enhanced_rehabilitation": "Pemulihan Dipertingkat",
        "economic_impact": "Kesan Ekonomi",
        "base_case": "Kes Asas",
        "scenario_factor": "Faktor Senario:",
        "generate_forecast_btn": "Jana Ramalan",
        "current_metrics": "Metrik Semasa",
        "forecast_results": "Keputusan Ramalan",
        "monthly_forecast": "Ramalan Bulanan",
        "demographic_analysis": "Analisis Demografi",
        "trend_analysis": "Analisis Trend",
        "seasonal_patterns": "Corak Musiman",
        
        # Additional Population Forecast translations
        "forecast_statistics": "Statistik Ramalan",
        "projected_population_end": "Populasi Unjuran (Akhir)",
        "peak_population": "Populasi Puncak",
        "average_forecast_period": "Purata (Tempoh Ramalan)",
        "scenario_impact": "Kesan Senario",
        "population_forecast_scenario_comparison": "Ramalan Populasi - Perbandingan Senario",
        "malaysian_prison_population_forecast": "Ramalan Populasi Penjara Malaysia",
        "population_forecast_malaysia_overall": "Ramalan Populasi - Keseluruhan Malaysia",
        "historical_data": "Data Sejarah",
        "forecast": "Ramalan",
        "confidence_interval": "Selang Keyakinan (±5%)",
        "number_of_prisoners": "Bilangan Banduan",
        
        # Resource Forecast Page
        "resource_forecast_header": "💰 Ramalan Sumber",
        "cost_forecast_planning": "Ramalan Kos dan Perancangan Belanjawan",
        "cost_projections": "Unjuran Kos",
        "cost_breakdown_forecast": "Ramalan Pecahan Kos",
        "capacity_planning_infrastructure": "Perancangan Kapasiti dan Infrastruktur",
        "efficiency_analysis_optimization": "Analisis Kecekapan dan Pengoptimuman",
        "resource_optimization_planning": "Pengoptimuman dan Perancangan Sumber",
        "optimization_opportunities": "**Peluang Pengoptimuman:**",
        "cost_summary": "**Ringkasan Kos:**",
        "monthly_cost_per_prisoner": "• Kos bulanan per banduan: MYR {:.0f}",
        "annual_cost_projection": "• Unjuran kos tahunan: MYR {:.1f}J",
        
        # Resource Forecast Tabs
        "cost_forecast_tab": "📈 Ramalan Kos",
        "capacity_planning_tab": "🏗️ Perancangan Kapasiti", 
        "efficiency_analysis_tab": "⚡ Analisis Kecekapan",
        "resource_optimization_tab": "📊 Pengoptimuman Sumber",
        
        # Resource Forecast Parameters
        "resource_forecast_level": "Tahap Ramalan Sumber",
        "malaysia_overall": "Keseluruhan Malaysia",
        "cost_inflation_rate": "Kadar Inflasi Kos (%/tahun)",
        "efficiency_improvement_target": "Sasaran Peningkatan Kecekapan (%)",
        
        # Resource Forecast Info Messages
        "showing_data_for": "📊 Menunjukkan data untuk {:.1%} daripada populasi kebangsaan",
        "capacity_planning_analysis": "📋 Analisis perancangan kapasiti berdasarkan data yang ditapis",
        "energy_efficiency_analysis": "⚡ Analisis kecekapan tenaga dan operasi", 
        "resource_optimization_recommendations": "📊 Cadangan pengoptimuman sumber",
        
        # Model Performance Page
        "model_performance_header": "🤖 Prestasi Model",
        "model_overview": "Gambaran Model",
        "performance_metrics": "Metrik Prestasi",
        
        # Chart Titles and Labels
        "date": "Tarikh",
        "number_of_prisoners": "Bilangan Banduan",
        "current_staff_distribution": "Taburan Kakitangan Semasa",
        "monthly_cost_distribution": "Taburan Kos Bulanan (MYR)",
        "correlation_matrix": "Matriks Korelasi",
        "variables": "Pembolehubah",
        "seasonal_pattern": "Corak Musiman",
        "month": "Bulan",
        "model_performance_comparison": "Perbandingan Prestasi Model",
        "forecast": "Ramalan",
        
        # Common Terms
        "prisoners": "banduan",
        "staff": "kakitangan",
        "myr": "MYR",
        "months_unit": "bulan",
        "years": "tahun",
        "total": "Jumlah",
        "average": "Purata",
        "maximum": "Maksimum",
        "minimum": "Minimum",
    },
}


# Add custom CSS for mobile-friendly layout
st.markdown(
    """
    <style>
    .block-container { padding: 0.5rem 0.5rem; }
    .stMetric { font-size: 1.1rem; }
    @media (max-width: 600px) {
        .stMetric { font-size: 0.95rem; }
        .element-container { padding-left: 0.2rem !important; padding-right: 0.2rem !important; }
    }
    </style>
""",
    unsafe_allow_html=True,
)


def main():
    # Custom CSS for professional styling
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stTitle {
        color: #1e3a8a;
        font-weight: 700;
        text-align: center;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
        margin: 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    .professional-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    .professional-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
        background: white;
        border-radius: 50%;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Language selection
    lang = st.sidebar.selectbox(
        LANG["en"]["language"] + " / " + LANG["ms"]["language"],
        options=["en", "ms"],
        format_func=lambda x: LANG[x]["english"] if x == "en" else LANG[x]["malay"],
    )
    l = LANG[lang]
    
    # Professional header with official Malaysian Prison logo
    st.markdown(f'''
    <div class="professional-header">
        <div style="display: flex; align-items: center; justify-content: center; gap: 2rem; width: 100%;">
            <div style="flex-shrink: 0;">
                <img src="data:image/jpeg;base64,{get_logo_base64()}" style="width: 80px; height: 80px; border-radius: 10px; border: 2px solid white;">
            </div>
            <div style="text-align: center; flex-grow: 1;">
                <h1 style="margin: 0; font-size: 2.5rem; color: white;">{l["title"]}</h1>
                <p style="font-size: 1.1rem; opacity: 0.9; margin-top: 0.5rem; color: white;">{l["subtitle"]}</p>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

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
            
        # Add model status information to data
        if models:
            expected_models = ['population', 'staffing', 'resource']
            available_models = [model for model in expected_models if model in models]
            
            model_status = {
                "all_models_operational": len(available_models) == len(expected_models),
                "available_models": available_models,
                "total_models": len(expected_models),
                "available_count": len(available_models)
            }
            
            if len(available_models) < len(expected_models):
                missing_models = [model for model in expected_models if model not in available_models]
                model_status["retraining_info"] = f"Missing models: {', '.join(missing_models)}. Please retrain these models."
                
            data["model_status"] = model_status
        else:
            data["model_status"] = {
                "all_models_operational": False,
                "available_models": [],
                "total_models": 3,
                "available_count": 0,
                "retraining_info": "No models found. Please train the models first."
            }
            
    except Exception as e:
        st.error(l["error_loading"] + str(e))
        st.info(l["ensure_data"])
        return

    # Route to different pages
    if page == l["pages"][0]:
        show_dashboard_overview(data, models, l)
    elif page == l["pages"][1]:
        from modules.population_forecast import show_population_forecast

        show_population_forecast(data, models, l)
    # elif page == l['pages'][2]:  # Staffing/Staffing Forecast (disabled)
    #     from modules.staffing_forecast import show_staffing_forecast
    #     show_staffing_forecast(data, models, l)
    elif page == l["pages"][2]:
        from modules.resource_forecast import show_resource_forecast

        show_resource_forecast(data, models, l)
    elif page == l["pages"][3]:
        from modules.model_performance import show_model_performance

        show_model_performance(data, models, l)


def show_dashboard_overview(data, models, l):
    st.markdown(f"## {l['dashboard_header']}")
    st.markdown("---")
    
    # Professional KPI cards
    if "population_data" in data and "resource_data" in data:
        latest_pop = data["population_data"].iloc[-1]
        latest_res = data["resource_data"].iloc[-1]
        
        # Top row metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">{l["current_population"]}</p>
                <p class="metric-value">{latest_pop["total_prisoners"]:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">{l["daily_cost"]}</p>
                <p class="metric-value">RM {latest_res["daily_cost_per_prisoner"]:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">{l["capacity_utilization"]}</p>
                <p class="metric-value">{latest_res["capacity_utilization"]:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">{l["avg_sentence"]}</p>
                <p class="metric-value">{latest_pop["avg_sentence_months"]:.1f} {l["months"]}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Second row - Demographics
        col1, col2, col3, col4 = st.columns(4)
        
        if "male_prisoners" in latest_pop and "female_prisoners" in latest_pop:
            total = latest_pop["male_prisoners"] + latest_pop["female_prisoners"]
            male_pct = 100 * latest_pop["male_prisoners"] / total if total > 0 else 0
            female_pct = 100 * latest_pop["female_prisoners"] / total if total > 0 else 0
            
            with col1:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);">
                    <p class="metric-label">{l["male_prisoners"]}</p>
                    <p class="metric-value">{male_pct:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #ec4899 0%, #be185d 100%);">
                    <p class="metric-label">{l["female_prisoners"]}</p>
                    <p class="metric-value">{female_pct:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        if "drug_crimes" in latest_pop and "violent_crimes" in latest_pop:
            drug_pct = 100 * latest_pop["drug_crimes"] / latest_pop["total_prisoners"]
            violent_pct = 100 * latest_pop["violent_crimes"] / latest_pop["total_prisoners"]
            
            with col3:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);">
                    <p class="metric-label">{l["drug_crimes"]}</p>
                    <p class="metric-value">{drug_pct:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col4:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);">
                    <p class="metric-label">{l["violent_crimes"]}</p>
                    <p class="metric-value">{violent_pct:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # Enhanced state/prison selector
    if "prison_detail_data" in data and "malaysia_prisons" in data:
        st.markdown(f"### {l['prison_system_by_state']}")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_state = st.selectbox(
                l["state_selector"],
                [l["all_states"]] + list(data["malaysia_prisons"].keys()),
                key="state_selector"
            )
        
        with col2:
            if selected_state != l["all_states"]:
                prisons_in_state = data["malaysia_prisons"][selected_state]
                prison_names = [prison if isinstance(prison, str) else prison.get("name", str(prison)) 
                              for prison in prisons_in_state]
                selected_prison = st.selectbox(
                    l["prison_selector"], 
                    [l["all_prisons_in_state"]] + prison_names,
                    key="prison_selector"
                )
            else:
                selected_prison = l["all_prisons"]
                st.selectbox(l["prison_selector"], [l["all_prisons"]], disabled=True)

        # Display specific state/prison data
        if selected_state != l["all_states"]:
            latest_detail = data["prison_detail_data"][
                data["prison_detail_data"]["date"] == data["prison_detail_data"]["date"].max()
            ]
            
            if selected_prison != l["all_prisons_in_state"]:
                # Individual prison data
                prison_data = latest_detail[
                    (latest_detail["state"] == selected_state) & 
                    (latest_detail["prison_name"] == selected_prison)
                ]
                if not prison_data.empty:
                    prison_info = prison_data.iloc[0]
                    st.info(f"**{selected_prison}** - Total Population: {prison_info['prison_population']:,} prisoners")
                    
                    # Prison-specific metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Male", f"{prison_info['male_prisoners']:,}")
                    col2.metric("Female", f"{prison_info['female_prisoners']:,}")
                    col3.metric("Drug Crimes", f"{prison_info['drug_crimes']:,}")
                    col4.metric("Violent Crimes", f"{prison_info['violent_crimes']:,}")
            else:
                # State-level data
                state_data = latest_detail[latest_detail["state"] == selected_state]
                if not state_data.empty:
                    total_state_pop = state_data["prison_population"].sum()
                    prison_count = len(data["malaysia_prisons"][selected_state])
                    
                    st.info(f"**{selected_state} State** - Total Population: {total_state_pop:,} prisoners across {prison_count} prisons")
                    
                    # List prisons in state
                    st.write("**Prisons in this state:**")
                    for _, prison_row in state_data.iterrows():
                        st.write(f"• {prison_row['prison_name']}: {prison_row['prison_population']:,} prisoners")

    st.markdown("---")

    # Professional overview charts
    st.markdown(f"### {l['historical_trends']}")
    if data:
        plot_overview_metrics(data, chart_key="main_dashboard_overview", lang=l)

    # Model status and system information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 AI Model Status")
        if models and len(models) > 0:
            st.success("✅ All predictive models are operational")
            if 'metrics' in models:
                st.write("**Model Performance:**")
                for model_name, metrics in models['metrics'].items():
                    if 'r2' in metrics:
                        st.write(f"• {model_name.title()}: R² = {metrics['r2']:.3f}")
        else:
            st.error("❌ Models not available - please retrain")
    
    with col2:
        st.markdown(f"### {l['system_info']}")
        st.markdown(l["data_coverage"])
        
        if models:
            st.markdown("✅ " + l["models_loaded"])
        else:
            st.markdown("❌ " + l["models_not_available"])

    # Quick actions section
    st.markdown("---")
    st.markdown(f"### {l['quick_actions']}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 " + l["generate_forecast"], use_container_width=True):
            st.info("Navigate to Population Forecast or Resource Forecast to generate predictions.")
    
    with col2:
        if st.button("📊 " + l["export_data"], use_container_width=True):
            if "population_data" in data:
                csv_data = data["population_data"].to_csv(index=False)
                st.download_button(
                    label="📥 " + l["download_population_data"],
                    data=csv_data,
                    file_name="malaysia_prison_population_data.csv",
                    mime="text/csv"
                )
    
    with col3:
        if st.button("🔍 " + l["model_status_check"], use_container_width=True):
            if models:
                st.success(l["all_models_operational"])
            else:
                st.warning(l["some_models_need_retraining"])

    # State comparison chart
    if "prison_detail_data" in data:
        st.markdown("---")
        st.markdown(f"### {l['prison_population_by_state']}")
        
        latest_detail = data["prison_detail_data"][
            data["prison_detail_data"]["date"] == data["prison_detail_data"]["date"].max()
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
            labels={"x": "Prison Population", "y": "State"},
            color=state_totals.values,
            color_continuous_scale="Blues"
        )
        fig.update_layout(
            height=500,
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font=dict(size=16, color='#1e3a8a')
        )
        st.plotly_chart(fig, use_container_width=True, key="welcome_overview_chart")

    # System information (mobile-friendly: stack vertically)
    st.markdown("---")
    st.subheader(l["system_info"])
    st.info(l["data_coverage"])
    if models:
        st.success(l["models_loaded"])
    else:
        st.error(l["models_not_available"])

    # Quick actions (mobile-friendly: stack vertically)
    st.markdown("---")
    st.subheader(l["quick_actions"])
    if st.button(l["generate_forecast"]):
        st.info(
            "Navigate to specific forecast pages to generate predictions"
            if l["generate_forecast"].startswith("Generate")
            else "Navigasi ke halaman ramalan khusus untuk menjana prediksi"
        )
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
                    detail_df = detail_df[detail_df["prison_name"] == selected_prison]
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
