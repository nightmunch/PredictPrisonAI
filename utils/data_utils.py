import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import joblib

def load_or_generate_data():
    """
    Load existing data or generate new data if not available
    """
    data_files = {
        'population_data': 'data/population_data.csv',
        'staffing_data': 'data/staffing_data.csv',
        'resource_data': 'data/resource_data.csv',
        'prison_detail_data': 'data/prison_detail_data.csv'
    }
    
    data = {}
    
    # Check if data files exist
    all_files_exist = all(os.path.exists(file_path) for file_path in data_files.values())
    
    if all_files_exist:
        # Load existing data
        try:
            for key, file_path in data_files.items():
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                data[key] = df
            
            # Load prison structure if available
            import json
            if os.path.exists('data/malaysia_prisons.json'):
                with open('data/malaysia_prisons.json', 'r') as f:
                    data['malaysia_prisons'] = json.load(f)
            
            print("Data loaded successfully from files")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return generate_synthetic_data()
    else:
        # Generate new data
        print("Data files not found. Generating synthetic data...")
        return generate_synthetic_data()

def generate_synthetic_data():
    """
    Generate synthetic data for demonstration purposes
    """
    try:
        # Define Malaysian prison structure by state
        malaysia_prisons = {
            'Kedah': ['Pokok Sena Prison', 'Sungai Petani Prison', 'Alor Star Prison'],
            'Penang': ['Penang Prison', 'Seberang Prai Prison'],
            'Perak': ['Taiping Prison', 'Tapah Prison', 'Kamunting Detention Centre'],
            'Selangor': ['Sungai Buloh Prison', 'Kajang Prison', 'Kajang Women\'s Prison'],
            'Negeri Sembilan': ['Seremban Prison', 'Jelebu Prison'],
            'Melaka': ['Ayer Keroh Prison', 'Sungai Udang Prison', 'Banda Hilir Prison'],
            'Johor': ['Simpang Renggam Prison', 'Kluang Prison'],
            'Pahang': ['Bentong Prison', 'Penor Prison'],
            'Terengganu': ['Marang Prison'],
            'Kelantan': ['Pengkalan Chepa Prison'],
            'Sarawak': ['Puncak Borneo Prison', 'Sibu Prison', 'Miri Prison', 'Bintulu Prison', 'Sri Aman Prison', 'Limbang Prison'],
            'Sabah': ['Kota Kinabalu Prison', 'Kota Kinabalu Women\'s Prison', 'Tawau Prison', 'Sandakan Prison']
        }
        
        # Generate realistic Malaysian prison data
        data = {}
        
        # Generate population data based on Malaysian prison statistics
        # 5 years of data = 60 months
        dates = pd.date_range(start='2020-01-01', periods=60, freq='ME')
        
        # Population data - Based on Malaysian Prison Department statistics
        # Malaysia has approximately 50,000-55,000 prisoners (more realistic current figure)
        base_population = 52000  # More realistic baseline for current Malaysian prison population
        
        # Gradual increase due to population growth and crime trends over 5 years
        trend = np.linspace(0, 3000, 60)  # Conservative 5-year growth to ~55k
        
        # Seasonal variation (holidays, court schedules affect admissions)
        # More pronounced seasonal effects for Malaysian context
        seasonal = 1800 * np.sin(2 * np.pi * np.arange(60) / 12)
        
        # Additional seasonal factors for Malaysian context
        # Ramadan/Hari Raya effects (month 4-5, 9-10)
        ramadan_effect = np.zeros(60)
        for i in range(60):
            month = (i % 12) + 1
            if month in [4, 5, 9, 10]:  # Ramadan/festive seasons
                ramadan_effect[i] = -300  # Lower crime during religious periods
        
        # Chinese New Year effect (month 1-2)
        cny_effect = np.zeros(60)
        for i in range(60):
            month = (i % 12) + 1
            if month in [1, 2]:
                cny_effect[i] = -150  # Lower activity during CNY
        
        # Monsoon season effect (month 11-1) - higher indoor crimes
        monsoon_effect = np.zeros(60)
        for i in range(60):
            month = (i % 12) + 1
            if month in [11, 12, 1]:
                monsoon_effect[i] = 200  # Slightly higher crimes during monsoon
        
        # Random variation with lower volatility for government data
        noise = np.random.normal(0, 450, 60)
        
        total_prisoners = base_population + trend + seasonal + ramadan_effect + cny_effect + monsoon_effect + noise
        total_prisoners = np.maximum(total_prisoners, 48000)  # Realistic minimum
        
        # Realistic demographic breakdowns based on Malaysian statistics
        # Male ratio is higher in Malaysian prisons (typically 92-95%)
        male_ratio = np.random.normal(0.935, 0.008, 60)  # More stable ratio
        male_prisoners = (total_prisoners * male_ratio).astype(int)
        female_prisoners = total_prisoners.astype(int) - male_prisoners
        
        # Age groups - Malaysian prison demographics with more realistic distribution
        # Young (18-30), Middle (31-50), Old (50+)
        young_ratio = np.random.normal(0.48, 0.015, 60)  # Higher youth incarceration in Malaysia
        middle_ratio = np.random.normal(0.40, 0.015, 60)
        old_ratio = 1 - young_ratio - middle_ratio
        old_ratio = np.maximum(old_ratio, 0.08)  # Ensure minimum elderly population
        
        young_prisoners = (total_prisoners * young_ratio).astype(int)
        middle_prisoners = (total_prisoners * middle_ratio).astype(int)
        old_prisoners = total_prisoners.astype(int) - young_prisoners - middle_prisoners
        
        # Crime types - Malaysian crime pattern (drug crimes are dominant due to strict laws)
        drug_crimes_ratio = np.random.normal(0.58, 0.025, 60)  # Very high drug crimes in Malaysia
        violent_crimes_ratio = np.random.normal(0.16, 0.015, 60)  # Lower violent crime
        property_crimes_ratio = np.random.normal(0.14, 0.015, 60)  # Property crimes
        other_crimes_ratio = 1 - drug_crimes_ratio - violent_crimes_ratio - property_crimes_ratio
        other_crimes_ratio = np.maximum(other_crimes_ratio, 0.08)  # Ensure minimum other crimes
        
        drug_crimes = (total_prisoners * drug_crimes_ratio).astype(int)
        violent_crimes = (total_prisoners * violent_crimes_ratio).astype(int)
        property_crimes = (total_prisoners * property_crimes_ratio).astype(int)
        other_crimes = total_prisoners.astype(int) - drug_crimes - violent_crimes - property_crimes
        
        # Sentence lengths and flow - Malaysian judicial patterns
        # Malaysia has longer average sentences due to drug laws
        avg_sentence_months = np.random.normal(32, 6, 60)  # Longer average sentences for drug crimes
        avg_sentence_months = np.maximum(avg_sentence_months, 8)
        
        # Monthly flow rates based on Malaysian court processing
        # Lower turnover due to longer sentences and court backlogs
        monthly_releases = np.random.poisson(2200, 60)  # Lower release rate
        monthly_admissions = np.random.poisson(2280, 60)  # Slight net increase matching trend
        
        # Add COVID-19 impact for 2020-2021 period
        for i in range(60):
            date_current = dates[i]
            if date_current.year == 2020 and date_current.month >= 3:  # MCO period
                monthly_admissions[i] = int(monthly_admissions[i] * 0.7)  # Reduced court operations
                monthly_releases[i] = int(monthly_releases[i] * 0.8)  # Reduced releases due to health protocols
            elif date_current.year == 2021:  # Recovery period
                monthly_admissions[i] = int(monthly_admissions[i] * 0.85)  # Gradual recovery
                monthly_releases[i] = int(monthly_releases[i] * 0.9)
        
        # Realistic state distribution based on actual Malaysian prison capacity and crime statistics
        state_populations = {
            'Selangor': 0.28,    # Largest - Sungai Buloh complex, high urban crime
            'Sarawak': 0.18,     # Large state, drug trafficking routes, rural crime
            'Johor': 0.16,       # High crime rate, Singapore border, drug trafficking
            'Sabah': 0.12,       # Large state, border security issues, immigrant crime
            'Perak': 0.08,       # Kamunting Detention Centre, historic prisons
            'Kedah': 0.06,       # Northern border, drug trafficking routes
            'Pahang': 0.05,      # Large but sparse population, drug cultivation
            'Penang': 0.03,      # Urban crime, drug hub
            'Melaka': 0.02,      # Historic, tourist area, smaller population
            'Negeri Sembilan': 0.015, # Smaller industrial state
            'Kelantan': 0.005,   # Conservative state, lower crime, religious influence
            'Terengganu': 0.005  # Oil state, lower urban crime, religious influence
        }
        
        # Generate state-level and prison-level data
        state_data_list = []
        
        for date in dates:
            date_idx = list(dates).index(date)
            total_pop = int(total_prisoners[date_idx])
            
            for state, ratio in state_populations.items():
                state_total = int(total_pop * ratio)
                prisons_in_state = malaysia_prisons[state]
                
                # Distribute population among prisons in the state
                if len(prisons_in_state) == 1:
                    prison_pops = [state_total]
                else:
                    # Create realistic distribution (some prisons are larger)
                    if state == 'Selangor':
                        # Sungai Buloh is the largest
                        ratios = [0.6, 0.25, 0.15]  # Sungai Buloh, Kajang, Kajang Women's
                    elif state == 'Perak':
                        # Kamunting is supermax (smaller), Taiping is historic (larger)
                        ratios = [0.5, 0.35, 0.15]  # Taiping, Tapah, Kamunting
                    else:
                        # Even distribution with slight variation
                        base_ratio = 1.0 / len(prisons_in_state)
                        ratios = [base_ratio + np.random.normal(0, 0.05) for _ in prisons_in_state]
                        ratios = [max(0.1, r) for r in ratios]  # Minimum 10%
                        total_ratio = sum(ratios)
                        ratios = [r/total_ratio for r in ratios]  # Normalize
                    
                    prison_pops = [int(state_total * r) for r in ratios]
                    # Adjust to match state total
                    prison_pops[-1] += state_total - sum(prison_pops)
                
                # Add data for each prison
                for i, prison_name in enumerate(prisons_in_state):
                    prison_pop = max(50, prison_pops[i])  # Minimum 50 prisoners per prison
                    
                    # Calculate prison-level demographics
                    prison_male = int(prison_pop * male_ratio[date_idx])
                    prison_female = prison_pop - prison_male
                    
                    # Special case for women's prisons
                    if 'Women' in prison_name:
                        prison_male = 0
                        prison_female = prison_pop
                    
                    prison_young = int(prison_pop * young_ratio[date_idx])
                    prison_middle = int(prison_pop * middle_ratio[date_idx])
                    prison_old = prison_pop - prison_young - prison_middle
                    
                    prison_drug = int(prison_pop * drug_crimes_ratio[date_idx])
                    prison_violent = int(prison_pop * violent_crimes_ratio[date_idx])
                    prison_property = int(prison_pop * property_crimes_ratio[date_idx])
                    prison_other = prison_pop - prison_drug - prison_violent - prison_property
                    
                    state_data_list.append({
                        'date': date,
                        'state': state,
                        'prison_name': prison_name,
                        'prison_population': prison_pop,
                        'male_prisoners': prison_male,
                        'female_prisoners': prison_female,
                        'young_prisoners': prison_young,
                        'middle_prisoners': prison_middle,
                        'old_prisoners': prison_old,
                        'drug_crimes': prison_drug,
                        'violent_crimes': prison_violent,
                        'property_crimes': prison_property,
                        'other_crimes': prison_other
                    })
        
        # Create detailed prison data
        prison_detail_data = pd.DataFrame(state_data_list)
        
        # Original aggregate population data
        population_data = pd.DataFrame({
            'date': dates,
            'total_prisoners': total_prisoners.astype(int),
            'male_prisoners': male_prisoners,
            'female_prisoners': female_prisoners,
            'young_prisoners': young_prisoners,
            'middle_prisoners': middle_prisoners,
            'old_prisoners': old_prisoners,
            'drug_crimes': drug_crimes,
            'violent_crimes': violent_crimes,
            'property_crimes': property_crimes,
            'other_crimes': other_crimes,
            'avg_sentence_months': avg_sentence_months,
            'monthly_releases': monthly_releases,
            'monthly_admissions': monthly_admissions
        })
        
        # Staffing data - More realistic Malaysian prison staffing
        # Malaysian prisons typically understaffed
        base_ratio = 0.26  # Lower staff-to-prisoner ratio (understaffing issue)
        ratio_variation = np.random.normal(0, 0.015, 60)
        staff_ratio = base_ratio + ratio_variation
        staff_ratio = np.maximum(staff_ratio, 0.22)  # Minimum staffing level
        
        total_staff = (total_prisoners * staff_ratio).astype(int)
        
        # Staff distribution reflecting Malaysian prison structure
        security_staff_ratio = np.random.normal(0.68, 0.02, 60)  # Higher security needs
        admin_staff_ratio = np.random.normal(0.14, 0.015, 60)   # Efficient admin
        medical_staff_ratio = np.random.normal(0.07, 0.01, 60)  # Limited medical staff
        other_staff_ratio = 1 - security_staff_ratio - admin_staff_ratio - medical_staff_ratio
        other_staff_ratio = np.maximum(other_staff_ratio, 0.08)  # Ensure minimum other staff
        
        security_staff = (total_staff * security_staff_ratio).astype(int)
        admin_staff = (total_staff * admin_staff_ratio).astype(int)
        medical_staff = (total_staff * medical_staff_ratio).astype(int)
        other_staff = total_staff - security_staff - admin_staff - medical_staff
        
        # Higher overtime due to understaffing
        overtime_hours = np.random.normal(140, 25, 60)
        overtime_hours = np.maximum(overtime_hours, 80)
        
        # Malaysian context - higher sick leave in tropical climate
        sick_leave_rate = np.random.normal(0.09, 0.02, 60)
        vacation_rate = np.random.normal(0.11, 0.015, 60)
        
        available_staff = total_staff * (1 - sick_leave_rate - vacation_rate)
        available_staff = np.maximum(available_staff, total_staff * 0.75)  # Minimum 75% availability
        
        staffing_data = pd.DataFrame({
            'date': dates,
            'total_staff': total_staff,
            'security_staff': security_staff,
            'admin_staff': admin_staff,
            'medical_staff': medical_staff,
            'other_staff': other_staff,
            'overtime_hours': overtime_hours,
            'sick_leave_rate': sick_leave_rate,
            'vacation_rate': vacation_rate,
            'available_staff': available_staff.astype(int),
            'staff_prisoner_ratio': staff_ratio
        })
        
        # Resource data - Malaysian prison system capacity and costs
        total_capacity = 65000  # Realistic Malaysian prison system capacity (2024)
        capacity_utilization = (total_prisoners / total_capacity) * 100
        
        # More realistic Malaysian prison costs reflecting local economy
        base_daily_cost = 15  # RM 15 per prisoner per day (more realistic for Malaysian context)
        
        # Add inflation impact over 5 years (average 2.5% annually)
        inflation_factor = np.zeros(60)
        for i in range(60):
            years_elapsed = i / 12.0
            inflation_factor[i] = (1 + 0.025) ** years_elapsed
        
        daily_cost_variation = np.random.normal(0, 0.8, 60)  # Reduced variation
        daily_cost_per_prisoner = (base_daily_cost * inflation_factor) + daily_cost_variation
        daily_cost_per_prisoner = np.maximum(daily_cost_per_prisoner, 12)  # Minimum RM 12/day
        
        # Cost breakdown reflecting Malaysian operations and climate
        monthly_food_cost = total_prisoners * daily_cost_per_prisoner * 30 * 0.45  # Food costs (largest component)
        monthly_medical_cost = total_prisoners * daily_cost_per_prisoner * 30 * 0.12  # Medical costs
        monthly_utility_cost = total_prisoners * daily_cost_per_prisoner * 30 * 0.28  # Utilities (tropical climate, AC)
        monthly_other_cost = total_prisoners * daily_cost_per_prisoner * 30 * 0.15  # Admin, security, maintenance
        
        total_monthly_cost = monthly_food_cost + monthly_medical_cost + monthly_utility_cost + monthly_other_cost
        
        # Maintenance costs with seasonal variations (monsoon damage)
        base_maintenance = 480000
        seasonal_maintenance = np.zeros(60)
        for i in range(60):
            month = (i % 12) + 1
            if month in [11, 12, 1, 2]:  # Monsoon season higher maintenance
                seasonal_maintenance[i] = 150000
            elif month in [3, 4]:  # Post-monsoon repairs
                seasonal_maintenance[i] = 100000
            else:
                seasonal_maintenance[i] = 0
        
        maintenance_cost = np.random.normal(base_maintenance, 80000, 60) + seasonal_maintenance
        maintenance_cost = np.maximum(maintenance_cost, 300000)
        
        # Malaysian context adjustments
        food_waste_rate = np.random.normal(0.10, 0.025, 60)  # Better food management
        energy_efficiency = np.random.normal(0.72, 0.04, 60)  # Tropical climate challenges
        
        # Medical supplies reflecting tropical diseases and healthcare needs
        medical_supplies_cost = total_prisoners * np.random.normal(9.5, 1.2, 60)  # Higher medical needs
        
        # Security equipment costs with technology upgrades over time
        base_security_cost = 120000
        tech_upgrade_factor = np.linspace(1.0, 1.4, 60)  # Gradual tech improvements
        security_equipment_cost = np.random.normal(base_security_cost, 25000, 60) * tech_upgrade_factor
        
        resource_data = pd.DataFrame({
            'date': dates,
            'capacity_utilization': capacity_utilization,
            'total_capacity': total_capacity,
            'daily_cost_per_prisoner': daily_cost_per_prisoner,
            'monthly_food_cost': monthly_food_cost,
            'monthly_medical_cost': monthly_medical_cost,
            'monthly_utility_cost': monthly_utility_cost,
            'monthly_other_cost': monthly_other_cost,
            'total_monthly_cost': total_monthly_cost,
            'maintenance_cost': maintenance_cost,
            'food_waste_rate': food_waste_rate,
            'energy_efficiency': energy_efficiency,
            'medical_supplies_cost': medical_supplies_cost,
            'security_equipment_cost': security_equipment_cost
        })
        
        data = {
            'population_data': population_data,
            'staffing_data': staffing_data,
            'resource_data': resource_data,
            'prison_detail_data': prison_detail_data,
            'malaysia_prisons': malaysia_prisons
        }
        
        # Save the generated data
        os.makedirs('data', exist_ok=True)
        for key, df in data.items():
            if isinstance(df, pd.DataFrame):
                df.to_csv(f'data/{key}.csv', index=False)
            elif key == 'malaysia_prisons':
                # Save prison structure as a separate reference file
                import json
                with open('data/malaysia_prisons.json', 'w') as f:
                    json.dump(df, f, indent=2)
        
        print("Synthetic data generated and saved successfully")
        
        # Train models automatically after generating data
        try:
            from models.model_trainer import train_all_models
            print("Training models with generated data...")
            models = train_all_models(data)
            if models:
                print("Models trained and saved successfully")
            else:
                print("Warning: Model training failed")
        except Exception as e:
            print(f"Warning: Could not train models automatically: {e}")
        
        return data
        
    except Exception as e:
        print(f"Error generating synthetic data: {e}")
        return None

def load_models():
    """
    Load trained models if available
    """
    import warnings
    
    # Suppress scikit-learn version warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        
        models_dir = 'models'
        models = {}
        
        try:
            if os.path.exists(models_dir):
                for model_type in ['population', 'staffing', 'resource']:
                    model_path = os.path.join(models_dir, f'{model_type}_model.pkl')
                    if os.path.exists(model_path):
                        models[model_type] = joblib.load(model_path)
                
                # Load metrics if available
                metrics_path = os.path.join(models_dir, 'model_metrics.pkl')
                if os.path.exists(metrics_path):
                    models['metrics'] = joblib.load(metrics_path)
                    
                # Load feature importance if available
                importance_path = os.path.join(models_dir, 'feature_importance.pkl')
                if os.path.exists(importance_path):
                    models['feature_importance'] = joblib.load(importance_path)
            
            return models if models else None
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return None

def get_latest_data_point(data, metric):
    """
    Get the latest data point for a specific metric
    """
    try:
        if metric in data.columns:
            return data[metric].iloc[-1]
        else:
            return None
    except Exception as e:
        print(f"Error getting latest data point: {e}")
        return None

def calculate_growth_rate(data, metric, periods=12):
    """
    Calculate growth rate for a metric over specified periods
    """
    try:
        if metric in data.columns and len(data) >= periods:
            current_value = data[metric].iloc[-1]
            past_value = data[metric].iloc[-periods]
            growth_rate = ((current_value - past_value) / past_value) * 100
            return growth_rate
        else:
            return None
    except Exception as e:
        print(f"Error calculating growth rate: {e}")
        return None

def prepare_forecast_data(data, target_column, forecast_periods=24):
    """
    Prepare data for forecasting
    """
    try:
        df = data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Create future dates
        last_date = df['date'].max()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=forecast_periods,
            freq='M'
        )
        
        return df, future_dates
        
    except Exception as e:
        print(f"Error preparing forecast data: {e}")
        return None, None

def export_data_to_csv(data, filename):
    """
    Export data to CSV file
    """
    try:
        data.to_csv(filename, index=False)
        return True
    except Exception as e:
        print(f"Error exporting data: {e}")
        return False

def get_summary_statistics(data, columns):
    """
    Get summary statistics for specified columns
    """
    try:
        summary = data[columns].describe()
        return summary
    except Exception as e:
        print(f"Error calculating summary statistics: {e}")
        return None
