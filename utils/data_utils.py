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
        'resource_data': 'data/resource_data.csv'
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
        # Generate data using the same logic as in the notebook
        data = {}
        
        # Generate population data
        dates = pd.date_range(start='2019-01-01', periods=84, freq='M')
        
        # Population data
        base_population = 75000
        trend = np.linspace(0, 5000, 84)
        seasonal = 2000 * np.sin(2 * np.pi * np.arange(84) / 12)
        noise = np.random.normal(0, 1000, 84)
        
        total_prisoners = base_population + trend + seasonal + noise
        total_prisoners = np.maximum(total_prisoners, 60000)
        
        # Demographic breakdowns
        male_ratio = np.random.normal(0.85, 0.02, 84)
        male_prisoners = (total_prisoners * male_ratio).astype(int)
        female_prisoners = total_prisoners.astype(int) - male_prisoners
        
        # Age groups
        young_ratio = np.random.normal(0.25, 0.03, 84)
        middle_ratio = np.random.normal(0.55, 0.03, 84)
        old_ratio = 1 - young_ratio - middle_ratio
        
        young_prisoners = (total_prisoners * young_ratio).astype(int)
        middle_prisoners = (total_prisoners * middle_ratio).astype(int)
        old_prisoners = total_prisoners.astype(int) - young_prisoners - middle_prisoners
        
        # Crime types
        drug_crimes_ratio = np.random.normal(0.35, 0.05, 84)
        violent_crimes_ratio = np.random.normal(0.25, 0.03, 84)
        property_crimes_ratio = np.random.normal(0.20, 0.03, 84)
        other_crimes_ratio = 1 - drug_crimes_ratio - violent_crimes_ratio - property_crimes_ratio
        
        drug_crimes = (total_prisoners * drug_crimes_ratio).astype(int)
        violent_crimes = (total_prisoners * violent_crimes_ratio).astype(int)
        property_crimes = (total_prisoners * property_crimes_ratio).astype(int)
        other_crimes = total_prisoners.astype(int) - drug_crimes - violent_crimes - property_crimes
        
        # Sentence lengths and flow
        avg_sentence_months = np.random.normal(36, 6, 84)
        avg_sentence_months = np.maximum(avg_sentence_months, 12)
        
        monthly_releases = np.random.poisson(2800, 84)
        monthly_admissions = np.random.poisson(3000, 84)
        
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
        
        # Staffing data
        base_ratio = 0.28
        ratio_variation = np.random.normal(0, 0.02, 84)
        staff_ratio = base_ratio + ratio_variation
        
        total_staff = (total_prisoners * staff_ratio).astype(int)
        
        security_staff_ratio = np.random.normal(0.65, 0.03, 84)
        admin_staff_ratio = np.random.normal(0.15, 0.02, 84)
        medical_staff_ratio = np.random.normal(0.08, 0.01, 84)
        other_staff_ratio = 1 - security_staff_ratio - admin_staff_ratio - medical_staff_ratio
        
        security_staff = (total_staff * security_staff_ratio).astype(int)
        admin_staff = (total_staff * admin_staff_ratio).astype(int)
        medical_staff = (total_staff * medical_staff_ratio).astype(int)
        other_staff = total_staff - security_staff - admin_staff - medical_staff
        
        overtime_hours = np.random.normal(120, 20, 84)
        overtime_hours = np.maximum(overtime_hours, 60)
        
        sick_leave_rate = np.random.normal(0.08, 0.02, 84)
        vacation_rate = np.random.normal(0.12, 0.02, 84)
        
        available_staff = total_staff * (1 - sick_leave_rate - vacation_rate)
        
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
        
        # Resource data
        total_capacity = 95000
        capacity_utilization = (total_prisoners / total_capacity) * 100
        
        base_daily_cost = 45
        daily_cost_variation = np.random.normal(0, 3, 84)
        daily_cost_per_prisoner = base_daily_cost + daily_cost_variation
        
        monthly_food_cost = total_prisoners * daily_cost_per_prisoner * 30 * 0.4
        monthly_medical_cost = total_prisoners * daily_cost_per_prisoner * 30 * 0.15
        monthly_utility_cost = total_prisoners * daily_cost_per_prisoner * 30 * 0.20
        monthly_other_cost = total_prisoners * daily_cost_per_prisoner * 30 * 0.25
        
        total_monthly_cost = monthly_food_cost + monthly_medical_cost + monthly_utility_cost + monthly_other_cost
        
        maintenance_cost = np.random.normal(500000, 100000, 84)
        maintenance_cost = np.maximum(maintenance_cost, 200000)
        
        food_waste_rate = np.random.normal(0.12, 0.03, 84)
        energy_efficiency = np.random.normal(0.75, 0.05, 84)
        
        medical_supplies_cost = total_prisoners * np.random.normal(8, 1, 84)
        security_equipment_cost = np.random.normal(150000, 30000, 84)
        
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
            'resource_data': resource_data
        }
        
        # Save the generated data
        os.makedirs('data', exist_ok=True)
        for key, df in data.items():
            df.to_csv(f'data/{key}.csv', index=False)
        
        print("Synthetic data generated and saved successfully")
        return data
        
    except Exception as e:
        print(f"Error generating synthetic data: {e}")
        return None

def load_models():
    """
    Load trained models if available
    """
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
