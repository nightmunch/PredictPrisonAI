import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PrisonForecastModels:
    """
    A comprehensive class for training and managing prison forecasting models
    """
    
    def __init__(self):
        self.models = {}
        self.model_metrics = {}
        self.feature_importance = {}
        
    def prepare_features(self, data, target_column, lookback_window=4):
        """
        Prepare features for time series forecasting - enhanced for 5-year data
        """
        df = data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Create lag features with longer lookback for 5-year data
        for lag in range(1, lookback_window + 1):
            df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        
        # Create rolling statistics with multiple windows for better seasonality capture
        for window in [3, 6, 12]:  # Added 12-month window for annual patterns
            df[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(window=window).mean()
            df[f'{target_column}_rolling_std_{window}'] = df[target_column].rolling(window=window).std()
        
        # Enhanced seasonal features for Malaysian context
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['days_from_start'] = (df['date'] - df['date'].min()).dt.days
        
        # Malaysian specific seasonal indicators
        df['is_ramadan_period'] = df['month'].isin([4, 5, 9, 10]).astype(int)
        df['is_festive_season'] = df['month'].isin([1, 2, 12]).astype(int)
        df['is_monsoon_season'] = df['month'].isin([11, 12, 1]).astype(int)
        
        # Year-over-year change
        df[f'{target_column}_yoy_change'] = df[target_column].pct_change(periods=12)
        
        # Trend indicator
        df['trend'] = np.arange(len(df))
        
        # Fill NaN values strategically
        # Forward fill for rolling statistics
        for col in df.columns:
            if 'rolling' in col or 'yoy_change' in col:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Remove rows with remaining NaN values (mainly from lag features)
        df = df.dropna()
        
        return df
    
    def train_population_model(self, population_data):
        """
        Train models for prison population forecasting
        """
        print("Training population forecasting models...")
        
        # Prepare features
        df = self.prepare_features(population_data, 'total_prisoners')
        
        # Define feature columns
        feature_cols = [col for col in df.columns if col not in ['date', 'total_prisoners']]
        
        X = df[feature_cols]
        y = df['total_prisoners']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train multiple models with enhanced parameters for 5-year data
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=150, 
                max_depth=12, 
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=120, 
                learning_rate=0.08,
                max_depth=8,
                min_samples_split=4,
                random_state=42
            ),
            'linear_regression': LinearRegression()
        }
        
        best_model = None
        best_score = float('inf')
        best_model_name = 'linear_regression'  # Default fallback
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                self.model_metrics[f'population_{name}'] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': np.sqrt(mse)
                }
                
                if mse < best_score:
                    best_score = mse
                    best_model = model
                    best_model_name = name
            except Exception as e:
                print(f"Error training {name} model: {e}")
                continue
        
        # Ensure we have a model
        if best_model is None:
            print("Warning: No models trained successfully, using linear regression as fallback")
            best_model = LinearRegression()
            best_model.fit(X_train, y_train)
            best_model_name = 'linear_regression'
        
        self.models['population'] = best_model
        
        # Store feature importance for tree-based models
        if best_model is not None and hasattr(best_model, 'feature_importances_'):
            self.feature_importance['population'] = dict(zip(feature_cols, best_model.feature_importances_))
        
        print(f"Best population model: {best_model_name} (RMSE: {np.sqrt(best_score):.2f})")
        
        return best_model
    
    def train_staffing_model(self, staffing_data, population_data):
        """
        Train models for staffing forecasting
        """
        print("Training staffing forecasting models...")
        
        # Merge with population data for additional features
        merged_data = pd.merge(staffing_data, population_data[['date', 'total_prisoners']], on='date')
        
        # Prepare features
        df = self.prepare_features(merged_data, 'total_staff')
        
        # Define feature columns
        feature_cols = [col for col in df.columns if col not in ['date', 'total_staff']]
        
        X = df[feature_cols]
        y = df['total_staff']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        
        best_model = None
        best_score = float('inf')
        best_model_name = 'linear_regression'
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                self.model_metrics[f'staffing_{name}'] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': np.sqrt(mse)
                }
                
                if mse < best_score:
                    best_score = mse
                    best_model = model
                    best_model_name = name
            except Exception as e:
                print(f"Error training staffing {name} model: {e}")
                continue
        
        if best_model is None:
            print("Warning: No staffing models trained successfully, using linear regression as fallback")
            best_model = LinearRegression()
            best_model.fit(X_train, y_train)
            best_model_name = 'linear_regression'
        
        self.models['staffing'] = best_model
        
        # Store feature importance
        if best_model is not None and hasattr(best_model, 'feature_importances_'):
            self.feature_importance['staffing'] = dict(zip(feature_cols, best_model.feature_importances_))
        
        print(f"Best staffing model: {best_model_name} (RMSE: {np.sqrt(best_score):.2f})")
        
        return best_model
    
    def train_resource_model(self, resource_data, population_data):
        """
        Train models for resource forecasting
        """
        print("Training resource forecasting models...")
        
        # Merge with population data
        merged_data = pd.merge(resource_data, population_data[['date', 'total_prisoners']], on='date')
        
        # Prepare features for total monthly cost
        df = self.prepare_features(merged_data, 'total_monthly_cost')
        
        # Define feature columns
        feature_cols = [col for col in df.columns if col not in ['date', 'total_monthly_cost']]
        
        X = df[feature_cols]
        y = df['total_monthly_cost']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        
        best_model = None
        best_score = float('inf')
        best_model_name = 'linear_regression'
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                self.model_metrics[f'resource_{name}'] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': np.sqrt(mse)
                }
                
                if mse < best_score:
                    best_score = mse
                    best_model = model
                    best_model_name = name
            except Exception as e:
                print(f"Error training resource {name} model: {e}")
                continue
        
        if best_model is None:
            print("Warning: No resource models trained successfully, using linear regression as fallback")
            best_model = LinearRegression()
            best_model.fit(X_train, y_train)
            best_model_name = 'linear_regression'
        
        self.models['resource'] = best_model
        
        # Store feature importance
        if best_model is not None and hasattr(best_model, 'feature_importances_'):
            self.feature_importance['resource'] = dict(zip(feature_cols, best_model.feature_importances_))
        
        print(f"Best resource model: {best_model_name} (RMSE: {np.sqrt(best_score):.2f})")
        
        return best_model
    
    def save_models(self, models_dir='models'):
        """
        Save all trained models
        """
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            joblib.dump(model, os.path.join(models_dir, f'{model_name}_model.pkl'))
        
        # Save metrics and feature importance
        joblib.dump(self.model_metrics, os.path.join(models_dir, 'model_metrics.pkl'))
        joblib.dump(self.feature_importance, os.path.join(models_dir, 'feature_importance.pkl'))
        
        print(f"Models saved to {models_dir} directory")
    
    def predict_future(self, model_name, historical_data, periods=24):
        """
        Generate future predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # This is a simplified prediction method
        # In practice, you would implement more sophisticated forecasting logic
        last_values = historical_data.tail(1)
        predictions = []
        
        for i in range(periods):
            # For demonstration, we'll add some trend and noise
            # In practice, you'd use proper feature engineering
            trend = np.random.normal(1.01, 0.02)  # Small positive trend
            noise = np.random.normal(0, 0.05)
            
            pred = 0  # Initialize pred
            if model_name == 'population':
                pred = last_values['total_prisoners'].iloc[0] * trend + noise * 1000
            elif model_name == 'staffing':
                pred = last_values['total_staff'].iloc[0] * trend + noise * 100
            elif model_name == 'resource':
                pred = last_values['total_monthly_cost'].iloc[0] * trend + noise * 100000
            else:
                # Fallback for unknown model types
                pred = last_values.iloc[0, 1] * trend + noise * 100
            
            predictions.append(max(0, pred))  # Ensure non-negative predictions
        
        return predictions

def train_all_models(data_dict):
    """
    Train all forecasting models
    """
    trainer = PrisonForecastModels()
    
    # Extract data
    population_data = data_dict['population_data']
    staffing_data = data_dict['staffing_data']
    resource_data = data_dict['resource_data']
    
    # Train models
    trainer.train_population_model(population_data)
    trainer.train_staffing_model(staffing_data, population_data)
    trainer.train_resource_model(resource_data, population_data)
    
    # Save models
    trainer.save_models()
    
    return trainer.models

def load_trained_models(models_dir='models'):
    """
    Load previously trained models
    """
    models = {}
    
    try:
        for model_type in ['population', 'staffing', 'resource']:
            model_path = os.path.join(models_dir, f'{model_type}_model.pkl')
            if os.path.exists(model_path):
                models[model_type] = joblib.load(model_path)
        
        # Load metrics and feature importance
        metrics_path = os.path.join(models_dir, 'model_metrics.pkl')
        importance_path = os.path.join(models_dir, 'feature_importance.pkl')
        
        if os.path.exists(metrics_path):
            models['metrics'] = joblib.load(metrics_path)
        
        if os.path.exists(importance_path):
            models['feature_importance'] = joblib.load(importance_path)
            
    except Exception as e:
        print(f"Error loading models: {e}")
        return None
    
    return models if models else None

if __name__ == "__main__":
    # Example usage
    print("Model trainer module loaded successfully")
