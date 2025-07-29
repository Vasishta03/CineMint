import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

class TeluguBoxOfficePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.genre_encoder = None
        self.mpaa_encoder = LabelEncoder()
        self.feature_names = None
        
    def load_and_prepare_data(self, csv_path='telugu_movies_boxoffice.csv'):
        """Load and prepare the Telugu movies dataset"""
        # Load data
        df = pd.read_csv(csv_path)
        
        # Handle missing values
        df['budget'] = df['budget'].fillna(df['budget'].median())
        df['opening_theatres'] = df['opening_theatres'].fillna(df['opening_theatres'].median())
        df['opening_revenue'] = df['opening_revenue'].fillna(df['opening_revenue'].median())
        
        # Create additional features
        df['budget_per_theatre'] = df['budget'] / df['opening_theatres']
        df['opening_revenue_per_theatre'] = df['opening_revenue'] / df['opening_theatres']
        df['release_year'] = pd.to_datetime(df['release_days'], errors='coerce').dt.year
        df['release_month'] = pd.to_datetime(df['release_days'], errors='coerce').dt.month
        
        # Festival season feature (Jan, Apr, Oct, Nov - major Telugu festivals)
        df['festival_season'] = df['release_month'].apply(
            lambda x: 1 if x in [1, 4, 10, 11] else 0
        )
        
        # Genre features (multi-hot encoding)
        all_genres = set()
        for genres in df['genres'].dropna():
            all_genres.update(genres.split('|'))
        
        for genre in all_genres:
            df[f'genre_{genre}'] = df['genres'].apply(
                lambda x: 1 if isinstance(x, str) and genre in x else 0
            )
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        # Select numerical features
        numerical_features = [
            'budget', 'opening_theatres', 'opening_revenue', 'release_days',
            'budget_per_theatre', 'opening_revenue_per_theatre', 'release_year',
            'release_month', 'festival_season'
        ]
        
        # Add genre features
        genre_features = [col for col in df.columns if col.startswith('genre_')]
        
        # MPAA encoding
        df['mpaa_encoded'] = self.mpaa_encoder.fit_transform(df['MPAA'].fillna('UA'))
        
        # Combine all features
        feature_cols = numerical_features + genre_features + ['mpaa_encoded']
        X = df[feature_cols].fillna(0)
        
        self.feature_names = X.columns.tolist()
        return X
    
    def train_models(self, X, y):
        """Train multiple models and select the best one"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = -np.inf
        results = {}
        
        for name, model in models.items():
            if name in ['Linear Regression', 'Ridge', 'Lasso']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {
                'R²': r2,
                'RMSE': rmse,
                'MAE': mae,
                'model': model
            }
            
            if r2 > best_score:
                best_score = r2
                best_model = model
                self.model_type = name
        
        self.model = best_model
        return results
    
    def predict(self, movie_data):
        """Predict box office revenue for a new movie"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare features similar to training data
        features = np.zeros(len(self.feature_names))
        feature_dict = dict(zip(self.feature_names, features))
        
        # Fill in provided data
        for key, value in movie_data.items():
            if key in feature_dict:
                feature_dict[key] = value
        
        # Handle genres
        if 'genres' in movie_data:
            genres = movie_data['genres'].split('|')
            for genre in genres:
                genre_key = f'genre_{genre}'
                if genre_key in feature_dict:
                    feature_dict[genre_key] = 1
        
        # MPAA encoding
        if 'MPAA' in movie_data:
            try:
                feature_dict['mpaa_encoded'] = self.mpaa_encoder.transform([movie_data['MPAA']])[0]
            except:
                feature_dict['mpaa_encoded'] = self.mpaa_encoder.transform(['UA'])[0]
        
        X_pred = np.array(list(feature_dict.values())).reshape(1, -1)
        
        if self.model_type in ['Linear Regression', 'Ridge', 'Lasso']:
            X_pred = self.scaler.transform(X_pred)
        
        prediction = self.model.predict(X_pred)[0]
        return max(0, prediction)  # Ensure non-negative prediction
    
    def save_model(self, filepath='telugu_boxoffice_model.pkl'):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'mpaa_encoder': self.mpaa_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='telugu_boxoffice_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.mpaa_encoder = model_data['mpaa_encoder']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        print(f"Model loaded from {filepath}")

# Train the model
def train_telugu_predictor():
    predictor = TeluguBoxOfficePredictor()
    
    # Create the CSV file (save the dataset above as 'telugu_movies_boxoffice.csv')
    
    # Load and prepare data
    df = predictor.load_and_prepare_data()
    X = predictor.prepare_features(df)
    y = df['world_revenue']  # Predict worldwide revenue
    
    # Train models
    results = predictor.train_models(X, y)
    
    # Print results
    print("\nModel Performance Comparison:")
    print("-" * 60)
    for name, metrics in results.items():
        print(f"{name:15s} | R²: {metrics['R²']:.3f} | RMSE: {metrics['RMSE']:.2f} | MAE: {metrics['MAE']:.2f}")
    
    # Save the best model
    predictor.save_model()
    
    return predictor

if __name__ == "__main__":
    predictor = train_telugu_predictor()
