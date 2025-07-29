from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# ==== Model Loader Class ====

class ModelPredictor:
    def __init__(self):
        self.load_model()

    def load_model(self):
        try:
            model_data = joblib.load('telugu_boxoffice_model.pkl')
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.mpaa_encoder = model_data['mpaa_encoder']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            self.model = None

    def predict(self, movie_data):
        features = np.zeros(len(self.feature_names))
        feature_dict = dict(zip(self.feature_names, features))

        # Fill fields from incoming movie_data
        mapping = ['budget', 'opening_theatres', 'opening_revenue', 'release_days',
                   'release_year', 'release_month']
        for key in mapping:
            if key in movie_data and key in feature_dict:
                feature_dict[key] = movie_data[key]

        # Derived features
        if 'budget' in movie_data and 'opening_theatres' in movie_data:
            feature_dict['budget_per_theatre'] = float(movie_data['budget']) / max(1, float(movie_data['opening_theatres']))
        if 'opening_revenue' in movie_data and 'opening_theatres' in movie_data:
            feature_dict['opening_revenue_per_theatre'] = float(movie_data['opening_revenue']) / max(1, float(movie_data['opening_theatres']))
        if 'release_month' in movie_data:
            feature_dict['festival_season'] = 1 if int(movie_data['release_month']) in [1, 4, 10, 11] else 0

        # Genre features
        if 'genres' in movie_data and isinstance(movie_data['genres'], str):
            genres = movie_data['genres'].split('|')
            for genre in genres:
                genre_key = f'genre_{genre}'
                if genre_key in feature_dict:
                    feature_dict[genre_key] = 1

        # MPAA encoding
        if 'MPAA' in movie_data:
            try:
                feature_dict['mpaa_encoded'] = self.mpaa_encoder.transform([movie_data['MPAA']])[0]
            except Exception:
                feature_dict['mpaa_encoded'] = self.mpaa_encoder.transform(['UA'])[0]

        X_pred = np.array(list(feature_dict.values())).reshape(1, -1)
        if self.model_type in ['Linear Regression', 'Ridge', 'Lasso']:
            X_pred = self.scaler.transform(X_pred)
        pred = float(self.model.predict(X_pred)[0])  # Fix: convert to float

        return max(0, round(pred, 2))  # Ensure non-negative, round to 2 decimals

predictor = ModelPredictor()

# ==== API Endpoints ====

@app.route('/predict', methods=['POST'])
def predict_boxoffice():
    try:
        data = request.json
        required = ['budget', 'opening_theatres', 'genres', 'MPAA']
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        prediction = predictor.predict(data)
        response = {
            'predicted_worldwide_revenue': prediction,
            'predicted_domestic_revenue': round(prediction * 0.7, 2),
            'predicted_overseas_revenue': round(prediction * 0.3, 2),
            'movie_details': {
                'title': data.get('title', 'Unknown'),
                'budget': data.get('budget'),
                'opening_theatres': data.get('opening_theatres'),
                'genres': data.get('genres'),
                'MPAA': data.get('MPAA')
            },
            'roi_estimate': round((prediction / float(data.get('budget', 1)) - 1) * 100, 2) if float(data.get('budget', 0)) > 0 else 0
        }
        # convert all nums to native Python types for safety
        for k in ['predicted_worldwide_revenue', 'predicted_domestic_revenue', 'predicted_overseas_revenue', 'roi_estimate']:
            response[k] = float(response[k])
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': predictor.model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
