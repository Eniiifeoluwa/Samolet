from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import sys

from train import preprocess
from scraper import scrape_apartment, fill_defaults

app = Flask(__name__)
model_data = None

def load_model():
    """Initializes the global model data from the serialized file."""
    global model_data
    try:
        model_data = joblib.load('model.pkl')
        if isinstance(model_data, dict) and 'model' in model_data:
            return True
        return False
    except Exception as e:
        print(f"Load Error: {e}")
        return False

def explain_prediction(model_data):
    """Maps permutation importances to their respective names and normalizes them."""
    importances = model_data['importances']
    feature_names = model_data['preprocessor']['features']
    
    explanation = []
    # Floor negative importances to 0, then calculate total for percentage
    total_imp = sum(max(0, imp) for imp in importances)
    
    for name, imp in zip(feature_names, importances):
        # Calculate percentage weight (0 to 1 scale)
        weight = max(0, imp) / total_imp if total_imp > 0 else 0
        explanation.append({
            'feature': name,
            'importance': float(weight)
        })
    
    explanation.sort(key=lambda x: x['importance'], reverse=True)
    return explanation[:10]

@app.route('/')
def index():
    """Renders the main web interface."""
    return render_template('index.html')

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    """Processes manual apartment feature inputs to return a price estimate."""
    try:
        data = request.json
        
        apartment_df = pd.DataFrame([{
            'RoomCount': float(data['rooms']),
            'TotalArea': float(data['area']),
            'Floor': int(data['floor']),
            'FloorsTotal': int(data['floors_total']),
            'CeilingHeight': float(data.get('ceiling_height', 2.7)),
            'LivingArea': float(data.get('living_area', 0)),
            'KitchenArea': float(data.get('kitchen_area', 0)),
            'BalconyArea': float(data.get('balcony_area', 0)),
            'District': data.get('district', 'Unknown'),
            'Class': data.get('class', 'Комфорт'),
            'BuildingType': data.get('building_type', 'Монолит'),
            'Finishing': data.get('finishing', 'Чистовая'),
            'LayoutType': f"{int(data['rooms'])} ккв"
        }])
        
        X_processed, _, _ = preprocess(apartment_df, is_training=False, preprocessor=model_data['preprocessor'])
        
        prediction = model_data['model'].predict(X_processed)[0]
        
        # FIXED: Now passing only the single model_data dictionary
        explanation = explain_prediction(model_data)
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'price_per_meter': float(prediction / float(data['area'])),
            'explanation': explanation
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/predict_url', methods=['POST'])
def predict_url():
    """Orchestrates web scraping, default filling, and prediction for a given URL."""
    try:
        data = request.json
        url = data.get('url')
        
        if not url:
            return jsonify({'success': False, 'error': 'URL is required'}), 400
            
        scraped_raw = scrape_apartment(url)
        apartment_features = fill_defaults(scraped_raw)
        
        apartment_df = pd.DataFrame([apartment_features])
        X_processed, _, _ = preprocess(apartment_df, is_training=False, preprocessor=model_data['preprocessor'])
        
        prediction = model_data['model'].predict(X_processed)[0]
        
        # FIXED: Now passing only the single model_data dictionary
        explanation = explain_prediction(model_data)
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'price_per_meter': float(prediction / float(apartment_features['TotalArea'])),
            'explanation': explanation,
            'scraped_data': scraped_raw
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    if not load_model():
        print("ERROR: Model not found. Run 'python train.py' first.")
        sys.exit(1)
    
    app.run(debug=True, port=5000)