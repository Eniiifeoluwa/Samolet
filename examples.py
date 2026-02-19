import joblib
import pandas as pd
from train import preprocess

def load_model():
    """Loads the serialized machine learning model and metadata."""
    return joblib.load('model.pkl')

def predict_example(model_data, apartment):
    """Processes a single apartment example and returns its price and feature importances."""
    df = pd.DataFrame([apartment])
    
    X_processed, _, _ = preprocess(df, is_training=False, preprocessor=model_data['preprocessor'])
    prediction = model_data['model'].predict(X_processed)[0]
    
    importances = model_data.get('importances', [])
    features = model_data['preprocessor']['features']
    
    contributions = []
    if len(importances) > 0:
        total_imp = sum(max(0, imp) for imp in importances)
        for name, imp in zip(features, importances):
            weight = max(0, imp) / total_imp if total_imp > 0 else 0
            contributions.append((name, weight))
        
        contributions.sort(key=lambda x: x[1], reverse=True)
    
    return prediction, contributions

def main():
    """Executes sample predictions to validate the model's behavior."""
    print("="*70)
    print("🏠 REAL ESTATE VALUATION EXAMPLES")
    print("="*70)
    
    try:
        model_data = load_model()
    except FileNotFoundError:
        print("Error: 'model.pkl' not found. Please train the model first.")
        return

    print(f"Model: HistGradientBoostingRegressor (Optimized)")
    if 'metrics' in model_data:
        print(f"Test R²: {model_data['metrics']['test_r2']:.4f}")
        print(f"Test MAE: {model_data['metrics']['test_mae']:,.0f} RUB\n")
    
    examples = [
        {
            'name': '3-room Comfort Apartment',
            'data': {
                'RoomCount': 3,
                'TotalArea': 75.0,
                'Floor': 10,
                'FloorsTotal': 20,
                'CeilingHeight': 2.8,
                'LivingArea': 45.0,
                'KitchenArea': 12.0,
                'BalconyArea': 4.5,
                'District': 'МО Ленинский г.о.',
                'Class': 'Комфорт',
                'BuildingType': 'Монолит',
                'PropertyType': 'Многокв. дом',
                'Finishing': 'Чистовая',
                'LayoutType': '3 ккв'
            }
        },
        {
            'name': '1-room Economy Apartment',
            'data': {
                'RoomCount': 1,
                'TotalArea': 38.0,
                'Floor': 1,
                'FloorsTotal': 12,
                'CeilingHeight': 2.7,
                'LivingArea': 18.0,
                'KitchenArea': 9.0,
                'BalconyArea': 0,
                'District': 'МО Ленинский г.о.',
                'Class': 'Эконом',
                'BuildingType': 'Панель',
                'PropertyType': 'Многокв. дом',
                'Finishing': 'Без отделки',
                'LayoutType': '1 ккв'
            }
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\nEXAMPLE {i}: {example['name']}")
        print('-'*40)
        
        data = example['data']
        prediction, importances = predict_example(model_data, data)
        
        print(f"📊 Predicted Price: {prediction:,.0f} RUB")
        print(f"📐 Price per m²:    {prediction/data['TotalArea']:,.0f} RUB/m²")
        
        print(f"\n🔍 Top 5 Drivers of this Valuation:")
        for feat, imp in importances[:5]:
            print(f"  {feat:20s}: {imp*100:>6.1f}% Influence")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()