import pandas as pd
import numpy as np
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path):
    """Loads the dataset from the specified file path."""
    return pd.read_csv(path)

def preprocess(df, is_training=True, preprocessor=None):
    """Applies standardized feature engineering and filtering to the dataset."""
    df = df.copy()
    #df = df[:10000]  # Limit to first 10000 rows for faster processing
    
    if 'LayoutType' in df.columns:
        df['RoomCount'] = df['LayoutType'].str.extract(r'(\d+)').astype(float).fillna(1)
    
    df['LivingRatio'] = df['LivingArea'] / (df['TotalArea'] + 1)
    df['FloorRatio'] = df['Floor'] / (df['FloorsTotal'] + 1)
    df['AreaPerRoom'] = df['TotalArea'] / (df['RoomCount'] + 1)
    df['IsTopFloor'] = (df['Floor'] == df['FloorsTotal']).astype(int)

    if is_training:
        df = df[df['TotalArea'].between(df['TotalArea'].quantile(0.01), df['TotalArea'].quantile(0.99))]
        df = df[df['TotalCost'].between(df['TotalCost'].quantile(0.01), df['TotalCost'].quantile(0.98))]

    cat_cols = ['District', 'Class', 'BuildingType', 'Finishing', 'PropertyType']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    features = [
        'RoomCount', 'TotalArea', 'Floor', 'FloorsTotal', 'CeilingHeight',
        'LivingArea', 'KitchenArea', 'AreaPerRoom', 'FloorRatio',
        'IsTopFloor', 'District', 'Class', 'BuildingType', 'Finishing'
    ]
    
    if preprocessor and 'features' in preprocessor:
        features = preprocessor['features']
    
    X = df[[f for f in features if f in df.columns]]
    y = df['TotalCost'] if 'TotalCost' in df.columns else None
    
    return X, y, preprocessor

def main(data_path):
    """Executes the training pipeline, evaluates metrics, and saves model artifacts."""
    print("Loading and Preprocessing...")
    df_raw = load_data(data_path)
    X, y, _ = preprocess(df_raw, is_training=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cat_idx = [i for i, col in enumerate(X.columns) if X[col].dtype == 'category']

    base_regressor = HistGradientBoostingRegressor(
        max_iter=1000,
        learning_rate=0.03,
        max_depth=8,
        l2_regularization=1.5,
        categorical_features=cat_idx,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=15
    )

    model = TransformedTargetRegressor(
        regressor=base_regressor,
        func=np.log1p,
        inverse_func=np.expm1
    )

    print("Training the Optimized Model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("-" * 30)
    print(f"Optimized R²: {r2:.4f}")
    print(f"MAE: {mae:,.0f} RUB")
    print("-" * 30)

    print("Calculating feature importances...")
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats= 2, random_state=42)
    
    # Save the model, features, and the calculated importances
    model_data = {
        'model': model,
        'preprocessor': {'features': list(X_train.columns)},
        'metrics': {'test_r2': r2, 'test_mae': mae},
        'importances': perm_importance.importances_mean.tolist()
    }
    
    joblib.dump(model_data, 'model.pkl')
    print("Model saved as 'model.pkl'")
    # Generate Evaluation Plots
    print("Generating evaluation plots...")
    plt.figure(figsize=(15, 6))

    # Plot 1: Actual vs. Predicted Prices
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.4, color='#667eea', edgecolor=None)
    # Draw the perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel("Actual Price (RUB)")
    plt.ylabel("Predicted Price (RUB)")
    plt.title("Model Fit: Actual vs Predicted Prices")
    plt.ticklabel_format(style='plain', axis='both')  # Prevents scientific notation (e.g., 1e7)
    plt.legend()

    # Plot 2: Top 10 Feature Importances
    plt.subplot(1, 2, 2)
    features_list = list(X_train.columns)
    importances_list = perm_importance.importances_mean
    
    # Create a dataframe for easy sorting and plotting
    feat_imp_df = pd.DataFrame({'Feature': features_list, 'Importance': importances_list})
    # Filter out negatives and sort
    feat_imp_df['Importance'] = feat_imp_df['Importance'].clip(lower=0) 
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(10)
    
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='magma')
    plt.title("Top 10 Drivers of Apartment Value")
    plt.xlabel("Permutation Importance Weight")
    plt.ylabel("")

    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    print("Plots saved successfully as 'model_evaluation.png'")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Please provide a path to the dataset.")
