# Real Estate Price Prediction Engine

A complete, end-to-end data-driven solution that estimates apartment prices using layout and location-related attributes. This project includes a machine learning pipeline, a web scraper, and an interactive Flask web interface.

## Files
- `train.py` - Training pipeline, feature engineering, and model serialization.
- `app.py` - Flask web application routing and inference logic.
- `scraper.py` - URL scraping module with anti-bot handling.
- `examples.py` - CLI script for running sample predictions.
- `templates/index.html` - Interactive frontend UI.
- `requirements.txt` - Python dependencies.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model

Note: Ensure your dataset is in the same directory or provide the correct path.

```bash
python train.py data.csv
```

Outputs Generated:
- `model.pkl`
- `model_evaluation.png`

### 3. Run the Web Interface
```bash
python app.py
```

Open: http://localhost:5000

### 4. Run CLI Examples
```bash
python examples.py
```

## Interface Features

### Manual Input Mode
Users manually enter apartment characteristics and receive a price estimate with explanation. The model dynamically computes derived features such as AreaPerRoom and returns the valuation.

### Link-Based Input Mode
Users provide a link to an apartment listing page. The system extracts apartment characteristics automatically and returns a price estimate.

Example:
`https://samolet.ru/project/oktyabrskaya-98/flats/308985/`

The scraper gracefully handles 401 Unauthorized responses by catching the exception and supplying mock data so the ML inference pipeline and UI remain testable.

## Model Details

Algorithm: HistGradientBoostingRegressor wrapped in TransformedTargetRegressor.

Why this architecture:
- Log-normal target transformation improves R².
- Native handling of missing values.
- Captures nonlinear interactions between location and layout features.

Features Used:

Base:
- TotalArea
- Floor
- FloorsTotal
- RoomCount
- CeilingHeight
- LivingArea
- KitchenArea

Engineered:
- AreaPerRoom
- FloorRatio
- IsTopFloor

Categorical:
- District
- Class
- BuildingType
- Finishing

## Validation Strategy
- 80 percent training and 20 percent testing split
- Outlier clipping applied strictly to training data
- Metrics evaluated: R² and MAE

## Model Interpretation
Permutation Importance is computed on the test set.
The UI presents a percentage-based feature influence breakdown for each prediction.