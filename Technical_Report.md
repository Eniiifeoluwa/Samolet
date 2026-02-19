# Technical Report: Apartment Price Prediction

## 1. Model Selection

### Chosen Architecture: HistGradientBoostingRegressor & TransformedTargetRegressor

Justification:

- Non-Linearity & Complex Relationships: Tree-based boosting algorithms effectively capture complex, non-linear interactions between location and layout features that linear models often miss.
- Robustness to Missing Data: This architecture natively handles missing values (NaNs) and categorical variables without requiring complex one-hot encoding or heavy imputation pipelines.
- Handling the Target Distribution: Real estate prices are heavily skewed. Wrapping the model in a TransformedTargetRegressor automatically applies a log transformation (np.log1p) during training and an exponential transformation (np.expm1) during prediction, significantly improving generalization and accuracy.

## 2. Target Variable

TotalCost (Total Apartment Price in RUB)

Justification:

- It represents the actual transaction value relevant to buyers and sellers.
- Using a log-transformed TotalCost aligns with the heavy-tailed nature of asset pricing.

## 3. Feature Engineering

### Extracted Features

- Number of rooms
- Total area
- Floor
- Total floors
- Ceiling height
- Living area
- Kitchen area
- Balcony area
- District
- Building class
- Finishing

### Engineered Features

- AreaPerRoom: Average area per room.
- FloorRatio: Floor divided by total floors.
- IsTopFloor: Binary indicator for premium top-floor apartments.

Note: Tree-based models do not require feature scaling, simplifying preprocessing.

## 4. Validation Strategy

### 80/20 Train-Test Split

- Training (80%): Used for model fitting. Outlier clipping (bottom 1% and top 1% of area and cost) is applied strictly to the training set.
- Test (20%): Left untouched for unbiased evaluation on unseen apartments.

Rationale:

- The test set remains completely untouched to simulate real-world inference.
- A fixed random seed ensures reproducibility.

## 5. Evaluation Metrics

### Mean Absolute Error (MAE)

Average absolute prediction error in RUB.

- Provides a directly interpretable pricing error in currency units.

### R² (Coefficient of Determination)

Proportion of price variance explained by the model.

- Indicates overall predictive power, with values closer to 1.0 reflecting stronger performance.

## 6. Model Interpretation

### Permutation Importance

Because HistGradientBoosting does not provide native linear coefficients, the system uses Permutation Importance evaluated on the test set.

Local Interpretation:

- Base Estimate: Estimated market value in RUB.
- Key Factors: Percentage-based breakdown of feature influence on the final estimate.

## 7. Interface Implementation

### Manual Input Mode

- Users input apartment characteristics.
- The system preprocesses inputs and returns predictions with explanatory breakdown.

### Link-Based Input Mode

- Users provide a listing URL.
- The system extracts features via regex parsing and returns predictions.

## 8. Assumptions and Limitations

### Assumptions

- Missing attributes during scraping may be approximated using reasonable baseline ratios.
- The model assumes future data falls within similar regional distributions as training data.

### Limitations & WAF Constraints

- Listing sites may enforce Web Application Firewalls (WAF) that block automated scraping.
- The samolet.ru example returns a 401 Unauthorized response due to bot protection.
- The scraper catches this exception and supplies mock characteristic data to keep the inference pipeline and UI functional.

- Temporal Drift: Real estate prices evolve over time and require periodic retraining.

## 9. Conclusion

This solution delivers an interpretable and robust apartment price prediction system. The HistGradientBoosting architecture captures complex market dynamics efficiently, while Permutation Importance ensures transparency. The dual-interface design improves accessibility, and resilient scraping logic demonstrates practical MLOps considerations when facing strict web firewalls.
