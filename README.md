# TIME-SERIES-PROJECT
 
A collection of time series forecasting experiments — from walk-forward validated tree models to two-stage/weighted hybrids combining XGBoost with Prophet, LSTM, and GRU. Each notebook tackles a different dataset, and where a naive baseline was computed, the model is benchmarked against it.
 
## Repository Structure
 
| Notebook | Dataset | Approach |
|---|---|---|
| `AEP HYBRID MODEL XGBOOST + XGBOOST.ipynb` | AEP hourly energy consumption (MW) | XGBoost + a second XGBoost trained on stage-1 residuals |
| `AIR PASSENGERS WALK FORWARD VALIDATION.ipynb` | Monthly airline passenger counts | XGBoost, walk-forward (retrain-per-step) validation |
| `BALITIMORE WATER USAGE WITH WALK FORWARD VALIDATION.ipynb` | Baltimore yearly water usage | XGBoost, walk-forward validation |
| `CHAMPAGNE MONTHLY SALES WALK FORWARD VALIDATION.ipynb` | Monthly champagne sales | XGBoost, walk-forward validation |
| `DEMAND FORECASTING PROPHET + XGBOOST.ipynb` | Retail store demand forecast | Prophet, then XGBoost trained on Prophet's residuals |
| `FAVORITA STORE SALES XGBOOST + LSTM WEIGHTED PREDICTION HYBRID MODEL.ipynb` | Favorita store sales (Kaggle) | XGBoost + LSTM, combined as a weighted average |
| `PJMW MW HOURLY PREDICTION LSTM.ipynb` | PJME hourly power consumption (MW) | LSTM |
| `SALES PREDICTION HYBRID MODEL XGBOOST + GRU.ipynb` | Superstore sales | XGBoost + a GRU trained on stage-1 residuals |
| `data/` | Datasets used by the notebooks | — |
 
## Libraries Used
 
| Notebook | Libraries |
|---|---|
| AEP Hybrid (XGBoost + XGBoost) | `pandas`, `matplotlib`, `xgboost` (`XGBRegressor`), `sklearn.metrics` |
| Air Passengers (walk-forward) | `pandas`, `numpy`, `xgboost`, `sklearn.metrics`, `matplotlib` |
| Baltimore Water Usage (walk-forward) | `pandas`, `xgboost`, `sklearn.metrics`, `matplotlib` |
| Champagne Monthly Sales (walk-forward) | `pandas`, `xgboost`, `sklearn.metrics`, `math.sqrt`, `matplotlib` |
| Demand Forecasting (Prophet + XGBoost) | `pandas`, `prophet` (`Prophet`), `xgboost`, `sklearn.metrics`, `matplotlib` |
| Favorita Store Sales (XGBoost + LSTM) | `pandas`, `numpy`, `holidays`, `xgboost`, `tensorflow.keras` (`Sequential`, `Dense`, `LSTM`, `Dropout`, `L2`, `SGD`), `sklearn.metrics`, `sklearn.model_selection.TimeSeriesSplit`, `statsmodels` (`plot_acf`, `plot_pacf`) |
| PJMW Hourly Prediction (LSTM) | `pandas`, `numpy`, `tensorflow.keras` (`Sequential`, `Dense`, `LSTM`), `sklearn.metrics` |
| Sales Prediction (XGBoost + GRU) | `pandas`, `numpy`, `xgboost`, `tensorflow.keras` (`Sequential`, `Dense`, `GRU`, `RMSprop`), `sklearn.metrics`, `sklearn.model_selection.TimeSeriesSplit` |
 
## Naive Baseline vs. Model Predictions (MAE unless noted)
 
Only notebooks that actually computed a naive baseline are included here. Numbers are pulled straight from each notebook's printed output.
 
| Notebook | Naive Baseline | Model Prediction |
|---|---|---|
| **Baltimore Water Usage** (walk-forward) | Persistence: 20.89 | Walk-forward MAE: 20.32 |
| **Champagne Monthly Sales** (walk-forward) | Persistence: 1737.37 | Walk-Forward MASE: 24.376721409116286 |
| **Demand Forecasting** (Prophet + XGBoost) | Persistence: 116.32 | Prophet only — Train: 86.29, Test: 86.07 <br> Final (+ XGBoost on residuals) — Train: 7.31, Test: 7.59 |
| **Favorita Store Sales** (XGBoost + LSTM) | Persistence (7-day rolling mean): 257.35 | XGBoost — Train: 185.33, Val: 237.87, Test: 227.13 <br> LSTM — Train: 62.57, Val: 258.25, Test: 235.32 <br> Weighted hybrid (0.6·XGB + 0.4·LSTM) — Train: 221.37, Val: 239.42, Test: 225.24 |
 
**Takeaways:**
- Baltimore Water Usage's XGBoost model barely edges out the persistence baseline (20.32 vs. 20.89) — the naive forecast is nearly as good here.
- Demand Forecasting's residual-correction step is the standout: Prophet alone (86 MAE) already beats persistence (116 MAE), and adding XGBoost on the residuals drops it to ~7.5 MAE.
- Favorita's weighted hybrid outperforms the persistence baseline (257.35) but is close to (and on the test set, not much better than) the plain XGBoost/LSTM models — the weighting isn't adding a lot here.
## Other Notebooks (no naive baseline computed yet)
 
These notebooks don't yet compare against a naive forecast, so they're described here by goal and approach instead.
 
**AEP Hybrid (XGBoost + XGBoost)**
Dataset: AEP hourly power consumption (MW), outlier-filtered to the 11,000–20,000 MW range.
Goal: forecast hourly power consumption. A first XGBoost model predicts consumption from lag (1h, 2h), rolling mean (7h, 24h), and calendar features; a second XGBoost model is trained on the first model's residuals and used to correct its predictions.
Result: Train MAE 534.32 → 321.88 after residual correction; Test MAE 569.23 → 324.78 after residual correction.
 
**Air Passengers (walk-forward)**
Dataset: the classic monthly international airline passengers series (1949–1960).
Goal: forecast monthly passenger counts using lag (1, 6, 12 months), rolling mean/std, and calendar features, with an XGBoost model retrained at every step of a walk-forward validation loop (the model is refit after each new test point is revealed).
Result: walk-forward MAE 123.48.
 
**PJMW Hourly Prediction (LSTM)**
Dataset: PJME hourly power consumption (MW), filtered to values ≥ 20,000 MW.
Goal: forecast the next hour's power consumption from a 30-step window of lag, rolling mean/std, and calendar features, using a stacked LSTM + Dense network.
Result: Train MAE 204.74, Val MAE 370.76, Test MAE 605.26 — the growing gap from train to test suggests some overfitting or distribution shift over time worth investigating.
 
**Sales Prediction (XGBoost + GRU)**
Dataset: Superstore retail sales data (outlier-filtered, categorical features one-hot encoded).
Goal: forecast sales in two stages — an XGBoost model predicts sales from lag/rolling/calendar/category features, then a GRU is trained on that model's residuals (using a 30-step window) to capture patterns XGBoost missed, and the two predictions are summed for the final forecast.
Result: Train MAE 56.44 → 21.33 after residual correction; Val MAE 72.81 → 89.85 after residual correction; Test MAE 71.49 → 92.90 after residual correction. Note the GRU correction *worsens* val/test error despite improving train error — a sign of overfitting on the residual model.
 
## How to Use
 
1. Install dependencies:
```bash
   pip install pandas numpy matplotlib xgboost tensorflow prophet scikit-learn statsmodels holidays
```
2. Open any notebook in Jupyter and run cells top to bottom.
3. Datasets referenced by the notebooks live in the `data/` folder (update local file paths in each notebook as needed — several currently point to absolute paths on the original author's machine).
## Notes
 
This is a learning repository — models and techniques are being implemented as they're learned, so results and code will keep evolving. Feedback and corrections are welcome via issues.
