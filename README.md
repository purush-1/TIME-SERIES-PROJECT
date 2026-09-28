# TIME-SERIES-PROJECT
 
A collection of time series forecasting experiments — from walk-forward validated tree models to two-stage/weighted hybrids combining XGBoost with Prophet, LSTM, and GRU, plus a Box-Cox + feature-selection pipeline. Each notebook tackles a different dataset, and where a naive baseline was computed, the model is benchmarked against it.
 
## Repository Structure
 
| Notebook | Dataset | Approach |
|---|---|---|
| `AEP HYBRID MODEL XGBOOST + XGBOOST.ipynb` | AEP hourly energy consumption (MW) | XGBoost + a second XGBoost trained on stage-1 residuals |
| `AIR PASSENGERS WALK FORWARD VALIDATION.ipynb` | Monthly airline passenger counts | XGBoost, walk-forward (retrain-per-step) validation |
| `BALITIMORE WATER USAGE WITH WALK FORWARD VALIDATION.ipynb` | Baltimore yearly water usage | XGBoost, walk-forward validation |
| `CHAMPAGNE MONTHLY SALES WALK FORWARD VALIDATION.ipynb` | Monthly champagne sales | XGBoost, walk-forward validation |
| `DEMAND FORECASTING PROPHET + XGBOOST.ipynb` | Retail store inventory / demand | Prophet, then XGBoost trained on Prophet's residuals |
| `FAVORITA STORE SALES XGBOOST + LSTM WEIGHTED PREDICTION HYBRID MODEL.ipynb` | Favorita store sales (Kaggle) | XGBoost + LSTM, combined as a weighted average |
| `PJMW MW HOURLY PREDICTION LSTM.ipynb` | PJME hourly power consumption (MW) | LSTM |
| `SALES PREDICTION HYBRID MODEL XGBOOST + GRU.ipynb` | Superstore sales (daily totals) | XGBoost + a GRU trained on stage-1 residuals |
| `SALES PREDICTION WITH BOX COX TRANSFORMATION AND FEATURE SELECTION.ipynb` | Superstore sales (daily totals) | `SelectKBest` feature selection + Box-Cox scaling, then XGBoost |
| `data/` | Datasets used by the notebooks | — |
 
## Libraries Used
 
| Notebook | Libraries |
|---|---|
| AEP Hybrid (XGBoost + XGBoost) | `pandas`, `matplotlib`, `xgboost` (`XGBRegressor`), `sklearn.metrics` |
| Air Passengers (walk-forward) | `pandas`, `numpy`, `xgboost`, `sklearn.metrics`, `matplotlib` |
| Baltimore Water Usage (walk-forward) | `pandas`, `xgboost`, `sklearn.metrics`, `matplotlib` |
| Champagne Monthly Sales (walk-forward) | `pandas`, `xgboost`, `sklearn.metrics`, `math.sqrt`, `matplotlib` |
| Demand Forecasting (Prophet + XGBoost) | `pandas`, `prophet` (`Prophet`), `xgboost`, `sklearn.metrics`, `matplotlib` |
| Favorita Store Sales (XGBoost + LSTM) | `pandas`, `numpy`, `holidays`, `xgboost`, `tensorflow.keras` (`Sequential`, `Dense`, `LSTM`, `Dropout`, `L2`, `SGD`), `sklearn.metrics`, `sklearn.model_selection.TimeSeriesSplit`, `statsmodels` (`plot_acf`, `plot_pacf`), `matplotlib` |
| PJMW Hourly Prediction (LSTM) | `pandas`, `numpy`, `tensorflow.keras` (`Sequential`, `Dense`, `LSTM`), `sklearn.metrics`, `matplotlib` |
| Sales Prediction (XGBoost + GRU) | `pandas`, `numpy`, `xgboost`, `tensorflow.keras` (`Sequential`, `Dense`, `GRU`, `Dropout`, `L2`, `RMSprop`), `sklearn.metrics`, `sklearn.model_selection.TimeSeriesSplit`, `matplotlib` |
| Sales Prediction (Box-Cox + Feature Selection) | `pandas`, `numpy`, `xgboost`, `sklearn.feature_selection` (`SelectKBest`, `f_regression`), `sklearn.preprocessing` (`PowerTransformer`), `sklearn.metrics`, `matplotlib` |
 
## Naive Baseline vs. Model Predictions (MAE unless noted)
 
Only notebooks that computed a usable naive baseline are included here. Numbers are pulled straight from each notebook's printed output. See [Known Issues](#known-issues--to-fix) before drawing conclusions from any of them.
 
| Notebook | Naive Baseline | Model Prediction |
|---|---|---|
| **Baltimore Water Usage** (walk-forward) | Persistence: 20.89 | Walk-forward MAE: 20.32 |
| **Champagne Monthly Sales** (walk-forward, **RMSE**) | Persistence RMSE: 2677.20 | Walk-forward RMSE: 24.38 (⚠️ likely inflated by leakage, see Known Issues) |
| **Demand Forecasting** (Prophet + XGBoost) | Persistence: 116.32 | Prophet only — Train: 86.29, Test: 86.07 <br> Final (+ XGBoost on residuals) — Train: 7.31, Test: 7.59 (⚠️ see Known Issues) |
| **Favorita Store Sales** (XGBoost + LSTM) | Persistence (7-day rolling mean): 257.35 | XGBoost — Train: 185.33, Val: 237.87, Test: 227.13 <br> XGBoost, walk-forward (retrain every 7 days) — Val: 249.85, Test: 235.90 <br> LSTM — Train: 62.57, Val: 258.25, Test: 235.32 <br> Weighted hybrid (0.6·XGB + 0.4·LSTM) — Train: 221.37, Val: 239.42, Test: 225.24 |
| **Sales Prediction, Box-Cox + Feature Selection** | 30-day moving average (train set): 30,289.64 | XGBoost — Train: 11,847.12, Val: 10,863.33 (⚠️ val overlaps train, see Known Issues) |
 
**Takeaways:**
- Baltimore Water Usage's XGBoost model barely edges out the persistence baseline (20.32 vs. 20.89) — the naive forecast is nearly as good here.
- Demand Forecasting: Prophet alone (86 MAE) already beats persistence (116 MAE). The residual-correction step drops it to ~7.5 MAE, but that result is suspect until the `Lag 30` look-ahead feature is fixed.
- Favorita's weighted hybrid (test 225.24) beats the persistence baseline (257.35) but is only marginally better than plain XGBoost (227.13). On validation the hybrid (239.42) is actually slightly worse than plain XGBoost (237.87), so the 0.6/0.4 weighting isn't adding much.
- Box-Cox + feature selection: XGBoost's train error (11.8k) is well below the 30-day moving-average baseline (30.3k), but the model hasn't been evaluated on a true held-out test set yet.
## Other Notebooks (no usable naive baseline yet)
 
**AEP Hybrid (XGBoost + XGBoost)**
Dataset: AEP hourly power consumption (MW), outlier-filtered to the 11,000–20,000 MW range.
Goal: forecast hourly power consumption. A first XGBoost model predicts consumption from lag (1h, 2h), rolling mean (7h, 24h), and calendar features; a second XGBoost model is trained on the first model's residuals and used to correct its predictions.
Result: Train MAE 534.32 → 321.88 after residual correction; Test MAE 569.23 → 324.78 after residual correction.
 
**Air Passengers (walk-forward)**
Dataset: the classic monthly international airline passengers series (1949–1960).
Goal: forecast monthly passenger counts using lag (1, 6, 12 months), rolling mean/std, and calendar features, with an XGBoost model refit at every step of a walk-forward validation loop.
Result: walk-forward MAE 123.48. The mean residual is +119.65, so almost all of the error is systematic under-prediction — consistent with tree models struggling to extrapolate the upward trend (and with the history bug noted below).
 
**PJMW Hourly Prediction (LSTM)**
Dataset: PJME hourly power consumption (MW), filtered to values ≥ 20,000 MW.
Goal: forecast the next hour's power consumption from a 30-step window of lag, rolling mean/std, and calendar features, using a stacked LSTM + Dense network.
Result: Train MAE 204.74, Val MAE 370.76, Test MAE 605.26 (about 1.9% of the ~32,100 MW mean load). The growing gap from train to test suggests some overfitting or distribution shift over time worth investigating.
 
**Sales Prediction (XGBoost + GRU)**
Dataset: Superstore sales, aggregated to daily totals (Sales, Quantity, Price, Cost, Profit), outlier-filtered.
Goal: forecast daily sales in two stages — an XGBoost model predicts sales from lag/rolling/calendar features (out-of-fold predictions from `TimeSeriesSplit` are used for the training residuals), then a GRU is trained on those residuals (30-step window) to capture patterns XGBoost missed, and the two predictions are summed for the final forecast.
Result (MAE): XGBoost only — Train 6,040.33, Val 8,237.77, Test 7,603.23. After GRU residual correction — Train 3,101.73, Val 13,193.89, Test 11,745.18. The GRU correction improves train error but **worsens** val/test error substantially, a sign of overfitting in the residual model.
 
**Sales Prediction with Box-Cox Transformation and Feature Selection**
Dataset: Superstore sales, aggregated to daily totals (Sales, Profit, Quantity); non-positive-profit days and IQR outliers removed.
Goal: forecast next-day sales. Lag (1, 7, 30), rolling mean/std (7, 30), and calendar features are built, the best 10 are picked with `SelectKBest(f_regression)`, features and target are Box-Cox transformed with `PowerTransformer`, and an XGBoost model is trained on the result (predictions are inverse-transformed before scoring).
Result: Train MAE 11,847.12, Val MAE 10,863.33; the 30-day moving average baseline scores 30,289.64 on the train set.
 
## Known Issues / To Fix
 
Found while reviewing the notebooks against their outputs. None of these are hidden in the numbers above, but they mean some results shouldn't be read as clean generalization estimates yet.
 
- **Walk-forward history doesn't accumulate (Air Passengers, Champagne):** inside `validate`, `history_x` is rebuilt as `x_train + latest test row` each step instead of appending to the running history, so the model never sees more than one test point at a time. Baltimore's loop does this correctly and can be used as the template.
- **Champagne train/test leakage:** the train filter is `(index >= '1964-01-01') & (index > '1969-12-01')` with no upper bound, so the training set includes the validation *and* test months. The 24.38 RMSE is almost certainly a result of this. The variable is also named `mase` but the metric computed is RMSE, and the persistence RMSE (2677.20) is computed over the whole series rather than just the test window.
- **Demand Forecasting look-ahead:** `Lag 30` is built with `shift(-30)` (a *future* residual), which leaks information into the XGBoost stage. The persistence baseline also shifts by one row across different store/product combinations rather than within a single series.
- **Box-Cox notebook split:** `val` and `test` are both created by splitting the *full* dataset in half (`split(month_sales, 0.5)`), so `val` sits entirely inside the training range. The validation MAE is therefore not held-out, and the test set is never scored.
- **Sales XGBoost + GRU baseline and features:** the residual feature step overwrites `MA 7` (and the other lag/rolling columns) with residual-based versions, so the printed "PERSISTENT ERROR" is not a valid persistence baseline. Same-day Quantity/Price/Cost/Profit are also used to predict same-day Sales, which wouldn't be available at forecast time.
- **AEP:** train and test windows overlap by ~23 hours around 2015-03-09/10, and the 11,000–20,000 MW filter is hard-coded (the IQR bounds are computed but unused).
- **Favorita:** the notebook trains on a random 50,000-row sample of the full dataset (with duplicate sales values dropped), not on complete per-store series, so results are not directly comparable to the Kaggle leaderboard.
## How to Use
 
1. Install dependencies:
```bash
   pip install pandas numpy matplotlib xgboost tensorflow prophet scikit-learn statsmodels holidays
```
2. Open any notebook in Jupyter and run cells top to bottom.
3. Datasets referenced by the notebooks live in the `data/` folder (update local file paths in each notebook as needed — several currently point to absolute paths on the original author's machine).
## Notes
 
This is a learning repository — models and techniques are being implemented as they're learned, so results and code will keep evolving. Feedback and corrections are welcome via issues.
