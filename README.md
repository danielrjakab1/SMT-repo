# SMT-repo
Baseball Baserunning Analytics with Player Tracking Data

This project uses high-frequency player and ball tracking data to model key baserunning events in baseball using machine learning. Starting from raw positional data stored in DuckDB, I engineered custom kinematic features—including lead distance, velocity, acceleration, jerk, sprint speed, and catcher pop time—to quantify how player movement influences on-field outcomes.

The project builds end-to-end data pipelines in SQL and Python to:

Estimate player sprint speed directly from tracking data
Calculate catcher pop times from ball acquisition events
Model pickoff probability using runner movement before pitcher release
Predict stolen base success using runner, catcher, and tracking-based features
Analyze baserunner advancement on balls in play using pre-pitch positioning and movement

Several machine learning models are evaluated, including:

XGBoost
LightGBM
Random Forests
Logistic Regression (L1/L2)

Model performance is measured using repeated stratified cross-validation with ROC-AUC, class balancing, median imputation, and early stopping where appropriate. SHAP values are used to interpret feature importance and explain model predictions.

Tech Stack
Python
SQL (DuckDB)
Pandas & NumPy
Scikit-learn
XGBoost
LightGBM
SHAP
Matplotlib
Skills Demonstrated
Sports analytics
Feature engineering from tracking data
SQL query optimization
Machine learning for classification
Model evaluation and cross-validation
Explainable AI (SHAP)
Data visualization
Baseball biomechanics and baserunning analysis
