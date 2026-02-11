# importing Core libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

import shap
from sklearn.inspection import PartialDependenceDisplay

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (8, 6)
np.random.seed(42)
# Loading dataset
df = pd.read_csv(r"C:\Users\Admin\Downloads\Real estate (1).csv")
df.head()
df.columns = [c.strip().replace(" ", "_") for c in df.columns]
target = "Y_house_price_of_unit_area"
X = df.drop(columns=[target])
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

print(xgb_model.fit(X_train, y_train))

# SHAP for Random Forest - bagging model
rf_explainer = shap.TreeExplainer(rf_model)
rf_shap_values = rf_explainer.shap_values(X_test)

print(shap.summary_plot(rf_shap_values, X_test, show=True))
# SHAP for XGBoost - boosting model
xgb_explainer = shap.TreeExplainer(xgb_model)
xgb_shap_values = xgb_explainer.shap_values(X_test)

print(shap.summary_plot(xgb_shap_values, X_test, show=True))
# Mean absolute SHAP values
rf_importance = np.abs(rf_shap_values).mean(axis=0)
xgb_importance = np.abs(xgb_shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "RF_SHAP": rf_importance,
    "XGB_SHAP": xgb_importance
}).sort_values(by="RF_SHAP", ascending=False)

print(importance_df.head())
important_feature = importance_df.iloc[0]["Feature"]
print(important_feature)

PartialDependenceDisplay.from_estimator(
    rf_model,
    X_train,
    features=[important_feature],
    grid_resolution=50
)

plt.title(f"Random Forest PDP: {important_feature}")
plt.show()
PartialDependenceDisplay.from_estimator(
    xgb_model,
    X_train,
    features=[important_feature],
    grid_resolution=50
)

plt.title(f"XGBoost PDP: {important_feature}")
plt.show()


