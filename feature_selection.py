import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("omics_data.csv")

# Preprocess data: Remove non-numeric columns and handle missing values
data = data.dropna()
X = data.iloc[:, 1:-1]  
y = data.iloc[:, -1]  

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply LASSO for feature selection
lasso = LassoCV(cv=5).fit(X_scaled, y)
selected_features = data.columns[1:-1][lasso.coef_ != 0]

# Train a simple RandomForest model with selected features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled[:, lasso.coef_ != 0], y)

# Feature Importance Visualization
importance = rf.feature_importances_
plt.barh(selected_features, importance)
plt.xlabel("Feature Importance")
plt.ylabel("Selected Features")
plt.title("Feature Importance from Random Forest")
plt.show()

print(f"Selected Biomarkers: {list(selected_features)}")

