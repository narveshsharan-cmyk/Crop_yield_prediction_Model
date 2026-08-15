import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. LOADING THE DATA
# ==========================================
# Ensure your downloaded Excel sheet is in the same folder as this script
file_name = 'Crop yield.xlsx' 

try:
    df = pd.read_excel(file_name)
except:
    df =pd.read_csv(r"C:\Users\HP\OneDrive\Documents\Crop yield\crop_yield.csv")


print("✅ Data loaded successfully into Python.")

# ==========================================
# 2. MATCHING YOUR VARIABLES (From your Workspace)
# ==========================================
# Columns extracted from your screen's Predictor list
features = ['Crop', 'State', 'Area', 'Crop_Year', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Season']
target = 'Production'  

X = df[features]
y = df[target]

# Automatically convert text categories (Crop, State, Season) to numbers
X = pd.get_dummies(X, drop_first=True)

# ==========================================
# 3. SPLITTING THE DATA (80% Train, 20% Test)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ==========================================
# 4. TRAINING THE MODEL (Fine Tree equivalent)
# ==========================================
# Your screen shows a 'Fine Tree' model; in Python, this is a DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=20, random_state=42)
model.fit(X_train, y_train)
print("🏋️ Model training complete.")

# ==========================================
# 5. CREATING YOUR RESULTS TABLE
# ==========================================
y_pred = model.predict(X_test)

# This creates the exact equivalent of your 'resultsTable' variable
resultsTable = pd.DataFrame({
    'Actual_Production': y_test,
    'Predicted_Production': y_pred,
    'Error': y_test - y_pred
})
resultsTable.to_excel('resultsTable.xlsx', index=False)
print("💾 Results exported locally to 'resultsTable.xlsx'")

# ==========================================
# 6. SAVING THE MODEL TO YOUR SYSTEM (.pkl)
# ==========================================
# Just like a .mat file in MATLAB, Python uses .pkl files to store models
with open('trainedModel.pkl', 'wb') as f:
    pickle.dump(model, f)
print("💾 Python model saved onto your system as 'trainedModel.pkl'")

# ==========================================
# 7. GENERATING THE IMPORTANCE PLOT
# ==========================================
# This replicates the blue horizontal chart displayed on your screen
importances = model.feature_importances_
indices = np.argsort(importances)[-5:]  # Top 5 most impactful features

plt.figure(figsize=(10, 6))
plt.title('Permutation Importance for Model 1 (Fine Tree)')
plt.barh(range(len(indices)), importances[indices], color='blue', align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Mean Importance')
plt.tight_layout()
plt.show()