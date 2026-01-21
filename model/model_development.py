import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Data
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'data.csv')

print(f"Loading dataset from: {file_path}")
if not os.path.exists(file_path):
    print("ERROR: 'data.csv' not found!")
    exit()

# Load with headers (default behavior is correct for your file)
df = pd.read_csv(file_path)

# 2. Rename Columns to match our standard names
# Your file uses dots (e.g., 'Cl.thickness'). We rename them for clarity.
df.rename(columns={
    'Cl.thickness': 'Clump Thickness',
    'Cell.size': 'Uniformity of Cell Size',
    'Cell.shape': 'Uniformity of Cell Shape',
    'Marg.adhesion': 'Marginal Adhesion',
    'Mitoses': 'Mitoses'
}, inplace=True)

# 3. Feature Selection (We need these 5)
required_features = [
    'Clump Thickness', 
    'Uniformity of Cell Size', 
    'Uniformity of Cell Shape', 
    'Marginal Adhesion', 
    'Mitoses'
]
target = 'Class'

# 4. Preprocessing
print("Preprocessing data...")
# Drop missing values
df = df.dropna()

# Ensure inputs are numeric
for col in required_features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna() # Drop again if any coercing failed

# CHECK TARGET VALUES
# Your file already has 0 and 1, so we don't need to map 2 and 4.
unique_classes = df['Class'].unique()
print(f"Classes found: {unique_classes}")

X = df[required_features]
y = df['Class']

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train Model
print("Training SVM Model...")
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# 8. Save Model
save_path = os.path.join(current_dir, 'breast_cancer_model.pkl')
joblib.dump(model, save_path)
print(f"Model saved to: {save_path}")