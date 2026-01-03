import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Paths
DATASET_FOLDER = "Datasets"
FINAL_DATA_CSV = os.path.join(DATASET_FOLDER, "FINAL DATASET.csv")
MODEL_OUT = "pipe.pkl"

if not os.path.exists(FINAL_DATA_CSV):
    raise SystemExit(f"Missing {FINAL_DATA_CSV}. Run cleaning_data.py first to generate it.")

# Load data
df = pd.read_csv(FINAL_DATA_CSV)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)

# Preprocessing + model pipeline
trf = ColumnTransformer([
    ("trf", OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore'),
     ["batting_team", "bowling_team", "venue"])
], remainder="passthrough")

rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=100)
pipe = Pipeline([('step1', trf), ('step2', rf)])

print("Training RandomForest pipeline...")
pipe.fit(X_train, y_train)

# Evaluate
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")

# Save model
joblib.dump(pipe, MODEL_OUT)
print(f"Saved trained pipeline to {MODEL_OUT}")
