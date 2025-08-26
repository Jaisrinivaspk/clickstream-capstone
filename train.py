# train.py  -- run: python train.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, roc_auc_score, mean_absolute_error, r2_score
import joblib

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "processed" / "train_session.csv"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
print("Loaded processed train:", df.shape)

# features we engineered in prep.py
features = [c for c in ["n_clicks", "mean_price", "max_price", "n_models"] if c in df.columns]
print("Using features:", features)
X = df[features].fillna(0)
y_clf = df["converted"]
y_reg = df["revenue"]

# CLASSIFICATION (conversion)
X_train, X_val, y_train, y_val = train_test_split(X, y_clf, test_size=0.2, stratify=y_clf, random_state=42)
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
preds = clf.predict(X_val)
proba = clf.predict_proba(X_val)[:,1]
print("=== Classifier Results ===")
print(classification_report(y_val, preds))
print("ROC AUC:", roc_auc_score(y_val, proba))
joblib.dump(clf, MODELS / "clf_lr.pkl")

# REGRESSION (revenue)
X_train, X_val, y_train, y_val = train_test_split(X, y_reg, test_size=0.2, random_state=42)
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)
preds = reg.predict(X_val)
print("\n=== Regressor Results ===")
print("MAE:", mean_absolute_error(y_val, preds))
print("R² :", r2_score(y_val, preds))
joblib.dump(reg, MODELS / "reg_rf.pkl")

print("\nSaved models to:", MODELS)
