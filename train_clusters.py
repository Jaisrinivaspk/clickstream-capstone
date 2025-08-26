# train_clusters.py -- run: python train_clusters.py
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "processed" / "train_session.csv"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# load processed data
df = pd.read_csv(DATA)
print("Loaded:", df.shape)

# features for clustering (numeric session stats)
features = [c for c in ["n_clicks", "mean_price", "max_price", "n_models", "revenue"] if c in df.columns]
X = df[features].fillna(0)

# scale numeric features
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# try different k values and check silhouette score
best_k, best_sil, best_model = None, -1, None
for k in [3, 4, 5, 6]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    score = silhouette_score(Xs, labels)
    print(f"k={k}, silhouette={score:.4f}")
    if score > best_sil:
        best_k, best_sil, best_model = k, score, km

print(f"\nBest k={best_k} with silhouette={best_sil:.4f}")

# save scaler + model
joblib.dump((scaler, best_model), MODELS / "cluster_kmeans.pkl")
print("Saved clustering model:", MODELS / "cluster_kmeans.pkl")
