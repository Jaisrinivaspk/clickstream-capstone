# app.py  -- run: streamlit run app.py
import streamlit as st
import pandas as pd
import joblib

st.title("🛒 Clickstream Demo App")
st.write("Upload processed session-level data to predict conversion & revenue.")

uploaded = st.file_uploader("Upload CSV (from data/processed)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    features = [c for c in ["n_clicks", "mean_price", "max_price", "n_models"] if c in df.columns]
    X = df[features].fillna(0)

    # Conversion prediction
    if st.button("Predict Conversion"):
        clf = joblib.load("models/clf_lr.pkl")
        proba = clf.predict_proba(X)[:, 1]
        df_out = df.copy()
        df_out["p_convert"] = proba
        st.write("Top predictions:")
        st.dataframe(df_out.head(20))

    # Revenue prediction
    if st.button("Predict Revenue"):
        reg = joblib.load("models/reg_rf.pkl")
        preds = reg.predict(X)
        df_out = df.copy()
        df_out["pred_revenue"] = preds
        st.write("Top predictions:")
        st.dataframe(df_out.head(20))
        
# CLUSTERING
if st.button("Assign Clusters"):
    cluster_path = "models/cluster_kmeans.pkl"
    try:
        scaler, km = joblib.load(cluster_path)
        features = [c for c in ["n_clicks", "mean_price", "max_price", "n_models", "revenue"] if c in df.columns]
        X = df[features].fillna(0)
        labels = km.predict(scaler.transform(X))
        df_out = df.copy()
        df_out["cluster"] = labels
        st.write("Top cluster assignments:")
        st.dataframe(df_out.head(20))
    except Exception as e:
        st.error(f"Clustering model not found. Run train_clusters.py first. Error: {e}")
    
    
    
