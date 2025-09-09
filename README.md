#  Customer Conversion Analysis (GUVI Capstone Project)

This project is part of my **GUVI Data Science Capstone**.  
It uses **clickstream data** from an e-commerce site to:

-  Predict if a browsing session will **convert** (Classification)  
-  Predict the **revenue** for a session (Regression)  
-  Group sessions into **clusters** (Clustering)  
-  Provide an easy-to-use **Streamlit app** for demo  

---

##  Project Structure
```text
clickstream-capstone/
├── data/
│ ├── raw/ # Original train/test CSVs
│ └── processed/ # Session-level processed data
├── models/ # Saved models (.pkl)
├── prep.py # Data preprocessing
├── train.py # Train classification & regression
├── train_clusters.py # Train clustering
├── app.py # Streamlit web app
└── README.md
```

---

##  How to Run

### 1. Setup Environment
```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
source .venv/bin/activate # Mac/Linux

pip install pandas scikit-learn streamlit joblib matplotlib
```
---

### 2. Prepare Data

Place your files in data/raw/ as:
- train.csv
- test.csv
Then run:
```bash
python prep.py
```
Creates data/processed/train_session.csv and test_session.csv.

---

### 3. Train Models

- Classification + Regression:
```bash
python train.py
```
- Clustering:
```bash
python train_clusters.py
```
---

### 4. Launch Streamlit App
```bash
streamlit run app.py
```
Upload data/processed/test_session.csv and try:

- Predict Conversion → shows conversion probabilities

- Predict Revenue → shows revenue predictions

- Assign Clusters → assigns each session to a cluster
---

## Results (on validation data)

- Classifier (Logistic Regression): ROC AUC ≈ 0.78

- Regressor (Random Forest): R² ≈ 0.99, MAE ≈ 1.19

- Clustering (KMeans): Best k = 4, Silhouette Score ≈ 0.40
---  

## Tech Stack

- Python (pandas, scikit-learn)

- Streamlit (deployment)

- Joblib (model saving/loading)
--- 

## Notes

- Data was aggregated per session (clickstream → session-level features).

- “Conversion” is defined when a session visited the Sale category.

- This project shows a complete workflow: Data Prep → Modeling → Deployment.

---
 Jaisrinivas P K
--
This project was developed as part of the GUVI Capstone submission for the Customer Conversion Analysis. 







