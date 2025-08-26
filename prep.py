# prep.py  -- run: python prep.py
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

def read_try(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=';')

print("Reading raw files...")
train = read_try(RAW / "train.csv")
test  = read_try(RAW / "test.csv")

# Normalize column names to lowercase and underscores
def normalize_cols(df):
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

train = normalize_cols(train)
test = normalize_cols(test)

print("Train columns:", list(train.columns)[:50])

# find key column names (robust)
def find_col(cols, keywords):
    for k in keywords:
        for c in cols:
            if k in c:
                return c
    return None

session_col = find_col(train.columns, ["session", "session_id"])
price_col   = find_col(train.columns, ["price"])
page1_col   = find_col(train.columns, ["page1", "main_category", "page1_main_category"])
order_col   = find_col(train.columns, ["order", "click", "seq"])

if session_col is None:
    raise SystemExit("No session column found. Please check the files in data/raw and ensure a session id column exists (name containing 'session').")

print("Detected columns -> session:", session_col, " price:", price_col, " page1:", page1_col, " order:", order_col)

# ensure price numeric
if price_col:
    train[price_col] = pd.to_numeric(train[price_col], errors='coerce').fillna(0)
    test[price_col]  = pd.to_numeric(test[price_col], errors='coerce').fillna(0)

# create a simple sale flag: page1 == 4 (common for this dataset)
if page1_col:
    train[page1_col] = pd.to_numeric(train[page1_col], errors='coerce')
    test[page1_col]  = pd.to_numeric(test[page1_col], errors='coerce')
    train["is_sale"] = (train[page1_col] == 4).astype(int)
    test["is_sale"]  = (test[page1_col] == 4).astype(int)
else:
    train["is_sale"] = 0
    test["is_sale"]  = 0

# aggregate to session level
def make_session_df(df):
    g = df.groupby(session_col)
    converted = g["is_sale"].max().astype(int)
    revenue = g[price_col].sum() if price_col else g["is_sale"].sum()
    n_clicks = g.size()
    mean_price = g[price_col].mean() if price_col else g["is_sale"].mean()
    max_price = g[price_col].max() if price_col else g["is_sale"].max()

    # a fallback for unique models if column exists
    n_models = g["page2_clothing_model"].nunique() if "page2_clothing_model" in df.columns else g[page1_col].nunique() if page1_col else g.size()

    sess = pd.DataFrame({
        session_col: converted.index,
        "converted": converted.values,
        "revenue": revenue.values,
        "n_clicks": n_clicks.values,
        "mean_price": mean_price.values,
        "max_price": max_price.values,
        "n_models": n_models.values
    })
    return sess

train_sess = make_session_df(train)
test_sess  = make_session_df(test)

train_sess.to_csv(OUT / "train_session.csv", index=False)
test_sess.to_csv(OUT / "test_session.csv", index=False)

print("Wrote:", OUT / "train_session.csv", "shape:", train_sess.shape)
print("Wrote:", OUT / "test_session.csv", "shape:", test_sess.shape)
