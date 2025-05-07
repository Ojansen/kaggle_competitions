#!/usr/bin/env python
\"\"\"Podcast listening‑time prediction pipeline
Usage (from a shell):
    python podcast_flow.py --train train.csv --test test.csv --out submission.csv
The script prints an RMSE benchmark on a hold‑out split, then fits the model
on the full training data and writes predictions for the test set.
\"\"\"

import argparse, sys, os, math, json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics      import mean_squared_error
from sklearn.ensemble     import RandomForestRegressor
from sklearn.pipeline     import Pipeline
from sklearn.compose      import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute       import SimpleImputer

# --------------------------------------------------------------------------- #
#                               Feature helpers                               #
# --------------------------------------------------------------------------- #
def publication_hour(val):
    \"\"\"Convert '07:30' or 'Night' ➜ integer 0‑23, or np.nan.\"\"\"
    mapping = {\"Morning\": 9, \"Afternoon\": 15, \"Evening\": 19, \"Night\": 23}
    if pd.isna(val):
        return np.nan
    try:
        return int(str(val).split(\":\")[0])
    except ValueError:
        return mapping.get(str(val).title(), np.nan)

def build_preprocessor():
    numeric_features = [
        \"Episode_Length_minutes\",
        \"Host_Popularity_percentage\",
        \"Guest_Popularity_percentage\",
        \"Number_of_Ads\",
        \"Publication_Hour\",     # engineered below
    ]

    categorical_features = [
        \"Genre\",
        \"Publication_Day\",
        \"Episode_Sentiment\",
    ]

    numeric_pipeline = Pipeline([
        (\"imputer\", SimpleImputer(strategy=\"median\")),
    ])

    categorical_pipeline = Pipeline([
        (\"imputer\", SimpleImputer(strategy=\"most_frequent\")),
        (\"onehot\",  OneHotEncoder(handle_unknown=\"ignore\")),
    ])

    preprocessor = ColumnTransformer([
        (\"num\", numeric_pipeline, numeric_features),
        (\"cat\", categorical_pipeline, categorical_features),
    ])

    return preprocessor, numeric_features + categorical_features

# --------------------------------------------------------------------------- #
#                                   Main run                                  #
# --------------------------------------------------------------------------- #
def main(train_path, test_path, out_path):
    print(\"Loading data …\", file=sys.stderr)
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    # quick EDA prints
    print(f\"Train shape: {train.shape};  Test shape: {test.shape}\")
    print(\"Missing‑value ratio (train):\")
    print((train.isna().mean()*100).round(2).astype(str)+' %')

    # ---------------------------------------------------- #
    # Feature engineering
    # ---------------------------------------------------- #
    for df in (train, test):
        df[\"Publication_Hour\"] = df[\"Publication_Time\"].apply(publication_hour)

    drop_cols = [\"Episode_Title\", \"Publication_Time\", \"Podcast_Name\"]  # hi‑cardinality
    train = train.drop(columns=drop_cols)
    test  = test.drop (columns=drop_cols)

    y = train[\"Listening_Time_minutes\"]
    X = train.drop(columns=[\"Listening_Time_minutes\", \"id\"])
    X_test = test.drop(columns=[\"id\"])

    preprocessor, _ = build_preprocessor()

    # ---------------------------------------------------- #
    # Model & validation
    # ---------------------------------------------------- #
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = Pipeline([
        (\"prep\",  preprocessor),
        (\"rf\",    RandomForestRegressor(
                     n_estimators=400,
                     max_depth=None,
                     min_samples_leaf=1,
                     n_jobs=-1,
                     random_state=42)),
    ])

    print(\"Training Random‑Forest …\", file=sys.stderr)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    rmse = math.sqrt(mean_squared_error(y_val, val_pred))
    print(f\"Hold‑out RMSE: {rmse:0.3f} minutes\")

    # ---------------------------------------------------- #
    # Re‑fit on full data and predict test set
    # ---------------------------------------------------- #
    print(\"Refitting on full training set …\", file=sys.stderr)
    model.fit(X, y)

    print(f\"Predicting {len(X_test)} test rows …\", file=sys.stderr)
    test_pred = model.predict(X_test)

    submission = pd.DataFrame({
        \"id\": test[\"id\"],
        \"Listening_Time_minutes\": test_pred,
    })

    submission.to_csv(out_path, index=False)
    print(f\"Saved predictions to {out_path}\")

# --------------------------------------------------------------------------- #
if __name__ == \"__main__\":
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--train\", default=\"train.csv\", help=\"path to train.csv\")
    parser.add_argument(\"--test\",  default=\"test.csv\",  help=\"path to test.csv\")
    parser.add_argument(\"--out\",   default=\"submission.csv\", help=\"output CSV path\")
    args = parser.parse_args()
    main(args.train, args.test, args.out)
"""