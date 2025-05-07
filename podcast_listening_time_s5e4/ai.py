#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimised end‑to‑end pipeline for the 'Listening_Time_minutes' regression task.

Key improvements
----------------
1.  Robust Publication_Hour extraction (handles HH:MM, AM/PM, Morning/Night …)
2.  Frequency‑encoding of Podcast_Name  (powerful & tiny memory footprint)
3.  HashingVectorizer on Episode_Title  (adds text signal without huge matrices)
4.  RandomizedSearchCV on a 30 % sample  (fast hyper‑tuning of Random‑Forest)
5.  Competing HistGradientBoostingRegressor (hist‑GBDT) baseline
6.  Auto‑selects lower CV‑RMSE model, refits it on *all* rows, writes submission

Author: you (+ ChatGPT tweaks) – 2025‑04‑17
"""

###############################################################################
# Standard libraries
###############################################################################
import argparse, sys, os, math, warnings, joblib, json, gc
from pathlib import Path
import numpy as np
import pandas as pd

###############################################################################
# Scikit‑learn, etc.
###############################################################################
from scipy import sparse
from sklearn.base           import BaseEstimator, TransformerMixin
from sklearn.pipeline       import FunctionTransformer, Pipeline
from sklearn.compose        import ColumnTransformer
from sklearn.preprocessing  import OneHotEncoder
from sklearn.impute         import SimpleImputer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble       import RandomForestRegressor
from sklearn.experimental   import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble       import HistGradientBoostingRegressor
from sklearn.metrics        import mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)

###############################################################################
# ------------------------- helper transformers ------------------------------
###############################################################################
def publication_hour(val):
    """
    Parse Publication_Time values into an integer hour.
    Handles:
        - '07:30', '19:05'     -> 7, 19
        - '07' or '7'          -> 7
        - 'Morning', 'evening' -> 9, 19
        - 'AM', 'PM', 'Night'  -> 9, 15, 23
        - NaN                  -> np.nan
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # HH:MM or HH
    if s[:2].isdigit():
        try:
            return int(s.split(":")[0])
        except ValueError:
            pass
    # textual buckets
    s = s.lower()
    mapper = {
        "morning": 9,  "am": 9,
        "afternoon":15,"pm":15,
        "evening":19,
        "night":23, "latenight":23, "late_night":23,
    }
    return mapper.get(s, np.nan)

class PublicationHourExtractor(BaseEstimator, TransformerMixin):
    """Transform a single text column into a numeric hour column."""
    def fit(self, X, y=None): return self
    def transform(self, X):
        # X is a 2‑D array-like with one column
        return np.array([publication_hour(v) for v in np.ravel(X)]).reshape(-1, 1)

###############################################################################
# --------------------------- main pipeline ----------------------------------
###############################################################################
def load_data(train_path: Path, test_path: Path):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    return train, test

def add_frequency_encoding(train: pd.DataFrame, test: pd.DataFrame, col: str):
    freq = train[col].value_counts(normalize=True)
    train[f"{col}_Freq"] = train[col].map(freq)
    test [f"{col}_Freq"] = test [col].map(freq).fillna(0)
    return train, test

def build_preprocessor(numeric, categorical):
    """Return a ColumnTransformer with numeric, categorical, hour + text."""
    # pipelines
    num_pipe  = Pipeline([ ("imp", SimpleImputer(strategy="median")) ])
    cat_pipe  = Pipeline([
        ("imp",  SimpleImputer(strategy="most_frequent")),
        ("ohe",  OneHotEncoder(handle_unknown="ignore"))
    ])
    hour_pipe = Pipeline([
        ("ext", PublicationHourExtractor()),
        ("imp", SimpleImputer(strategy="median"))
    ])
    # hashing vectorizer (directly works as Transformer)
    text_vect = HashingVectorizer(
        n_features=2**18,      # 262 144 cols –  memory‑friendly & fast
        alternate_sign=False,
        norm=None,
        lowercase=True,
    )
    preprocess = ColumnTransformer([
        ("num",  num_pipe,                 numeric),
        ("cat",  cat_pipe,                 categorical),
        ("hour", hour_pipe,               ["Publication_Time"]),
        ("txt",  text_vect,               "Episode_Title"),
    ])
    return preprocess

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def fit_models(X, y, preprocess, n_jobs):
    """
    • RandomizedSearchCV → RandomForest on a 30 % row sample
    • HistGradientBoosting baseline (single fit, tuned params)
    Returns best model (full‑data refit) & a dict of CV scores.
    """
    scores = {}

    # ---- RF with hyper‑search on a sample -----------------------------------
    print("▶  RandomForest hyper‑search ...")
    # sample up to 250 k rows for speed
    samp_frac = 0.30 if len(X) > 250_000 else 1.0
    sample_idx = np.random.RandomState(42).choice(
        len(X), size=int(len(X)*samp_frac), replace=False
    )
    X_samp, y_samp = X.iloc[sample_idx], y[sample_idx]

    rf_pipe = Pipeline([
        ("prep", preprocess),
        ("rf",   RandomForestRegressor(
                    n_estimators=10,            # overridden in search
                    random_state=42,
                    n_jobs=n_jobs,
                 ))
    ])

    param_dist = {
        "rf__n_estimators":      [10],
        "rf__max_depth":         [20, 30, 40, 50, 60, None],
        "rf__min_samples_leaf":  [1, 3, 5],
        "rf__max_features":      ["sqrt", 0.3, 0.6],
    }
    rf_cv = RandomizedSearchCV(
        estimator=rf_pipe,
        param_distributions=param_dist,
        n_iter=3,
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        scoring="neg_root_mean_squared_error",
        verbose=1,
        n_jobs=n_jobs,
        random_state=42,
    )
    rf_cv.fit(X_samp, y_samp)
    rf_best_params = rf_cv.best_params_
    rf_cv_rmse     = -rf_cv.best_score_
    scores["RandomForest_CV_RMSE"] = rf_cv_rmse
    print(f"    RF best CV‑RMSE: {rf_cv_rmse:.4f}  best params: {rf_best_params}")

    # Re‑fit best RF on *all* rows -------------------------------------------
    best_rf = rf_cv.best_estimator_
    best_rf.fit(X, y)        # full data
    # ------------------------------------------------------------------------

    # # ---- HistGradientBoosting ----------------------------------------------
    # print("\n▶  HistGradientBoosting fit ...")
    # hgb_pipe = Pipeline([
    #     ("prep", preprocess),
    #     ("hgb",  HistGradientBoostingRegressor(
    #                 learning_rate=0.08,
    #                 max_depth=2,
    #                 max_iter=10,
    #                 l2_regularization=1.0,
    #                 random_state=42,
    #             ))
    # ])
    # cv = KFold(n_splits=3, shuffle=True, random_state=42)
    # cv_scores = []
    # for train_idx, val_idx in cv.split(X):
    #     hgb_pipe.fit(X.iloc[train_idx], y[train_idx])
    #     preds = hgb_pipe.predict(X.iloc[val_idx])
    #     cv_scores.append(rmse(y[val_idx], preds))
    # hgb_cv_rmse = np.mean(cv_scores)
    # scores["HGB_CV_RMSE"] = hgb_cv_rmse
    # print(f"    HGB mean CV‑RMSE: {hgb_cv_rmse:.4f}")

    # # fit on all rows
    # hgb_pipe.fit(X, y)

    # # choose model with lower CV RMSE
    # if hgb_cv_rmse < rf_cv_rmse:
    #     print("\n◎  Selected model:  HistGradientBoostingRegressor")
    #     return hgb_pipe, scores
    print("\n◎  Selected model:  RandomForestRegressor")
    return best_rf, scores

###############################################################################
# ----------------------------- main() ---------------------------------------
###############################################################################
def main(args):
    train, test = load_data(args.train, args.test)

    # frequency encoding
    train, test = add_frequency_encoding(train, test, "Podcast_Name")

    # target / features
    y = train["Listening_Time_minutes"].values
    X = train.drop(columns=["Listening_Time_minutes"])
    X_test  = test.copy()
    test_id = test["id"].values

    # feature categories
    numeric = [
        "Episode_Length_minutes",
        "Host_Popularity_percentage",
        "Guest_Popularity_percentage",
        "Number_of_Ads",
        # "Publication_Hour",     # for sanity if user already has numeric
        "Podcast_Name_Freq",
    ]
    categorical = [
        "Genre",
        "Publication_Day",
        "Episode_Sentiment",
    ]

    # build preprocessor
    preprocess = build_preprocessor(numeric, categorical)

    # train / validation quick split for visibility
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # fit models & pick the best
    best_model, cv_scores = fit_models(X_tr, y_tr, preprocess, args.jobs)

    # -------------------------------------------------------------
    # Validation RMSE for transparency
    val_pred = best_model.predict(X_val)
    val_rmse = rmse(y_val, val_pred)
    print(f"\nValidation hold‑out RMSE: {val_rmse:.4f}")

    # -------------------------------------------------------------
    # Train on *all* data (already done for best_model if RF/HGB)
    # Predict test & write submission
    test_pred = best_model.predict(X_test)

    out_df = pd.DataFrame({
        "id": test_id,
        "Listening_Time_minutes": test_pred
    })
    out_df.to_csv(args.output, index=False)
    print(f"\n✓ submission saved to: {args.output}")
    print("\nCV summary:", json.dumps(cv_scores, indent=2))

###############################################################################
# ------------------------------ CLI -----------------------------------------
###############################################################################
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Optimised podcast‑listening‑time pipeline")
    p.add_argument("--train",  type=Path, default=Path("data/train.csv"),
                   help="path to train.csv")
    p.add_argument("--test",   type=Path, default=Path("data/test.csv"),
                   help="path to test.csv")
    p.add_argument("--output", type=Path, default=Path("submission.csv"),
                   help="output CSV file")
    p.add_argument("--jobs",   type=int, default=-1,
                   help="n_jobs for parallel routines (‑1 = all cores)")
    args = p.parse_args()
    main(args)
