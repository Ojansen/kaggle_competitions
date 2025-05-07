#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep‑learning solution for the Podcast‑Listening‑Time regression task.
Keras Functional API with entity embeddings, numeric features, EarlyStopping.

Author: you (+ChatGPT tweaks) – 2025‑04‑17
"""

import argparse, hashlib, math, json, os, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore", category=FutureWarning)


# --------------------------------------------------------------------------- #
# ---------------------------- feature helpers ------------------------------ #
# --------------------------------------------------------------------------- #
def publication_hour(val):
    """Robust parse of Publication_Time ➜ hour (0‑23) or NaN."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if ":" in s and s.split(":")[0].isdigit():
        return int(s.split(":")[0])
    if s.isdigit():
        return int(s)
    txt = {
        "morning": 9,
        "am": 9,
        "afternoon": 15,
        "pm": 15,
        "evening": 19,
        "night": 23,
        "late_night": 23,
        "latenight": 23,
    }
    return txt.get(s, np.nan)


def hash_bucket(x: str, buckets: int = 50000) -> int:
    """Fast, reproducible string‑to‑bucket hash (MD5)."""
    return int(hashlib.md5(x.encode()).hexdigest(), 16) % buckets


# --------------------------------------------------------------------------- #
# -------------------------- model‑building utils --------------------------- #
# --------------------------------------------------------------------------- #
def build_tabular_dnn(
    numeric_dim: int,
    cat_vocab_sizes: dict,
    embed_dim_rule=lambda v: min(50, int(round(v**0.25) * 4) + 1),
):
    """
    Build a Keras Functional model with:
      • numeric Dense block (after StandardScaling),
      • one embedding per categorical input,
      • concatenation → MLP → single regression output.
    """
    inputs = {}
    embeds = []

    # numeric vector
    inputs["num"] = keras.Input(shape=(numeric_dim,), name="num")
    concat_list = [inputs["num"]]

    # categorical embeddings
    for name, vocab in cat_vocab_sizes.items():
        inputs[name] = keras.Input(shape=(1,), dtype="int32", name=name)
        d = embed_dim_rule(vocab)
        emb = layers.Embedding(vocab, d, name=f"emb_{name}")(inputs[name])
        emb = layers.Reshape((d,))(emb)
        concat_list.append(emb)
        embeds.append((name, vocab, d))

    x = layers.Concatenate()(concat_list)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(1, name="output")(x)

    model = keras.Model(inputs=list(inputs.values()), outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    print("\nEmbeddings summary:")
    for n, v, d in embeds:
        print(f"  {n:18s}: vocab={v:<6d} → dim={d}")
    model.summary()
    return model


# --------------------------------------------------------------------------- #
# ----------------------------- main routine -------------------------------- #
# --------------------------------------------------------------------------- #
def main(args):
    # ---------------- data load -------------------------------------------- #
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    print("Train shape:", train.shape, "  Test shape:", test.shape)

    # ---------------- feature engineering ---------------------------------- #
    for df in (train, test):
        df["Pub_Hour"] = df["Publication_Time"].apply(publication_hour)
        # frequency of Podcast_Name (helps more than plain hashing alone)
        freq = train["Podcast_Name"].value_counts(normalize=True)
        df["Podcast_Freq"] = df["Podcast_Name"].map(freq).fillna(0.0)
        # hash buckets
        df["Podcast_Bucket"] = df["Podcast_Name"].apply(hash_bucket)
        df["Genre_Bucket"] = df["Genre"].apply(hash_bucket, buckets=32)
        df["PubDay_Bucket"] = df["Publication_Day"].apply(hash_bucket, buckets=16)
        df["Sentiment_Bucket"] = df["Episode_Sentiment"].apply(hash_bucket, buckets=8)

    target = train["Listening_Time_minutes"].values

    numeric_cols = [
        "Episode_Length_minutes",
        "Host_Popularity_percentage",
        "Guest_Popularity_percentage",
        "Number_of_Ads",
        "Pub_Hour",
        "Podcast_Freq",
    ]
    cat_cols = {
        "Podcast_Bucket": 50000,
        "Genre_Bucket": 32,
        "PubDay_Bucket": 16,
        "Sentiment_Bucket": 8,
    }

    # ------------- numeric preprocessing (median + scaler) ----------------- #
    num_imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    train_num = scaler.fit_transform(num_imputer.fit_transform(train[numeric_cols]))
    test_num = scaler.transform(num_imputer.transform(test[numeric_cols]))

    # ------------- categorical to int32 np arrays -------------------------- #
    train_cat = {name: train[name].astype("int32").values for name in cat_cols}
    test_cat = {name: test[name].astype("int32").values for name in cat_cols}

    # ------------- Train‑valid split --------------------------------------- #
    idx = np.arange(len(train_num))
    idx_tr, idx_val = train_test_split(idx, test_size=0.15, random_state=42)

    X_tr_num, X_val_num = train_num[idx_tr], train_num[idx_val]
    y_tr, y_val = target[idx_tr], target[idx_val]
    X_tr_cat = {k: v[idx_tr] for k, v in train_cat.items()}
    X_val_cat = {k: v[idx_val] for k, v in train_cat.items()}
    # ------------- Build & train model ------------------------------------- #
    model = build_tabular_dnn(numeric_dim=X_tr_num.shape[1], cat_vocab_sizes=cat_cols)

    train_inputs = {"num": X_tr_num, **{k: X_tr_cat[k] for k in cat_cols}}
    val_inputs = {"num": X_val_num, **{k: X_val_cat[k] for k in cat_cols}}

    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=3, restore_best_weights=True, monitor="val_rmse"
        ),
        keras.callbacks.TensorBoard(
            log_dir="logs",
            histogram_freq=1,
            write_graph=True,
            write_images=True,
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_rmse", factor=0.5, patience=2),
    ]
    model.fit(
        train_inputs,
        y_tr,
        validation_data=(val_inputs, y_val),
        epochs=40,
        batch_size=8192,
        callbacks=callbacks,
        verbose=2,
    )

    # ------------- Validation RMSE ---------------------------------------- #
    val_pred = model.predict(val_inputs, batch_size=8192, verbose=0).squeeze()
    val_rmse = math.sqrt(mean_squared_error(y_val, val_pred))
    print(f"\nHold‑out RMSE: {val_rmse:.4f}")

    # ------------- Train on full data  ------------------------------------ #
    full_inputs = {"num": train_num, **train_cat}
    model.fit(
        full_inputs,
        target,
        epochs=callbacks[0].stopped_epoch + 1,  # same #epochs as before
        batch_size=8192,
        verbose=2,
    )

    # ------------- Predict test & write submission ------------------------ #
    test_inputs = {"num": test_num, **test_cat}
    test_pred = model.predict(test_inputs, batch_size=8192, verbose=0).squeeze()

    submission = pd.DataFrame({"id": test["id"], "Listening_Time_minutes": test_pred})
    submission.to_csv(args.output, index=False)
    print("\n✓ submission saved to:", args.output)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Keras DNN for podcast listening‑time")
    ap.add_argument("--train", type=Path, default=Path("data/train.csv"))
    ap.add_argument("--test", type=Path, default=Path("data/test.csv"))
    ap.add_argument("--output", type=Path, default=Path("submission-dnn.csv"))
    args = ap.parse_args()
    main(args)
