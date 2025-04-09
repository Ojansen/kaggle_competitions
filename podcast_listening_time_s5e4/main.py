# %%
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor
import keras
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
pd.options.mode.copy_on_write = True
# %%
df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

target = "Listening_Time_minutes"
numerical_features = [
    "Episode_Length_minutes",
    "Host_Popularity_percentage",
    "Guest_Popularity_percentage",
    "Number_of_Ads",
]
categorial_features = [
    "Genre",
    "Publication_Day",
    "Publication_Time",
    "Episode_Sentiment",
]

X = df_train[numerical_features + categorial_features]
y = df_train[target]
X_test = df_test[numerical_features]

X = X.fillna(0)

# %%
def data_preprocessing(df: pd.DataFrame, is_train=True) -> pd.DataFrame:
    df["Number_of_Ads"] = df["Number_of_Ads"].clip(lower=0, upper=6)
    df["Episode_Length_minutes"] = df["Episode_Length_minutes"].clip(lower=0, upper=120)

    # One-hot encode 'Publication_Day' and 'Publication_Time'
    # day_encoded = pd.get_dummies(df["Publication_Day"], prefix="Day", dtype="float")
    # time_encoded = pd.get_dummies(df["Publication_Time"], prefix="Time", dtype="float")
    # df = pd.concat([df, day_encoded, time_encoded], axis=1)
    df = df.drop(columns=["Publication_Day", "Publication_Time"])

    # Ordinal encode the Episode_Sentiment column
    encoder = OrdinalEncoder()
    df["Episode_Sentiment"] = encoder.fit_transform(df[["Episode_Sentiment"]])

    # One-hot encode the 'Genre' column
    df = pd.get_dummies(df, columns=["Genre"], prefix="Genre", dtype="float")

    scaler = MinMaxScaler()
    df[["Host_Popularity_percentage", "Guest_Popularity_percentage"]] = (
        scaler.fit_transform(
            df[["Host_Popularity_percentage", "Guest_Popularity_percentage"]]
        )
    )
    return df


# %%
X = data_preprocessing(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
def create_submission(model):
    model.fit(X, y)
    predictions = model.predict(X_test)

    # Check for missing rows in predictions
    if len(predictions) < len(df_test):
        missing_count = len(df_test) - len(predictions)
        mean_prediction = predictions.mean()
        predictions = np.append(predictions, [mean_prediction] * missing_count)

    # Create the submission DataFrame
    submission = pd.DataFrame(
        {"id": df_test["id"], "Listening_Time_minutes": predictions}
    )
    submission.to_csv("submission.csv", index=False)
    print("Submission file created: submission.csv")


# %%
def dnn_model():
    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=f"logs/tf_sgd_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        ),
    ]

    normalizer = keras.layers.Normalization()
    normalizer.adapt(np.array(X_train))

    model = keras.Sequential(
        [
            normalizer,
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dropout(0.4),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_absolute_error",
        metrics=["root_mean_squared_error"],
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=1024,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    metrics = model.evaluate(X_val, y_val, batch_size=256)

    print("Metrics:", metrics)

    hist_df = pd.DataFrame(history.history)
    hist_df.plot()
    plt.show()


# %%
if __name__ == "__main__":
    dnn_model()
    # randomforest = sklearn.ensemble.RandomForestRegressor(
    #     n_estimators=100,
    #     max_depth=10,
    #     min_samples_split=2,
    #     min_samples_leaf=1,
    #     max_features="sqrt",
    #     random_state=42,
    # )
    # randomforest.fit(X_train, y_train)
    # predictions = randomforest.predict(X_test)
    # print(
    #     f"Random Forest RMSE: {root_mean_squared_error(y_val, randomforest.predict(X_val)):.2f}"
    # )

    # submission = pd.DataFrame(
    #     {"id": df_test["id"], "Listening_Time_minutes": predictions}
    # )
    # submission.to_csv("submission.csv", index=False)
    # create_submission(randomforest)

# %%
