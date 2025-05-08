import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import ops
from keras import backend as K
import tensorflow as tf

from sklearn.model_selection import train_test_split

# Enable Metal GPU acceleration
try:
    # Check if Metal is available
    if len(tf.config.list_physical_devices('GPU')) > 0:
        # Configure TensorFlow to use Metal
        tf.config.experimental.set_visible_devices(
            tf.config.list_physical_devices('GPU')[0], 'GPU'
        )
        print("Metal GPU acceleration enabled")
    else:
        print("No GPU devices found")
except:
    print("Could not enable GPU acceleration")

# Print device placement for operations
print("TensorFlow operations will run on:", tf.config.list_physical_devices())



test_df = pd.read_csv("data/test.csv")
train_df = pd.read_csv("data/train.csv")

target = "Calories"
features = ["Sex", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

X = train_df[features]
y = train_df[target]

X_test = test_df[features]

X = pd.get_dummies(X, columns=["Sex"], dtype="float")
X_test = pd.get_dummies(X_test, columns=["Sex"], dtype="float")
# X.loc[:, "Body_Temp"] = np.exp(X["Body_Temp"])

print(X.head())

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=1)

normalizer = keras.layers.Normalization()
normalizer.adapt(np.array(X, dtype=np.float32))

model = keras.models.Sequential(
    [
        normalizer,
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(1),
    ]
)


def rmsle_np(y_true, y_pred):
    """
    Custom RMSLE loss function using NumPy.
    """
    # Add 1 to avoid log(0) and ensure numerical stability
    y_true = np.clip(y_true, a_min=0, a_max=None)
    y_pred = np.clip(y_pred, a_min=0, a_max=None)

    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    # Compute the squared difference
    squared_log_diff = np.square(log_true - log_pred)

    # Compute the mean and take the square root
    rmsle_value = np.sqrt(np.mean(squared_log_diff))
    return rmsle_value


model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.MeanSquaredError(),
    metrics=[
        keras.losses.MeanSquaredLogarithmicError()
    ]
)

model.fit(
    X_train,
    y_train,
    batch_size=1024,
    epochs=10,
    callbacks=[keras.callbacks.TensorBoard()],
    validation_data=[X_val, y_val],
)

# Example usage
y_val_pred = model.predict(X_val).flatten()  # Flatten to ensure correct shape
rmsle = rmsle_np(y_val, y_val_pred)
print("The root mean squared log error (RMSLE) on test set: {:.4f}".format(rmsle))

# rmsle = root_mean_squared_log_error(y_val, model.predict(X_val))
# print("The root mean squared log error (RMSLE) on test set: {:.4f}".format(rmsle))


def create_submission(model, test_df, test_X):
    predictions = model.predict(
        test_X
    ).flatten()  # Use the pipeline to handle preprocessing
    submission = pd.DataFrame({"id": test_df["id"], "Calories": predictions})
    submission.to_csv("submission_dnn.csv", index=False)
    print("Submission file created: submission_dnn.csv")


create_submission(model, test_df, X_test)
