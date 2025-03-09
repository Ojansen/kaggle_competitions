import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.src.callbacks import EarlyStopping
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

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

features = [
    "OverallQual",
    "BsmtFullBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageCars",
    "MoSold",
    "LotFrontage",
    "LotArea",
    "YearBuilt",
    "YearRemodAdd",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "GrLivArea",
    "GarageYrBlt",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
]
# Start feature selection
# ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Select columns corresponding to features, and preview the data
X = train_df[features].fillna(0)
y = train_df["SalePrice"]

X_test = test_df[features].fillna(0)


X_train, x_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print(X.shape)
normalizer = keras.layers.Normalization()
normalizer.adapt(np.array(X, dtype=np.float32))
print(np.array(X).dtype)

model = keras.Sequential(
    [
        normalizer,
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.4),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(1),
    ]
)

model.compile(
    optimizer="rmsprop",
    loss="mae",
    metrics=["root_mean_squared_error"],
)

early_stopping = EarlyStopping(
    patience=20,
    restore_best_weights=True,
)

history = model.fit(
    X_train,
    y_train,
    epochs=1000,
    verbose=1,
    batch_size=64,
    callbacks=[early_stopping],
    validation_data=(x_val, y_val),
)

metrics = model.evaluate(x_val, y_val, batch_size=128)

print("Metrics:", metrics)

# print(model.summary())

history_df = pd.DataFrame(history.history)
history_df[["loss", "val_loss"]].plot()
print("MAE:", metrics[0])
plt.show()

predictions = model.predict(X_test, batch_size=128)

# Create submission DataFrame
submission = pd.DataFrame({
    'Id': test_df.Id,
    'SalePrice': predictions.flatten()  # flatten() converts 2D array to 1D
})

# Save to CSV
submission.to_csv('submission_tf.csv', index=False)
print("Submission saved to submission_tf.csv")
