import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.src.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard

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
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Potassium",
    "Phosphorous"
]

# Select columns corresponding to features, and preview the data
X = train_df[features]
y = train_df["Fertilizer Name"]

y_encoded = pd.get_dummies(y)

# Convert X to float32
X = X.astype(np.float32)

# Select columns corresponding to features, and preview the data
X_test = test_df[features]

# Ensure X_test is also of a float32 type
X_test = X_test.astype(np.float32)

X_train, x_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)


normalizer = keras.layers.Normalization()
normalizer.adapt(np.array(X, dtype=np.float32))

model = keras.Sequential(
    [
        normalizer,
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(3),
    ]
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

early_stopping = EarlyStopping(
    patience=20,
    restore_best_weights=True,
)

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    verbose=1,
    batch_size=1024,
    # callbacks=[early_stopping, TensorBoard(log_dir="logs")],
    validation_data=(x_val, y_val),
)

metrics = model.evaluate(x_val, y_val, batch_size=128)

print("Metrics:", metrics)

history_df = pd.DataFrame(history.history)
history_df[["loss", "val_loss"]].plot()
print("MAE:", metrics[0])
plt.show()

predictions = model.predict(X_test, batch_size=128)

# --- Prepare Submission ---
submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': predictions
})
submission.to_csv('submission_tf.csv', index=False)
print("Submission file created: submission_tf.csv")