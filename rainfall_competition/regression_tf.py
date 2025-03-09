import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.src.callbacks import EarlyStopping
import keras

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


df = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

feature_columns = [
    "mintemp",
    "pressure",
    "sunshine",
    "winddirection",
    "windspeed",
    "maxtemp",
    "dewpoint",
    "temparature",
    "rainfall",
]

train_features = df[feature_columns].copy()

train_labels = train_features.pop("rainfall")

normalizer = keras.layers.Normalization()
normalizer.adapt(np.array(train_features))

early_stopping = EarlyStopping(
    min_delta=0.001, # minimum amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

model = keras.Sequential(
    [
        normalizer,
        keras.layers.Dense(24, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mae',
    metrics=['auc'],
)

model_options = {
    "epochs": 400,
    "batch_size": 32,
    "verbose": 1,
    "validation_split": 0.2,
    "callbacks": [early_stopping],
}

history = model.fit(
    train_features,
    train_labels,
    **model_options,
)

history_df = pd.DataFrame(history.history)
history_df.plot()
print("Max roc auc loss: {:0.4f}".format(history_df['val_auc'].max()))
plt.show()


# Plot training history
def plot_metrics(history):
    metrics = ['loss', 'auc']
    plt.figure(figsize=(12, 4))

    for n, metric in enumerate(['loss', 'auc']):
        plt.subplot(1, 2, n + 1)
        plt.plot(history.history[metric])
        plt.plot(history.history[f'val_{metric}'])
        plt.title(f'Model {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend(['Train', 'Validation'])

    plt.tight_layout()
    plt.show()

print(model.summary())
# plot_metrics(history)

def create_submission():
    X = df.drop('rainfall', axis=1).copy()
    y = df["rainfall"].copy()

    model.fit(
        X,
        y,
        **model_options
    )

    # Make predictions on test data
    predictions = model.predict(df_test)

    # Create submission file
    submission = pd.DataFrame({
        'id': df_test['id'],
        'rainfall': predictions.flatten()
    })
    submission.to_csv('submission_tf.csv', index=False)
    print("Submission file created successfully!")


# create_submission()