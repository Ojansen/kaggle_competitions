import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split

df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

print(df_train.describe())

X_train = df_train.drop(['PassengerId', 'Name', 'Transported', 'Destination', 'Cabin', 'HomePlanet'], axis=1).copy()
y_train = df_train['Transported']

X_train[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = X_train[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
X_train['Age'] = X_train['Age'].fillna(X_train['Age'].mean())
X_train['RoomService'] = X_train['RoomService'].fillna(X_train['RoomService'].mean())

X_train['VIP'] = X_train['VIP'].astype(np.float64)
X_train['CryoSleep'] = X_train['CryoSleep'].astype(np.float64)

print(X_train.shape)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

normalizer = keras.layers.Normalization()
normalizer.adapt(np.array(X_train))

model = keras.Sequential([
    normalizer,
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, y_test)
)

metrics = model.evaluate(X_test, y_test, batch_size=128)

print("Metrics:", metrics)

hist_df = pd.DataFrame(history.history)
hist_df.plot()
plt.show()

def create_submission():
    X_train = df_test.drop(['PassengerId', 'Name', 'Destination', 'Cabin', 'HomePlanet'], axis=1).copy()

    X_train[['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = X_train[
        ['VIP', 'CryoSleep', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
    X_train['Age'] = X_train['Age'].fillna(X_train['Age'].mean())
    X_train['RoomService'] = X_train['RoomService'].fillna(X_train['RoomService'].mean())

    X_train['VIP'] = X_train['VIP'].astype(np.float64)
    X_train['CryoSleep'] = X_train['CryoSleep'].astype(np.float64)

    predictions = model.predict(X_train)
    n_predictions = (predictions > 0.5).astype(bool)
    sample_submission_df = pd.read_csv('data/sample_submission.csv')
    sample_submission_df['Transported'] = n_predictions
    sample_submission_df.to_csv('submission.csv', index=False)
    sample_submission_df.head()


# create_submission()
