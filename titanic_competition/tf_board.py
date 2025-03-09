from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc, mean_squared_error, roc_curve
from sklearn.model_selection import (RandomizedSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler
from keras import callbacks, layers, models

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

X_test = pd.get_dummies(test[features])

X = pd.get_dummies(train[features])
y = train['Survived']

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X)

X = pd.DataFrame(imp.transform(X), columns=X.columns)
X_test = pd.DataFrame(imp.transform(X_test), columns=X_test.columns)

X_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=1)

# Prepare data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled = scaler.transform(X_train)
x_val_scaled = scaler.transform(x_val)

def create_tf_model():
    # Add L2 regularization
    regularizer = tf.keras.regularizers.l2(0.01)
    
    model = models.Sequential([
        # Input layer with batch normalization
        layers.Dense(32, kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),  # Increased dropout
        
        # Hidden layer 1
        layers.Dense(24, kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        
        # Hidden layer 2
        layers.Dense(16, kernel_regularizer=regularizer),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.001,  # Reduced learning rate
        momentum=0.9,
        nesterov=True
    )

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    return model


def train_tf_model():
    # Convert to tensorflow format
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train_scaled, y_train.values)
    ).shuffle(1000).batch(32)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (x_val_scaled, y_val.values)
    ).batch(32)

    # Create model
    model = create_tf_model()

    # Setup TensorBoard callback
    log_dir = f"logs/tf_sgd_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_callback = callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq='epoch',
        profile_batch=2,
        embeddings_freq=1
    )

    # Early stopping
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        min_delta=0.001
    )

    # Learning rate scheduler
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    # Model checkpoint
    checkpoint = callbacks.ModelCheckpoint(
        f"{log_dir}/best_model.keras",
        monitor='val_auc',
        save_best_only=True,
        mode='max'
    )

    # Train model with class weights to handle imbalance
    class_weights = {
        0: 1.0,
        1: (y_train == 0).sum() / (y_train == 1).sum()  # Adjust weights based on class distribution
    }

    history = model.fit(
        train_dataset,
        epochs=1000,
        validation_data=val_dataset,
        callbacks=[
            tensorboard_callback,
            early_stopping,
            reduce_lr,
            checkpoint
        ],
        class_weight=class_weights,  # Add class weights
        verbose=1
    )

    print(f"TensorBoard logs saved to: {log_dir}")
    print("To view TensorBoard, run:")
    print(f"tensorboard --logdir={log_dir}")

    return model, history


def evaluate_tf_model(model, history):
    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('Model AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Calculate and plot ROC curve
    y_pred = model.predict(x_val_scaled)
    fpr, tpr, _ = roc_curve(y_val, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def create_submission_tf(model, scaler):
    # Prepare test data
    X_test_scaled = scaler.transform(X_test)

    # Make predictions
    test_pred = (model.predict(X_test_scaled) > 0.5).astype(int)

    # Create submission
    output = pd.DataFrame({
        'PassengerId': test.PassengerId,
        'Survived': test_pred.flatten()
    })
    output.to_csv('submission_tf.csv', index=False)
    print("Your submission was successfully saved!")


if __name__ == '__main__':
    # Train model
    model, history = train_tf_model()

    # Evaluate model
    # evaluate_tf_model(model, history)

    # Create submission
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    create_submission_tf(model, scaler)