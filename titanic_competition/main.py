from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.metrics import accuracy_score, auc, mean_squared_error, roc_curve
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

X_test = pd.get_dummies(test[features])

X = pd.get_dummies(train[features])
y = train["Survived"]

imp = SimpleImputer(missing_values=np.nan, strategy="mean")
imp.fit(X)

X = pd.DataFrame(imp.transform(X), columns=X.columns)
X_test = pd.DataFrame(imp.transform(X_test), columns=X_test.columns)


X_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=1)


# model = HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=1, verbose=1)
# model.fit(X_train, y_train)
# val_pred = model.predict(x_val)
# print(accuracy_score(x_val, y_val))
#
def create_submission(model):
    model.fit(X, y)
    test_pred = model.predict(X_test)

    output = pd.DataFrame({"PassengerId": test.PassengerId, "Survived": test_pred})
    output.to_csv("submission.csv", index=False)
    print("Your submission was successfully saved!")


model_params = {"n_estimators": 200}


def mse_loss_visualization():
    model = GradientBoostingClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=5,
        random_state=1,
        verbose=1,
    )
    model.fit(X_train, y_train)

    test_score = np.zeros((model_params["n_estimators"],), dtype=np.float64)
    for i, y_pred in enumerate(model.staged_predict(x_val)):  # Changed X_test to x_val
        test_score[i] = mean_squared_error(y_val, y_pred)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title("Deviance")
    plt.plot(
        np.arange(model_params["n_estimators"]) + 1,
        model.train_score_,
        "b-",
        label="Training Set Deviance",
    )
    plt.plot(
        np.arange(model_params["n_estimators"]) + 1,
        test_score,
        "r-",
        label="Test Set Deviance",
    )
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance")
    fig.tight_layout()
    plt.show()


def sk_pipeline():
    model = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"), RandomForestClassifier(random_state=1)
    )
    model.fit(X_train, y_train)
    val_pred = model.predict(x_val)
    print(accuracy_score(x_val, y_val))


def plot_roc_curve():
    model = GradientBoostingClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=3,
        random_state=1,
        subsample=0.25,
        learning_rate=0.01,
        verbose=1,
    )
    # model = SGDClassifier(random_state=1, verbose=1)

    model.fit(X_train, y_train)

    # Get prediction probabilities
    y_pred_proba = model.predict_proba(x_val)[:, 1]

    # Calculate ROC curve points
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()


def plot_feature_importance():
    # Create and train multiple models to get feature importance distributions
    n_iterations = 100
    importances_matrix = np.zeros((n_iterations, X.shape[1]))

    for i in range(n_iterations):
        # Create new train-test split for each iteration
        X_train_iter, X_val_iter, y_train_iter, y_val_iter = train_test_split(
            X, y, test_size=0.4, random_state=i
        )

        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=i
        )
        model.fit(X_train_iter, y_train_iter)
        importances_matrix[i] = model.feature_importances_

    # Calculate mean importance
    mean_importances = np.mean(importances_matrix, axis=0)
    feature_names = X.columns

    # Sort features by mean importance
    indices = np.argsort(mean_importances)[::-1]
    sorted_names = [feature_names[i] for i in indices]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1.5])

    # Bar plot
    ax1.bar(range(X.shape[1]), mean_importances[indices])
    ax1.set_title("Mean Feature Importances")
    ax1.set_xticks(range(X.shape[1]))
    ax1.set_xticklabels(sorted_names, rotation=45, ha="right")

    # Box plot
    sorted_importances = importances_matrix[:, indices]
    ax2.boxplot(sorted_importances, labels=sorted_names)
    ax2.set_title("Feature Importance Distribution")
    ax2.set_xticklabels(sorted_names, rotation=45, ha="right")
    ax2.set_ylabel("Importance")

    plt.tight_layout()
    plt.show()

    # Print feature importance ranking
    print("\nFeature Importance Ranking:")
    for f in range(X.shape[1]):
        print("%d. %s (%f)" % (f + 1, sorted_names[f], mean_importances[indices[f]]))


class TensorBoardCV:
    def __init__(self, log_dir=None):
        if log_dir is None:
            log_dir = f"logs/sklearn_cv_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.log_dir = log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def log_metrics(self, metrics, fold=0, epoch=0):
        with self.writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(f"fold_{fold}/{name}", value, step=epoch)


def train_with_cv_tensorboard(n_folds=5):
    # Create TensorBoard logger
    tensorboard = TensorBoardCV()

    # Initialize base model
    base_model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )

    # Perform cross-validation
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train model
        model = clone(base_model)
        for epoch in range(model.n_estimators):
            model.fit(X_fold_train, y_fold_train)

            # Calculate metrics
            train_score = model.score(X_fold_train, y_fold_train)
            val_score = model.score(X_fold_val, y_fold_val)

            # ROC AUC
            train_pred = model.predict_proba(X_fold_train)[:, 1]
            val_pred = model.predict_proba(X_fold_val)[:, 1]

            train_fpr, train_tpr, _ = roc_curve(y_fold_train, train_pred)
            val_fpr, val_tpr, _ = roc_curve(y_fold_val, val_pred)

            train_auc = auc(train_fpr, train_tpr)
            val_auc = auc(val_fpr, val_tpr)

            # Log metrics
            metrics = {
                "train_accuracy": train_score,
                "val_accuracy": val_score,
                "train_auc": train_auc,
                "val_auc": val_auc,
            }
            tensorboard.log_metrics(metrics, fold=fold, epoch=epoch)

    print(f"TensorBoard logs saved to: {tensorboard.log_dir}")
    print("To view TensorBoard, run:")
    print(f"tensorboard --logdir={tensorboard.log_dir}")


if __name__ == "__main__":
    train_with_cv_tensorboard()
