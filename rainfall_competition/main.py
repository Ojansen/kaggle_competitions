import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Load data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Define feature columns
feature_columns = [
    "day",
    "mintemp",
    "pressure",
    "sunshine",
    "winddirection",
    "windspeed",
    "maxtemp",
    "dewpoint",
    "temparature",
]

# Prepare data
X = train_df[feature_columns].copy()
y = train_df["rainfall"].copy()

# Prepare test data
X_test = test_df[feature_columns].copy()

# Fill NA values in test data
for column in feature_columns:
    if X_test[column].isna().any():
        X_test[column] = X_test[column].fillna(X_test[column].mean())

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


def xgb():
    best_grid = {
        "learning_rate": 0.01,
        "max_depth": 3,
        "min_child_weight": 1,
        "n_estimators": 400,
        "subsample": 0.4,
    }
    model = XGBRegressor(**best_grid)
    model.fit(X_train, y_train)
    print(f"XGB ROC AUC: {roc_auc_score(y_val, model.predict(X_val)) * 100:.2f}%")


def random_forest():
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"RFR ROC AUC: {roc_auc_score(y_val, model.predict(X_val)) * 100:.2f}%")


def stacking():
    estimators = [
        ("ridge", RidgeCV()),
        ("lasso", LassoCV(random_state=42)),
        ("knr", KNeighborsRegressor(n_neighbors=20, metric="euclidean")),
    ]

    final_estimator = GradientBoostingRegressor(
        n_estimators=25,
        subsample=0.5,
        min_samples_leaf=25,
        max_features=1,
        random_state=42,
    )
    reg = StackingRegressor(estimators=estimators, final_estimator=final_estimator)

    reg.fit(X_train, y_train)
    print(f"Stacking ROC AUC: {roc_auc_score(y_val, reg.predict(X_val)) * 100:.2f}%")


def svm_pipeline():
    # Create and fit pipeline
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                SVR(C=0.01, epsilon=0.1, gamma=0.01, max_iter=10_000, kernel="rbf"),
            ),
        ]
    )

    # Fit the model
    pipe.fit(X_train, y_train)

    # Make predictions
    print(f"SVM ROC AUC: {roc_auc_score(y_val, pipe.predict(X_val)) * 100:.2f}%")
    return pipe.predict(X_val)


def create_submission(model):
    model.fit(X, y)
    predictions = model.predict(X_test)
    output = pd.DataFrame({"id": test_df["id"], "rainfall": predictions})
    output.to_csv("submission.csv", index=False)
    print("Your submission was successfully saved!")


if __name__ == "__main__":
    xgb()
    stacking()
    svm_pipeline()
    random_forest()
