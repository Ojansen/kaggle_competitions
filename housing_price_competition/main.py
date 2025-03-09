# Import helpful libraries
import graphviz
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import export_graphviz

# Set up filepaths
import os

if not os.path.exists("data/train.csv"):
    os.symlink("data/train.csv", "train.csv")
    os.symlink("data/test.csv", "test.csv")


def model():
    param_grid = {
        "n_estimators": [
            100,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
        ],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
    }

    base_model = XGBRegressor(objective="reg:squarederror", random_state=42)

    # Using RandomizedSearchCV as it's faster than GridSearchCV
    # and often produces similar results
    search = RandomizedSearchCV(
        base_model,
        param_grid,
        n_iter=25,  # number of parameter settings sampled
        scoring="neg_mean_absolute_error",
        cv=5,
        verbose=1,
        n_jobs=-1,  # use all CPU cores
    )

    # Fit the model
    search.fit(train_X, train_y)

    # Print best parameters and score
    print("Best parameters:", search.best_params_)
    print("Best MAE score: {:.0f}".format(-search.best_score_))

    # Return the best model
    return search.best_estimator_


# Load the data, and separate the target
iowa_file_path = "data/train.csv"
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice

# Create X (After completing the exercise, you can return to modify this line!)
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
X = home_data[features].fillna(0)
X.head()

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


def analyze_feature_importance(model, X):
    # Get feature importance
    importance = model.feature_importances_

    # Create DataFrame with features and their importance scores
    feature_importance = pd.DataFrame({"feature": X.columns, "importance": importance})

    # Sort by importance
    feature_importance = feature_importance.sort_values("importance", ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=feature_importance.head(20))
    plt.title("Top 20 Most Important Features")
    plt.xlabel("Feature Importance Score")
    plt.tight_layout()
    plt.show()

    # Print feature importance
    print("\nFeature Importance Scores:")
    print(feature_importance)

    return feature_importance


def get_mae():
    # Get the optimized model
    rf_model = model()

    # Get predictions
    rf_val_predictions = rf_model.predict(val_X)
    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

    print("Validation MAE for Model: {:,.0f}".format(rf_val_mae))

    # Analyze feature importance
    feature_importance = analyze_feature_importance(rf_model, X)

    # Optional: Print correlation with target variable
    print("\nCorrelation with SalePrice:")
    correlations = X.apply(lambda x: x.corr(y) if x.dtype.kind in "biufc" else 0)
    print(correlations.sort_values(ascending=False))


def visualize():
    sns.histplot(train_y, kde=True)
    plt.title("SalePrice Distribution")
    plt.show()


def create_submission():
    # To improve accuracy, create a new Random Forest model which you will train on all training data
    rf_model_on_full_data = model()

    # fit rf_model_on_full_data on all data from the training data
    rf_model_on_full_data.fit(X, y)

    # path to file you will use for predictions
    test_data_path = "data/test.csv"

    # read test data file using pandas
    test_data = pd.read_csv(test_data_path)

    # create test_X which comes from test_data but includes only the columns you used for prediction.
    # The list of columns is stored in a variable called features
    test_X = test_data[features]

    # make predictions which we will submit.
    test_preds = rf_model_on_full_data.predict(test_X)

    output = pd.DataFrame({"Id": test_data.Id, "SalePrice": test_preds})
    output.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    # print(pd.get_dummies(home_data.OverallQual, dtype=float))
    # print(pd.get_dummies(X['ExterCond'][0]))
    # visualize()
    # get_mae()
    create_submission()
