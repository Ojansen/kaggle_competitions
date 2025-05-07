# %%
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
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
categorical_features = [
    "Genre",
    "Publication_Day",
    "Publication_Time",
    "Episode_Sentiment",
]

X = df_train[numerical_features + categorical_features]
y = df_train[target]
X_test = df_test[numerical_features + categorical_features]

X = X.fillna(0)


# Define preprocessing for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Define the pipeline
xgb_model = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)

pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", xgb_model),
    ]
)

# Define the parameter grid for GridSearchCV
param_grid = {
    "regressor__n_estimators": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    "regressor__max_depth": [3, 5, 7, 10],
    "regressor__learning_rate": [0.01, 0.1, 0.001],
    "regressor__subsample": [0.8, 1.0, 0.6, 0.4],
}

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform GridSearchCV
grid_search = RandomizedSearchCV(
    pipe,
    param_grid,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring="neg_root_mean_squared_error",
    verbose=1,
    n_iter=10,
    n_jobs=-1,
)

# Fit the model
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best RMSE:", -grid_search.best_score_)

# Make predictions on the validation set
y_pred = grid_search.best_estimator_.predict(X_val)

# Evaluate the model
rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
print(f"Validation RMSE: {rmse:.2f}")


# Create submission
def create_submission(model, X_test, df_test):
    predictions = model.predict(X_test)  # Use the pipeline to handle preprocessing
    submission = pd.DataFrame(
        {"id": df_test["id"], "Listening_Time_minutes": predictions}
    )
    submission.to_csv("submission.csv", index=False)
    print("Submission file created: submission.csv")


# Predict on test data using the pipeline
create_submission(grid_search.best_estimator_, X_test, df_test)
