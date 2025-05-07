import pandas as pd
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

test_df = pd.read_csv("data/test.csv")
train_df = pd.read_csv("data/train.csv")

target = "Calories"
features = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

X = train_df[features]
y = train_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

model = DecisionTreeRegressor(random_state=1, max_depth=5)

model.fit(X_train, y_train)

rmsle = root_mean_squared_log_error(y_test, model.predict(X_test))
print("The root mean squared log error (RMSLE) on test set: {:.4f}".format(rmsle))


def create_submission(model, test_df):
    predictions = model.predict(
        test_df[features]
    )  # Use the pipeline to handle preprocessing
    submission = pd.DataFrame({"id": test_df["id"], "Calories": predictions})
    submission.to_csv("submission.csv", index=False)
    print("Submission file created: submission.csv")


create_submission(model, test_df)
