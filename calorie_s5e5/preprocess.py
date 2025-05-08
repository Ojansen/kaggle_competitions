import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

test_df = pd.read_csv("data/test.csv")
train_df = pd.read_csv("data/train.csv")

target = "Calories"
features = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

X = train_df[features]
y = train_df[target]

transformed = np.exp(X['Body_Temp'])

print(pd.Series(transformed).skew())
sns.histplot(transformed, kde=True)
plt.show()
