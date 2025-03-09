import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

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
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Histogram with KDE using log-transformed Fare
sns.histplot(data=np.log1p(train['Fare']), bins=50, kde=True, ax=ax1)
ax1.set_title('Distribution of Log-transformed Fare')
ax1.set_xlabel('Log(Fare + 1)')
ax1.set_ylabel('Count')

# Box plot of log-transformed Fare
sns.boxplot(data=np.log1p(train['Fare']), ax=ax2)
ax2.set_title('Box Plot of Log-transformed Fare')
ax2.set_xlabel('Log(Fare + 1)')

# Adjust layout
plt.tight_layout()
plt.show()

# Print basic statistics for both original and log-transformed Fare
print("\nOriginal Fare Statistics:")
print(train['Fare'].describe())
print("\nLog-transformed Fare Statistics:")
print(np.log1p(train['Fare']).describe())

# Select numerical columns
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']

# Create subplots
fig, axes = plt.subplots(len(numerical_cols), 2, figsize=(15, 4*len(numerical_cols)))

for idx, col in enumerate(numerical_cols):
    # Histogram with KDE
    sns.histplot(data=train[col].dropna(), bins=30, kde=True, ax=axes[idx, 0])
    axes[idx, 0].set_title(f'Distribution of {col}')
    axes[idx, 0].set_xlabel(col)
    
    # Box plot
    sns.boxplot(data=train[col].dropna(), ax=axes[idx, 1])
    axes[idx, 1].set_title(f'Box Plot of {col}')
    axes[idx, 1].set_xlabel(col)

plt.tight_layout()
plt.show()

# Print skewness statistics
print("\nSkewness Statistics:")
for col in numerical_cols:
    skew = train[col].skew()
    print(f"{col}: {skew:.3f}")
