import pandas as pd

# Read the submission file
df = pd.read_csv('submission_tf.csv')

# Check for null values
null_check = df.isnull().sum()
print("Null values in each column:")
print(null_check)
print(df["rainfall"].mean())
# #
# df["rainfall"] = df["rainfall"].fillna().mean()
# df.to_csv('submission_tf.csv', index=False)
