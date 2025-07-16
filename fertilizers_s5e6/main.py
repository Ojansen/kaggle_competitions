import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import label_ranking_average_precision_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier

# 1. Load Data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# 2. Encode Categorical Features
cat_features = ['Soil Type', 'Crop Type']
for col in cat_features:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

# 3. Encode Target
target_le = LabelEncoder()
train['Fertilizer Name'] = target_le.fit_transform(train['Fertilizer Name'])

# 4. Features/Target
features = [col for col in train.columns if col not in ['id', 'Fertilizer Name']]
X = train[features]
y = train['Fertilizer Name']
X_test = test[features]

# 5. Validation MAP@3 (optional)
params = {
    'objective': 'multiclass',
    'num_class': len(target_le.classes_),
    'metric': 'None',
    'learning_rate': 0.05,
    'verbosity': -1,
    'seed': 42,
}
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
lgb_train = lgb.Dataset(X_tr, y_tr)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
model = lgb.train(params, lgb_train, num_boost_round=200)
val_probs = model.predict(X_val)
map3 = label_ranking_average_precision_score(
    np.eye(val_probs.shape[1])[y_val], val_probs
)
print(f"Validation MAP: {map3:.4f}")

# --- LightGBM ---
lgbm_model = LGBMClassifier(
    objective='multiclass',
    num_class=len(target_le.classes_),
    learning_rate=0.05,
    n_estimators=100,
    num_leaves=31,
    random_state=42
)
lgbm_model.fit(X, y)
lgb_probs = lgbm_model.predict_proba(X_test)

# --- XGBoost ---
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(target_le.classes_),
    learning_rate=0.05,
    n_estimators=100,
    max_depth=3,
    random_state=42,
    verbosity=0,
    use_label_encoder=False
)
xgb_model.fit(X, y)
xgb_probs = xgb_model.predict_proba(X_test)

# --- Random Forest ---
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X, y)
rf_probs = rf_model.predict_proba(X_test)

# --- Logistic Regression ---
lr_model = LogisticRegression(
    multi_class='multinomial',
    max_iter=200,
    random_state=42
)
lr_model.fit(X, y)
lr_probs = lr_model.predict_proba(X_test)

# --- Extra Trees ---
et_model = ExtraTreesClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
et_model.fit(X, y)
et_probs = et_model.predict_proba(X_test)

# --- Gradient Boosting ---
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    random_state=42
)
gb_model.fit(X, y)
gb_probs = gb_model.predict_proba(X_test)

# --- Ensemble: average probabilities (now with 6 models) ---
ensemble_probs = (
    lgb_probs + xgb_probs + rf_probs + lr_probs + et_probs + gb_probs
) / 6

# --- Get Top 3 Predictions ---
top3 = np.argsort(ensemble_probs, axis=1)[:, -3:][:, ::-1]
top3_flat = top3.flatten()
top3_names_flat = target_le.inverse_transform(top3_flat)
top3_names = top3_names_flat.reshape(top3.shape)
top3_str = [' '.join(row) for row in top3_names]

# --- Prepare Submission ---
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': top3_str
})
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")