"""
Steps:
  1. Load station2_labelled.csv
  2. Split 80/20 stratified
  3. Balance training set with SMOTE
  4. Train XGBoost
  5. Evaluate on held-out test set
  6. Save model + supporting files as .pkl

Run: python train_pipeline.py
=============================================================
"""
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')



# CONFIG
# =============================================================

MODEL_DIR   = Path(__file__).parent
DATA_PATH   = Path(MODEL_DIR).parent / '..' / 'data'/ 'processed' / 'station2_labelled.csv'

FEATURES    = ['DO', 'PH', 'AMMONIA(mg/l)', 'TEMP', 'NITRATE(PPM)', 'TURBIDITY']
TARGET      = 'feed_label'
CLASS_NAMES = ['Prime Feed', 'Reduce Feed', 'Halt Feeding']


# LOAD
# =============================================================

df = pd.read_csv(DATA_PATH)

X = df[FEATURES]
y = df[TARGET]

print(f"Loaded: {len(df):,} rows  |  {len(FEATURES)} features")
for cls, name in enumerate(CLASS_NAMES):
    n = (y == cls).sum()
    print(f"  Class {cls} ({name}): {n:,}  ({n/len(y)*100:.1f}%)")


# SPLIT
# =============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.20,
    random_state = 42,
    stratify     = y        # keeps class proportions equal in both sets
)

print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")


# SMOTE (training set only)
# =============================================================

smote = SMOTE(random_state=42)
X_bal, y_bal = smote.fit_resample(X_train, y_train)

print(f"After SMOTE: {len(X_bal):,} training rows  (was {len(X_train):,})")


# TRAIN
# =============================================================

model = XGBClassifier(
    n_estimators     = 300,
    max_depth        = 6,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 5,
    eval_metric      = 'mlogloss',
    random_state     = 42,
    n_jobs           = -1
)

model.fit(X_bal, y_bal)
print("\nModel trained.")


# EVALUATE
# =============================================================

y_pred   = model.predict(X_test)
train_f1 = f1_score(y_bal,  model.predict(X_bal),  average='macro')
test_f1  = f1_score(y_test, y_pred,                average='macro')
gap      = train_f1 - test_f1

print(f"\nTrain F1 Macro : {train_f1:.4f}")
print(f"Test  F1 Macro : {test_f1:.4f}")
print(f"Gap            : {gap:.4f}  {'✅ No overfit' if gap < 0.01 else '⚠️ Check overfit'}")
print()
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))


# SAVE
# =============================================================

Path(MODEL_DIR).mkdir(exist_ok=True)

joblib.dump(model,       f'{MODEL_DIR}/feed_classifier.pkl')
joblib.dump(FEATURES,    f'{MODEL_DIR}/feature_list.pkl')
joblib.dump(CLASS_NAMES, f'{MODEL_DIR}/class_names.pkl')

print("Saved:")
print(f"  {MODEL_DIR}/feed_classifier.pkl  — trained XGBoost model")
print(f"  {MODEL_DIR}/feature_list.pkl     — feature order for the app")
print(f"  {MODEL_DIR}/class_names.pkl      — label names for the app")
