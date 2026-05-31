import os
import sys
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

# Import helpers from train_model
import train_model as tm

MODEL_PATH = os.path.join("output", "model.joblib")
NPZ_PATH = os.path.join("output", "dataset_package.npz")

if not os.path.exists(MODEL_PATH):
    print(f"Model not found: {MODEL_PATH}")
    sys.exit(2)

print("Loading data package and preparing training data (this may subsample)...")
data = tm.load_package(NPZ_PATH)
try:
    X, y = tm.prepare_training_data(data)
except Exception as e:
    print(f"Failed to prepare training data: {e}")
    sys.exit(3)

print(f"Prepared data: X={X.shape}, y={y.shape}, positives={y.sum()}")

# Reproduce the same test split used during training
if len(y) == 0:
    print("No labels available; cannot evaluate")
    sys.exit(4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if y.sum()>0 else None)

# Load model
model = joblib.load(MODEL_PATH)

# Predict on the held-out test set
try:
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
except Exception as e:
    print(f"Model prediction failed: {e}")
    # Try predict (no proba)
    y_pred = model.predict(X_test)
    y_score = None

print("Classification report on held-out test split:")
print(classification_report(y_test, y_pred, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

if y_score is not None:
    try:
        roc = roc_auc_score(y_test, y_score)
        ap = average_precision_score(y_test, y_score)
        print(f"Held-out ROC AUC: {roc:.6f}")
        print(f"Held-out AUC-PR: {ap:.6f}")
    except Exception as e:
        print(f"Failed to compute ROC/AUC-PR: {e}")

# Also compute simple global metrics on whole available dataset (for context)
try:
    y_pred_all = model.predict(X)
    print("\nOverall classification report (all prepared pixels):")
    print(classification_report(y, y_pred_all, zero_division=0))
except Exception:
    pass

print("Evaluation complete.")
