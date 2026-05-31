import os
import sys
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

MODEL_PATH = os.path.join("output", "model.joblib")
OUT_PNG = os.path.join("output", "tree.png")

if not os.path.exists(MODEL_PATH):
    print(f"Model not found: {MODEL_PATH}")
    sys.exit(2)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(3)

# choose first estimator if RandomForest
if hasattr(model, "estimators_") and len(getattr(model, "estimators_", [])) > 0:
    estimator = model.estimators_[0]
else:
    estimator = model

feature_names = getattr(model, "feature_names_in_", None)
if feature_names is None:
    n_in = getattr(estimator, "n_features_in_", None) or 10
    feature_names = [f"f{i}" for i in range(n_in)]

plt.figure(figsize=(18, 12))
ax = plt.gca()
plot_tree(estimator, feature_names=feature_names, filled=True, impurity=False, proportion=True, ax=ax, max_depth=5)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved decision tree image to {OUT_PNG}")
