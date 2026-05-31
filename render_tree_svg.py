import os
import sys
import joblib
from sklearn import tree
from sklearn.tree import export_text

MODEL_PATH = os.path.join("output", "model.joblib")
OUT_SVG = os.path.join("output", "tree.svg")
OUT_TXT = os.path.join("output", "tree.txt")

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

# Write textual dump
try:
    txt = export_text(estimator, feature_names=list(feature_names), max_depth=5)
    with open(OUT_TXT, "w", encoding="utf8") as f:
        f.write(txt)
    print(f"Saved textual tree dump to {OUT_TXT}")
except Exception as e:
    print(f"Failed to write textual dump: {e}")

# Write SVG using export_graphviz -> dot -> pydot if available, else fallback to using sklearn's plot_tree saved as SVG
try:
    try:
        import pydot
        from sklearn.tree import export_graphviz

        dot_data = export_graphviz(estimator, out_file=None, feature_names=feature_names, filled=True, rounded=True, special_characters=True, max_depth=6)
        (graphs,) = pydot.graph_from_dot_data(dot_data)
        graphs.write_svg(OUT_SVG)
        print(f"Saved SVG tree to {OUT_SVG} using pydot")
    except Exception:
        # fallback: render via matplotlib and save as SVG
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(24, 16))
        ax = fig.add_subplot(111)
        tree.plot_tree(estimator, feature_names=feature_names, filled=True, impurity=False, proportion=True, ax=ax, max_depth=6)
        plt.tight_layout()
        fig.savefig(OUT_SVG, format="svg", bbox_inches='tight')
        plt.close(fig)
        print(f"Saved SVG tree to {OUT_SVG} using matplotlib fallback")
except Exception as e:
    print(f"Failed to render SVG: {e}")
    sys.exit(4)
