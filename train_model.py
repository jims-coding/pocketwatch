import os
import io
import numpy as np
from rasterio.transform import Affine
from pyproj import Transformer
import geopandas as gpd

# scikit-learn imports will be used if available
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
    import joblib
except Exception:
    RandomForestClassifier = None


OUTPUT_DIR = "output"
NPZ_PATH = os.path.join(OUTPUT_DIR, "dataset_package.npz")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.joblib")
TRAIN_FRACTION = 0.5


def load_package(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    return data


def find_occurrence_key(data):
    # Try to find a layer that likely contains mineral occurrences
    for key in data.files:
        if key.startswith("vector__"):
            lname = key.replace("vector__", "").replace("__", ":")
            if "mineral" in lname.lower() or "occ" in lname.lower() or "minocc" in lname.lower() or "mo" in lname.lower():
                return key
    # fallback to first vector layer
    for key in data.files:
        if key.startswith("vector__"):
            return key
    return None


def prepare_training_data(data):
    # Load raster
    if "raster" not in data:
        raise RuntimeError("No raster in package")
    raster = data["raster"]
    transform_arr = data.get("raster_transform", np.array([]))
    raster_crs = data.get("raster_crs", np.array([None]))
    # raster_crs may be stored as a numpy array containing a string, or as a scalar string
    if hasattr(raster_crs, "tolist"):
        rc_val = raster_crs.tolist()
        if isinstance(rc_val, (list, tuple, np.ndarray)):
            raster_crs = rc_val[0] if len(rc_val) > 0 else None
        else:
            raster_crs = rc_val
    else:
        raster_crs = raster_crs

    if raster is None:
        raise RuntimeError("Raster is empty")

    height, width = raster.shape

    # Build pixel grid features across the full AOI
    rows, cols = np.indices(raster.shape)
    values = raster.flatten()
    valid_mask = ~np.isnan(values)
    if valid_mask.sum() == 0:
        raise RuntimeError("Raster contains only NaN")

    rows_f = rows.flatten()[valid_mask]
    cols_f = cols.flatten()[valid_mask]
    vals_f = values[valid_mask]

    # Prepare labels by mapping occurrence points to pixel indices
    occ_key = find_occurrence_key(data)
    if occ_key is None:
        raise RuntimeError("No vector layers available for labels")

    geojson_str = data[occ_key].tolist() if hasattr(data[occ_key], 'tolist') else data[occ_key]
    gdf = gpd.read_file(io.StringIO(geojson_str))
    if gdf.empty:
        raise RuntimeError("Occurrence GeoDataFrame is empty")

    # Ensure geometries are points
    gdf = gdf[gdf.geometry.notnull()]
    if gdf.empty:
        raise RuntimeError("No valid geometries in occurrences")

    # Transform occurrence coords to raster CRS if needed
    occ_coords = [(pt.x, pt.y) for pt in gdf.geometry]

    if raster_crs is None:
        raster_crs = "EPSG:4326"

    if isinstance(raster_crs, np.ndarray) or isinstance(raster_crs, list):
        raster_crs = str(raster_crs[0]) if len(raster_crs) > 0 else "EPSG:4326"

    if raster_crs.upper().startswith("EPSG"):
        src_crs = "EPSG:4326"
        dst_crs = raster_crs
    else:
        src_crs = "EPSG:4326"
        dst_crs = raster_crs

    if dst_crs is None:
        dst_crs = "EPSG:4326"

    if dst_crs != src_crs:
        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        occ_coords_trans = [transformer.transform(x, y) for (x, y) in occ_coords]
    else:
        occ_coords_trans = occ_coords

    # Build inverse affine
    if transform_arr is None or len(transform_arr) == 0:
        raise RuntimeError("No raster transform available")
    affine = Affine(*list(map(float, transform_arr)))
    inv_affine = ~affine

    labels = np.zeros(valid_mask.sum(), dtype=int)
    # For each occurrence, map to pixel and mark nearby pixels positive
    radius_px = 1
    for (x, y) in occ_coords_trans:
        try:
            col_f, row_f = inv_affine * (x, y)
        except Exception:
            continue
        col_i = int(round(col_f))
        row_i = int(round(row_f))
        for dr in range(-radius_px, radius_px + 1):
            for dc in range(-radius_px, radius_px + 1):
                r = row_i + dr
                c = col_i + dc
                if 0 <= r < height and 0 <= c < width:
                    # compute flattened index among masked pixels
                    flat_idx = r * width + c
                    # Need to find position in masked array
                    # Count number of valid (mask True) up to flat_idx
                    # We'll create a mapping from flat index to masked index
                    pass

    # Create mapping from flat index to masked position
    flat_indices = np.nonzero(valid_mask)[0]
    flat_to_mask_idx = {int(f): i for i, f in enumerate(flat_indices)}

    # Re-run to set labels using mapping
    for (x, y) in occ_coords_trans:
        try:
            col_f, row_f = inv_affine * (x, y)
        except Exception:
            continue
        col_i = int(round(col_f))
        row_i = int(round(row_f))
        for dr in range(-radius_px, radius_px + 1):
            for dc in range(-radius_px, radius_px + 1):
                r = row_i + dr
                c = col_i + dc
                if 0 <= r < height and 0 <= c < width:
                    flat_idx = r * width + c
                    if flat_idx in flat_to_mask_idx:
                        midx = flat_to_mask_idx[flat_idx]
                        labels[midx] = 1

    # Ensure we have some positive labels
    if labels.sum() == 0:
        print("Warning: No positive labels found from occurrences; model training will be trivial.")

    X_full = vals_f.reshape(-1, 1)

    rng = np.random.default_rng(42)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) > 0 and len(neg_idx) > 0:
        neg_target = min(len(neg_idx), len(pos_idx) * 3)
        neg_idx = rng.choice(neg_idx, size=neg_target, replace=False)
        keep_idx = np.concatenate([pos_idx, neg_idx])
        rng.shuffle(keep_idx)
    else:
        keep_idx = np.arange(len(labels))

    if TRAIN_FRACTION < 1.0 and len(keep_idx) > 0:
        target_size = max(1, int(round(len(keep_idx) * TRAIN_FRACTION)))
        if target_size < len(keep_idx):
            keep_idx = rng.choice(keep_idx, size=target_size, replace=False)
            rng.shuffle(keep_idx)

    X = X_full[keep_idx]
    y = labels[keep_idx]

    return X, y


def main():
    data = load_package(NPZ_PATH)

    if RandomForestClassifier is None:
        raise RuntimeError("scikit-learn is not installed in the environment. Install: pip install scikit-learn")

    X, y = prepare_training_data(data)
    print(f"Prepared training data (50% subsample, quartz-only): X={X.shape}, y={y.shape}, positives={y.sum()}")

    # Simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if y.sum()>0 else None)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"Held-out ROC AUC: {roc_auc_score(y_test, y_score):.6f}")
    print(f"Held-out AUC-PR: {average_precision_score(y_test, y_score):.6f}")

    joblib.dump(clf, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == '__main__':
    main()
