import os
import io
import numpy as np
from rasterio.transform import Affine
from pyproj import Transformer
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling

# optional local stats/distance
try:
    from scipy import ndimage as ndi
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# scikit-learn imports will be used if available
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
    import joblib
except Exception:
    RandomForestClassifier = None


OUTPUT_DIR = "output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.joblib")
TRAIN_FRACTION = 0.5
MAX_SAMPLE_DEPTH_M = 1.0
SHALLOW_PROXY_METHODS = {
    "gps observation (wgs-84/gda-94)",
    "gps coordinates",
    "copied point",
    "aerial photograph",
    "orthophotograph (mga)",
}


def _coerce_transform(value):
    if value is None:
        return None
    array_value = np.asarray(value)
    if array_value.size != 6:
        return None
    return tuple(float(x) for x in array_value.reshape(-1).tolist())


def _coerce_crs(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    array_value = np.asarray(value)
    if array_value.size == 0:
        return None
    if array_value.ndim == 0:
        return str(array_value.item())
    first_item = array_value.reshape(-1)[0]
    return str(first_item)


def _coerce_float(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "unknown":
            return None
        try:
            return float(text)
        except Exception:
            return None
    try:
        numeric = float(value)
    except Exception:
        return None
    return numeric if np.isfinite(numeric) else None


def _is_gold_commodity(value):
    text = str(value or "")
    return ("Au" in text) or ("gold" in text.lower())


def _sample_depth_from_row(row):
    depth_keys = (
        "sampleDepth_m",
        "sampleDepth",
        "depth_m",
        "depth",
        "intervalTo_m",
        "to_m",
        "upperDepth_m",
        "boreholeLength_m",
        "drillDepth_m",
        "from_m",
        "intervalFrom_m",
    )
    for key in depth_keys:
        if key not in row:
            continue
        value = _coerce_float(row.get(key))
        if value is not None:
            return value
    return None


def _filter_shallow_gold_rows(gdf, depth_limit_m=MAX_SAMPLE_DEPTH_M):
    if gdf.empty:
        return gdf, False

    commodity_series = gdf["commodity"].astype(str) if "commodity" in gdf.columns else None
    gold_mask = commodity_series.str.contains(r"\bAu\b|gold", case=False, regex=True, na=False) if commodity_series is not None else np.ones(len(gdf), dtype=bool)
    gold_gdf = gdf.loc[gold_mask].copy()
    if gold_gdf.empty:
        return gold_gdf, False

    keep_rows = []
    depth_values = []
    for idx, row in gold_gdf.iterrows():
        depth_m = _sample_depth_from_row(row)
        if depth_m is None:
            depth_values.append(None)
            keep_rows.append(False)
            continue
        depth_values.append(depth_m)
        keep_rows.append(depth_m <= depth_limit_m)

    depth_seen = any(value is not None for value in depth_values)
    if not depth_seen:
        if "observationMethod" not in gold_gdf.columns:
            print("Warning: no depth metadata found in the current gold layer; using gold-bearing sites without a confirmed 1 m cutoff.")
            return gold_gdf.copy(), False

        method_series = gold_gdf["observationMethod"].astype(str).str.strip().str.lower()
        proxy_mask = method_series.isin(SHALLOW_PROXY_METHODS)
        filtered = gold_gdf.loc[proxy_mask].copy()
        if filtered.empty:
            print("Warning: no shallow proxy methods matched; using all gold-bearing sites without a confirmed 1 m cutoff.")
            return gold_gdf.copy(), False
        return filtered, False

    filtered = gold_gdf.loc[keep_rows].copy()
    return filtered, True


def load_package(_unused_path=None):
    # Build a lightweight package-like object from output/rasters and output/vectors.
    rasters_dir = os.path.join(OUTPUT_DIR, "rasters")
    vectors_dir = os.path.join(OUTPUT_DIR, "vectors")
    mapping = {}
    if os.path.isdir(rasters_dir):
        for fn in sorted(os.listdir(rasters_dir)):
            if fn.lower().endswith(".tif") or fn.lower().endswith('.tiff'):
                key = f"raster__{os.path.splitext(fn)[0]}"
                path = os.path.join(rasters_dir, fn)
                try:
                    with rasterio.open(path) as ds:
                        arr = ds.read(1)
                        transform = ds.transform
                        crs = ds.crs
                        mapping[key] = arr
                        mapping[f"{key}__transform"] = np.array([transform.a, transform.b, transform.c, transform.d, transform.e, transform.f])
                        mapping[f"{key}__crs"] = np.array([str(crs)])
                except Exception:
                    continue

    # load vector geojson if present
    if os.path.isdir(vectors_dir):
        for fn in sorted(os.listdir(vectors_dir)):
            if not (fn.lower().endswith('.geojson') or fn.lower().endswith('.json')):
                continue
            key = f"vector__{os.path.splitext(fn)[0]}"
            try:
                with open(os.path.join(vectors_dir, fn), 'r', encoding='utf8') as f:
                    mapping[key] = np.array(f.read(), dtype=object)
            except Exception:
                continue

    class _NPZLike:
        def __init__(self, mp):
            self._mp = mp
            self.files = list(mp.keys())
        def __contains__(self, k):
            return k in self._mp
        def __getitem__(self, k):
            return self._mp[k]
        def get(self, k, default=None):
            return self._mp.get(k, default)

    return _NPZLike(mapping)


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
    # Load base raster (reference grid). Prefer explicit 'raster', fallback to first raster__* available
    if "raster" in data:
        raster = data["raster"]
        transform_arr = data.get("raster_transform", np.array([]))
        raster_crs = data.get("raster_crs", np.array([None]))
    else:
        # find first raster__ key that is an array (not __crs or __transform)
        base_key = None
        for k in data.files:
            if k.startswith("raster__") and not k.endswith("__crs") and not k.endswith("__transform"):
                base_key = k
                break
        if base_key is None:
            raise RuntimeError("No raster in package")
        raster = data[base_key]
        transform_arr = data.get(f"{base_key}__transform", np.array([]))
        raster_crs = data.get(f"{base_key}__crs", np.array([None]))
    # raster_crs may be stored as a numpy array containing a string, or as a scalar string
    raster_crs = _coerce_crs(raster_crs)

    if raster is None:
        raise RuntimeError("Raster is empty")

    height, width = raster.shape

    # Use every predictor raster present in the package.
    # This keeps the pipeline resilient when the package only contains a small subset of coverages.
    desired = []
    for key in data.files:
        if not key.startswith("raster__"):
            continue
        if key.endswith("__crs") or key.endswith("__transform"):
            continue
        if key == "raster":
            continue
        desired.append((key.replace("raster__", ""), key))

    if len(desired) == 0:
        raise RuntimeError("No predictor rasters found in package")

    # helper: resample source array to destination grid
    def resample_to_base(src_arr, src_transform, src_crs, dst_shape, dst_transform, dst_crs):
        dst = np.full(dst_shape, np.nan, dtype=float)
        if src_arr is None:
            return dst
        # construct affine transforms as rasterio expects
        src_transform_affine = _coerce_transform(src_transform)
        dst_transform_affine = _coerce_transform(dst_transform)
        try:
            reproject(
                source=src_arr,
                destination=dst,
                src_transform=Affine(*src_transform_affine) if src_transform_affine is not None else None,
                src_crs=_coerce_crs(src_crs),
                dst_transform=Affine(*dst_transform_affine) if dst_transform_affine is not None else None,
                dst_crs=_coerce_crs(dst_crs),
                resampling=Resampling.bilinear,
                num_threads=2,
            )
        except Exception:
            # fallback: try nearest
            try:
                reproject(
                    source=src_arr,
                    destination=dst,
                    src_transform=Affine(*src_transform_affine) if src_transform_affine is not None else None,
                    src_crs=_coerce_crs(src_crs),
                    dst_transform=Affine(*dst_transform_affine) if dst_transform_affine is not None else None,
                    dst_crs=_coerce_crs(dst_crs),
                    resampling=Resampling.nearest,
                    num_threads=2,
                )
            except Exception:
                pass
        return dst

    # Build pixel grid features across the full AOI
    rows, cols = np.indices(raster.shape)
    values = raster.flatten()
    valid_mask = ~np.isnan(values)
    if valid_mask.sum() == 0:
        raise RuntimeError("Raster contains only NaN")

    # resample each desired raster to base grid and collect arrays
    feature_arrays = []
    feature_names = []
    dst_shape = raster.shape
    dst_transform = transform_arr
    dst_crs = _coerce_crs(raster_crs)
    for name, key in desired:
        try:
            arr = data[key]
            t_key = f"{key}__transform"
            c_key = f"{key}__crs"
            src_transform = data[t_key] if t_key in data.files else None
            src_crs = _coerce_crs(data[c_key]) if c_key in data.files else None
            res = resample_to_base(arr, src_transform, src_crs, dst_shape, dst_transform, dst_crs)
            # convert nodata to nan
            res = res.astype(float)
            feature_arrays.append(res)
            feature_names.append(name)
        except Exception:
            continue

    # stack features into (n_pixels, n_features)
    stacked = np.stack([a.flatten() for a in feature_arrays], axis=1) if len(feature_arrays) > 0 else np.zeros((raster.size, 0))
    # mask valid by base raster
    valid_mask = ~np.isnan(stacked).all(axis=1) & (~np.isnan(values))
    if valid_mask.sum() == 0:
        raise RuntimeError("No valid pixels after stacking features")

    rows_f = rows.flatten()[valid_mask]
    cols_f = cols.flatten()[valid_mask]
    vals_f = values[valid_mask]
    stacked_f = stacked[valid_mask]

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

    gdf, depth_filtered = _filter_shallow_gold_rows(gdf, MAX_SAMPLE_DEPTH_M)
    if gdf.empty:
        if depth_filtered:
            raise RuntimeError("No known gold samples remain after applying the 1 m shallow cutoff")
        raise RuntimeError("The current gold layer does not expose a usable depth field, so shallow gold labels cannot be enforced")

    # Transform occurrence coords to raster CRS if needed
    occ_coords = [(pt.x, pt.y) for pt in gdf.geometry]

    if raster_crs is None:
        raster_crs = "EPSG:4326"
    else:
        raster_crs = _coerce_crs(raster_crs) or "EPSG:4326"

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

    # assemble feature matrix: include stacked predictors + base raster quartz-like value as first column
    if stacked_f.shape[1] > 0:
        X_full = np.hstack([stacked_f])
    else:
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

    prepare_training_data.last_feature_names = list(feature_names)

    return X, y


def main(force=False):
    data = load_package()

    if RandomForestClassifier is None:
        raise RuntimeError("scikit-learn is not installed in the environment. Install: pip install scikit-learn")

    # Ensure output dir exists and optionally skip training when a model exists
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except Exception:
        pass
    if os.path.exists(MODEL_PATH) and not force:
        print(f"Model already exists at {MODEL_PATH}; skipping training.")
        return

    X, y = prepare_training_data(data)
    print(
        f"Prepared training data (50% subsample, shallow gold): X={X.shape}, y={y.shape}, positives={y.sum()}"
    )

    # Simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if y.sum()>0 else None)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    feature_names = getattr(prepare_training_data, "last_feature_names", None)
    if feature_names:
        clf.feature_names_in_ = np.array(feature_names, dtype=object)

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
