import os
import numpy as np
import joblib
import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
import train_model

OUTPUT_DIR = "output"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.joblib")
PROB_TIF_PATH = os.path.join(OUTPUT_DIR, "probability_grid.tif")
PROB_NPY_PATH = os.path.join(OUTPUT_DIR, "probability_grid.npy")
HIGH_CONFIDENCE_TIF_PATH = os.path.join(OUTPUT_DIR, "high_confidence_grid.tif")
HIGH_CONFIDENCE_NPY_PATH = os.path.join(OUTPUT_DIR, "high_confidence_grid.npy")

# Conservative cutoff tuned for high precision. On the held-out split this
# corresponds to roughly 97% precision and about 5% recall.
HIGH_CONFIDENCE_THRESHOLD = 0.90


def load_package(_unused_path=None):
    rasters_dir = os.path.join(OUTPUT_DIR, "rasters")
    mapping = {}
    for fn in sorted(os.listdir(rasters_dir)) if os.path.isdir(rasters_dir) else []:
        if not fn.lower().endswith((".tif", ".tiff")):
            continue
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


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    return joblib.load(model_path)


def extract_raster_and_transform(data):
    if "raster" not in data:
        raise RuntimeError("No raster found in dataset package")

    raster = data["raster"]
    transform_arr = data.get("raster_transform", np.array([]))
    raster_crs = data.get("raster_crs", np.array([None]))

    if hasattr(raster_crs, "tolist"):
        crs_value = raster_crs.tolist()
        if isinstance(crs_value, (list, tuple, np.ndarray)):
            raster_crs = crs_value[0] if len(crs_value) > 0 else None
        else:
            raster_crs = crs_value

    if transform_arr is None or len(transform_arr) == 0:
        raise RuntimeError("No raster transform found in dataset package")

    transform = Affine(*list(map(float, transform_arr)))
    return raster, transform, raster_crs


def build_feature_matrix(raster):
    values = raster.flatten()
    valid_mask = ~np.isnan(values)

    if valid_mask.sum() == 0:
        raise RuntimeError("Raster contains only NaN values")

    vals_f = values[valid_mask]

    X = vals_f.reshape(-1, 1)
    return X, valid_mask


def build_stacked_features_from_package(data, feature_names=None):
    # determine base raster key
    if "raster" in data:
        base_key = "raster"
    else:
        base_key = None
        for k in data.files:
            if k.startswith("raster__") and not k.endswith("__crs") and not k.endswith("__transform"):
                base_key = k
                break
    if base_key is None:
        raise RuntimeError("No raster found in dataset package")

    base = data[base_key]
    base_transform = data.get(f"{base_key}__transform", data.get("raster_transform", None))
    base_crs = data.get(f"{base_key}__crs", data.get("raster_crs", None))
    base_transform_aff = train_model._coerce_transform(base_transform)
    base_crs_val = train_model._coerce_crs(base_crs)

    # collect predictors in the exact training order when available
    if feature_names:
        desired_prefixes = [name for name in feature_names if f"raster__{name}" in data.files]
    else:
        desired_prefixes = []
        for k in data.files:
            if not k.startswith("raster__"):
                continue
            if k.endswith("__crs") or k.endswith("__transform"):
                continue
            if k == "raster":
                continue
            desired_prefixes.append(k.replace("raster__", ""))

    feature_arrays = []
    dst_shape = base.shape
    for name in desired_prefixes:
        key = f"raster__{name}"
        if key not in data.files:
            continue
        src = data[key]
        src_t = data.get(f"{key}__transform", None)
        src_c = data.get(f"{key}__crs", None)
        src_t_aff = train_model._coerce_transform(src_t)
        src_crs_val = train_model._coerce_crs(src_c)

        dst = np.full(dst_shape, np.nan, dtype=float)
        try:
            reproject(
                source=src,
                destination=dst,
                src_transform=Affine(*src_t_aff) if src_t_aff is not None else None,
                src_crs=src_crs_val,
                dst_transform=Affine(*base_transform_aff) if base_transform_aff is not None else None,
                dst_crs=base_crs_val,
                resampling=Resampling.bilinear,
            )
        except Exception:
            try:
                reproject(
                    source=src,
                    destination=dst,
                    src_transform=Affine(*src_t_aff) if src_t_aff is not None else None,
                    src_crs=src_crs_val,
                    dst_transform=Affine(*base_transform_aff) if base_transform_aff is not None else None,
                    dst_crs=base_crs_val,
                    resampling=Resampling.nearest,
                )
            except Exception:
                dst = np.full(dst_shape, np.nan, dtype=float)

        feature_arrays.append(dst)

    if len(feature_arrays) == 0:
        raise RuntimeError("No predictor rasters found to build features")

    stacked = np.stack([a.flatten() for a in feature_arrays], axis=1)
    valid_mask = ~np.isnan(stacked).all(axis=1)
    X = stacked[valid_mask]
    return X, valid_mask, base, base_transform_aff, base_crs_val


def predict_probability_grid(model, raster):
    X, valid_mask = build_feature_matrix(raster)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Loaded model does not support predict_proba")

    proba = model.predict_proba(X)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise RuntimeError("Model predict_proba output does not contain class probabilities for class 1")

    positive_proba = proba[:, 1]

    prob_grid = np.full(raster.size, np.nan, dtype=np.float32)
    prob_grid[valid_mask] = positive_proba.astype(np.float32)
    prob_grid = prob_grid.reshape(raster.shape)
    return prob_grid


def save_probability_geotiff(prob_grid, transform, raster_crs, out_path):
    profile = {
        "driver": "GTiff",
        "height": int(prob_grid.shape[0]),
        "width": int(prob_grid.shape[1]),
        "count": 1,
        "dtype": "float32",
        "crs": raster_crs or "EPSG:4326",
        "transform": transform,
        "nodata": np.nan,
        "compress": "LZW",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(prob_grid.astype(np.float32), 1)


def save_binary_geotiff(mask_grid, transform, raster_crs, out_path):
    profile = {
        "driver": "GTiff",
        "height": int(mask_grid.shape[0]),
        "width": int(mask_grid.shape[1]),
        "count": 1,
        "dtype": "uint8",
        "crs": raster_crs or "EPSG:4326",
        "transform": transform,
        "nodata": 0,
        "compress": "LZW",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask_grid.astype(np.uint8), 1)


def main():
    data = load_package()
    model = load_model(MODEL_PATH)

    # Build stacked feature matrix from the package (resampled to base grid)
    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is not None:
        model_feature_names = [str(name) for name in list(model_feature_names)]
    X, valid_mask, base_raster, base_transform_aff, base_crs_val = build_stacked_features_from_package(data, feature_names=model_feature_names)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Loaded model does not support predict_proba")

    proba = model.predict_proba(X)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise RuntimeError("Model predict_proba output does not contain class probabilities for class 1")

    positive_proba = proba[:, 1].astype(np.float32)

    prob_grid = np.full(base_raster.size, np.nan, dtype=np.float32)
    prob_grid[valid_mask] = positive_proba
    prob_grid = prob_grid.reshape(base_raster.shape)

    high_confidence_grid = np.zeros_like(prob_grid, dtype=np.uint8)
    high_confidence_grid[np.isfinite(prob_grid) & (prob_grid >= HIGH_CONFIDENCE_THRESHOLD)] = 1

    np.save(PROB_NPY_PATH, prob_grid)
    save_probability_geotiff(prob_grid, base_transform_aff and Affine(*base_transform_aff) or None, base_crs_val, PROB_TIF_PATH)

    np.save(HIGH_CONFIDENCE_NPY_PATH, high_confidence_grid)
    save_binary_geotiff(high_confidence_grid, base_transform_aff and Affine(*base_transform_aff) or None, base_crs_val, HIGH_CONFIDENCE_TIF_PATH)

    finite = prob_grid[np.isfinite(prob_grid)]
    print(f"Saved probability grid to {PROB_NPY_PATH}")
    print(f"Saved probability GeoTIFF to {PROB_TIF_PATH}")
    print(f"Saved high-confidence mask to {HIGH_CONFIDENCE_NPY_PATH}")
    print(f"Saved high-confidence GeoTIFF to {HIGH_CONFIDENCE_TIF_PATH}")
    print(f"High-confidence threshold: {HIGH_CONFIDENCE_THRESHOLD:.2f}")
    print(f"Grid shape: {prob_grid.shape}")
    if finite.size:
        print(f"Probability range: {finite.min():.4f} to {finite.max():.4f}")
        print(f"Mean probability: {finite.mean():.4f}")
        print(f"High-confidence picks: {int(high_confidence_grid.sum())}")


if __name__ == "__main__":
    main()
