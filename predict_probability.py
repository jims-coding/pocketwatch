import os
import numpy as np
import joblib
import rasterio
from rasterio.transform import Affine

OUTPUT_DIR = "output"
NPZ_PATH = os.path.join(OUTPUT_DIR, "dataset_package.npz")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.joblib")
PROB_TIF_PATH = os.path.join(OUTPUT_DIR, "probability_grid.tif")
PROB_NPY_PATH = os.path.join(OUTPUT_DIR, "probability_grid.npy")


def load_package(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)
    return np.load(npz_path, allow_pickle=True)


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


def main():
    data = load_package(NPZ_PATH)
    model = load_model(MODEL_PATH)
    raster, transform, raster_crs = extract_raster_and_transform(data)

    prob_grid = predict_probability_grid(model, raster)

    np.save(PROB_NPY_PATH, prob_grid)
    save_probability_geotiff(prob_grid, transform, raster_crs, PROB_TIF_PATH)

    finite = prob_grid[np.isfinite(prob_grid)]
    print(f"Saved probability grid to {PROB_NPY_PATH}")
    print(f"Saved probability GeoTIFF to {PROB_TIF_PATH}")
    print(f"Grid shape: {prob_grid.shape}")
    if finite.size:
        print(f"Probability range: {finite.min():.4f} to {finite.max():.4f}")
        print(f"Mean probability: {finite.mean():.4f}")


if __name__ == "__main__":
    main()
