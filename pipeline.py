import argparse
import json
import logging
import os
import sys

import predict_probability
import script
import show
import train_model

OUTPUT_DIR = "output"
LOG_FILE = os.path.join(OUTPUT_DIR, "pipeline.log")
LAYERS_FILE = os.path.join(OUTPUT_DIR, "layers_used.json")
MAP_FILE = os.path.join(OUTPUT_DIR, "map_pipeline.html")


def setup_logging():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = logging.getLogger("pocketwatch.pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def save_layer_manifest(logger, payload, feature_names=None):
    manifest = {
        "vector_layers": sorted(list(payload.get("vector", {}).keys())),
        "raster_layers": sorted(list(payload.get("raster", {}).keys())),
        "training_feature_names": list(feature_names or []),
    }
    with open(LAYERS_FILE, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    logger.info("Layer manifest written to %s", LAYERS_FILE)
    logger.info("Vector layers used: %s", ", ".join(manifest["vector_layers"]) or "none")
    logger.info("Raster layers used: %s", ", ".join(manifest["raster_layers"]) or "none")
    if manifest["training_feature_names"]:
        logger.info("Training features: %s", ", ".join(manifest["training_feature_names"]))


def run_pipeline(lat, lon, target_epsg=7851, force=False, pad_fraction=0.20, pad_meters=None, log_wcs=False):
    logger = setup_logging()
    logger.info("Starting unified pipeline")
    logger.info("Target location lat=%s lon=%s target_epsg=%s", lat, lon, target_epsg)
    logger.info("WCS padding: fraction=%s meters=%s log_wcs=%s", pad_fraction, pad_meters, log_wcs)
    if force:
        logger.info("Force flag enabled: forcing retrain and full raster downloads")

    pipeline = script.WAExplorationPipeline(target_epsg=target_epsg, pad_fraction=pad_fraction, pad_meters=pad_meters, log_wcs=log_wcs)

    logger.info("Discovering WFS layers")
    mineral_layer = pipeline.select_mineral_occurrence_layer()
    if mineral_layer:
        logger.info("Selected mineral layer: %s", mineral_layer)
    fault_matches = pipeline.discover_wfs_layers("fault")
    logger.info("Fault matches: %s", ", ".join(fault_matches) or "none")

    vector_layers = [mineral_layer] if mineral_layer else []
    # If a trained model already exists and we're not forcing retrain, restrict
    # downloads to only the rasters required by the model to run prediction
    # (plus the primary raster). When `force` is True do full downloads.
    requested_rasters = None
    try:
        if (not force) and os.path.exists(train_model.MODEL_PATH):
            logger.info("Existing model detected; restricting raster downloads to prediction inputs")
            try:
                model = predict_probability.load_model(train_model.MODEL_PATH)
                model_feature_names = getattr(model, "feature_names_in_", None)
                if model_feature_names is not None:
                    requested_rasters = [str(n) for n in list(model_feature_names)]
            except Exception:
                requested_rasters = None
    except Exception:
        requested_rasters = None

    payload = pipeline.execute_pipeline(lat, lon, vector_layers, requested_rasters=requested_rasters)
    logger.info("Ingested vector count: %d", len(payload.get("vector", {})))
    logger.info("Ingested raster count: %d", len(payload.get("raster", {})))

    pipeline.save_payload(payload)
    logger.info("Saved vector GeoJSONs and raster GeoTIFFs")

    logger.info("Training model")
    train_model.main(force=force)
    feature_names = getattr(train_model.prepare_training_data, "last_feature_names", [])
    if feature_names:
        logger.info("Model feature names: %s", ", ".join(feature_names))

    logger.info("Generating probability grids")
    predict_probability.main()
    logger.info("Prediction outputs refreshed")

    logger.info("Generating map")
    show.build_map(out_map=MAP_FILE, tif_path=predict_probability.HIGH_CONFIDENCE_TIF_PATH)
    logger.info("Map written to %s", MAP_FILE)

    save_layer_manifest(logger, payload, feature_names)
    logger.info("Unified pipeline complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full pocketwatch pipeline in one script")
    parser.add_argument("--lat", type=float, default=-30.7766, help="Target latitude for ingestion")
    parser.add_argument("--lon", type=float, default=121.5065, help="Target longitude for ingestion")
    parser.add_argument("--target-epsg", type=int, default=7851, help="Target EPSG for ingestion")
    parser.add_argument("--force", action="store_true", help="Force retraining and full raster downloads even if a model exists")
    parser.add_argument("--pad-fraction", type=float, default=0.20, help="Fractional padding for WCS requests (e.g. 0.2 => 20%)")
    parser.add_argument("--pad-meters", type=float, default=None, help="Fixed padding in metres for WCS requests (overrides pad-fraction if set)")
    parser.add_argument("--log-wcs", action="store_true", help="Log expanded WCS request bounds for debugging")
    args = parser.parse_args()
    run_pipeline(args.lat, args.lon, target_epsg=args.target_epsg, force=args.force, pad_fraction=args.pad_fraction, pad_meters=args.pad_meters, log_wcs=args.log_wcs)
