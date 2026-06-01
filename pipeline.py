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


def run_pipeline(lat, lon, target_epsg=7851):
    logger = setup_logging()
    logger.info("Starting unified pipeline")
    logger.info("Target location lat=%s lon=%s target_epsg=%s", lat, lon, target_epsg)

    pipeline = script.WAExplorationPipeline(target_epsg=target_epsg)

    logger.info("Discovering WFS layers")
    mineral_layer = pipeline.select_mineral_occurrence_layer()
    if mineral_layer:
        logger.info("Selected mineral layer: %s", mineral_layer)
    fault_matches = pipeline.discover_wfs_layers("fault")
    logger.info("Fault matches: %s", ", ".join(fault_matches) or "none")

    vector_layers = [mineral_layer] if mineral_layer else []
    payload = pipeline.execute_pipeline(lat, lon, vector_layers)
    logger.info("Ingested vector count: %d", len(payload.get("vector", {})))
    logger.info("Ingested raster count: %d", len(payload.get("raster", {})))

    pipeline.save_payload(payload)
    logger.info("Saved vector GeoJSONs and raster GeoTIFFs")

    logger.info("Training model")
    train_model.main()
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
    args = parser.parse_args()
    run_pipeline(args.lat, args.lon, target_epsg=args.target_epsg)
