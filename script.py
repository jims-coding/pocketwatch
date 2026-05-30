import io
import os
import requests
import re
import numpy as np
import geopandas as gpd
from pyproj import Transformer
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import Affine

class WAExplorationPipeline:
    def __init__(self, target_epsg=7851):
        self.target_epsg = target_epsg
        self.wfs_url = "https://geossdi.dmp.wa.gov.au/services/wfs"
        self.raster_resolution_m = 30
        self.raster_wcs_url = "https://services.ga.gov.au/gis/machine-learning-models/wcs"
        self.raster_coverage_id = "ml__radmap_v4_2019_filtered_ML_pctk"
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

        self.to_meters = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)
        self.to_degrees = Transformer.from_crs(f"EPSG:{target_epsg}", "EPSG:4326", always_xy=True)

    def _get_wfs_capability_layers(self):
        params = {"service": "WFS", "version": "1.0.0", "request": "GetCapabilities"}
        response = requests.get(self.wfs_url, params=params, timeout=15)
        response.raise_for_status()

        layer_names = set(re.findall(r"<Name>(.*?)</Name>", response.text))
        return sorted(name for name in layer_names if ":" in name)

    def select_mineral_occurrence_layer(self):
        layer_names = self._get_wfs_capability_layers()
        preferred = ["mo:MinOccView", "gsml:MappedFeature", "er:MiningFeatureOccurrence"]
        for layer_name in preferred:
            if layer_name in layer_names:
                print(f"   ✅ Selected mineral-occurrence layer: {layer_name}")
                return layer_name

        print("   ⚠️ No explicit mineral-occurrence layer found")
        return None

    def discover_wfs_layers(self, search_term):
        """Hacks the GetCapabilities document to find the exact hidden layer names."""
        try:
            layer_names = self._get_wfs_capability_layers()

            matches = [name for name in layer_names if search_term.lower() in name.lower()]
            if matches:
                for match in matches:
                    print(f"   ✅ Found valid server layer: {match}")
            else:
                print(f"   ⚠️ No layers found containing '{search_term}'")
            return matches
        except Exception as e:
             print(f"❌ Discovery Error: {e}")
             return []

    def calculate_bbox(self, lat, lon, buffer_meters=1000):
        x_center, y_center = self.to_meters.transform(lon, lat)
        x_min, x_max = x_center - buffer_meters, x_center + buffer_meters
        y_min, y_max = y_center - buffer_meters, y_center + buffer_meters
        lon_min, lat_min = self.to_degrees.transform(x_min, y_min)
        lon_max, lat_max = self.to_degrees.transform(x_max, y_max)
        
        bbox_str = f"{lon_min},{lat_min},{lon_max},{lat_max}"
        return bbox_str

    def pull_vector_features(self, bbox_str, layer_name):
        params = {
            "service": "WFS",
            "version": "1.0.0", 
            "request": "GetFeature",
            "typeName": layer_name, 
            "bbox": bbox_str,
            "outputFormat": "application/json",
            "srsName": "EPSG:4326"
        }
        try:
            print(f"-> Pulling vector layer: {layer_name}")
            response = requests.get(self.wfs_url, params=params, timeout=30)

            response.raise_for_status()

            raw_text = response.text.strip()
            if raw_text.startswith("<ServiceExceptionReport") or raw_text.startswith("<?xml"):
                print(f"   ❌ Server XML Exception: {raw_text[:300]}")
                return None

            gdf = gpd.read_file(io.StringIO(raw_text))
            if gdf.empty:
                return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{self.target_epsg}")

            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326)

            return gdf.to_crs(epsg=self.target_epsg)
            
        except Exception as e:
            print(f"❌ WFS Network Error: {e}")
            return None

    def pull_aster_ga_wcs(self, bbox_str):
        """Pull a public GA raster coverage via WCS and read it in memory."""
        print("-> Pulling public GA raster via WCS...")

        lon_min, lat_min, lon_max, lat_max = bbox_str.split(",")
        params = [
            ("service", "WCS"),
            ("version", "2.0.1"),
            ("request", "GetCoverage"),
            ("coverageId", self.raster_coverage_id),
            ("format", "image/tiff"),
            ("subset", f"Lat({lat_min},{lat_max})"),
            ("subset", f"Long({lon_min},{lon_max})"),
        ]

        try:
            response = requests.get(self.raster_wcs_url, params=params, timeout=120)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            if not content_type.startswith("image/"):
                print(f"   ⚠️ Unexpected raster response: {content_type}")
                print(f"   ⚠️ Response head: {response.text[:300]}")
                raise ValueError("WCS did not return an image/tiff payload")

            with MemoryFile(response.content) as memfile:
                with memfile.open() as dataset:
                    raster_array = dataset.read(1).astype(float)
                    nodata = dataset.nodata
                    if nodata is not None:
                        raster_array = np.where(raster_array == nodata, np.nan, raster_array)
                    transform = (
                        float(dataset.transform.a),
                        float(dataset.transform.b),
                        float(dataset.transform.c),
                        float(dataset.transform.d),
                        float(dataset.transform.e),
                        float(dataset.transform.f),
                    )
                    crs = str(dataset.crs) if dataset.crs is not None else None
                    return {"array": raster_array, "transform": transform, "crs": crs}

        except Exception as e:
            print(f"❌ Raster Error: {e}")
            grid_size = int(round(2000 / self.raster_resolution_m))
            nan_grid = np.full((grid_size, grid_size), np.nan)
            return {"array": nan_grid, "transform": None, "crs": f"EPSG:{self.target_epsg}"}

    def save_payload(self, payload, out_path=None):
        """Package vector GeoJSONs and raster array/metadata into a single .npz file.

        Vector layers are stored as GeoJSON strings under keys `vector__<layername>`
        (colons replaced with double-underscores). Raster is stored as `raster`,
        with `raster_transform` and `raster_crs` metadata.
        """
        data = {}
        layer_names = []

        for lname, gdf in payload.get("vector", {}).items():
            layer_names.append(lname)
            key = f"vector__{lname.replace(':', '__')}"
            try:
                if gdf is None or gdf.empty:
                    geojson = "{}"
                else:
                    geojson = gdf.to_crs(epsg=4326).to_json()
            except Exception:
                geojson = "{}"
            data[key] = np.array(geojson, dtype=object)

        raster_info = payload.get("raster", {}).get("aster_quartz")
        if raster_info is not None:
            data["raster"] = raster_info.get("array")
            data["raster_transform"] = np.array(raster_info.get("transform"), dtype=float) if raster_info.get("transform") is not None else np.array([], dtype=float)
            data["raster_crs"] = np.array(raster_info.get("crs"), dtype=object)

        data["vector_layers"] = np.array(layer_names, dtype=object)

        out_path = out_path or os.path.join(self.output_dir, "dataset_package.npz")
        np.savez_compressed(out_path, **data)
        print(f"-> Saved package to: {out_path}")

    def save_raster_geotiff(self, raster_info, out_path=None, compress=True):
        """Save the extracted raster (numpy array + transform + crs) as a GeoTIFF file."""
        if raster_info is None:
            raise ValueError("raster_info is None")

        arr = raster_info.get("array")
        transform = raster_info.get("transform")
        crs = raster_info.get("crs") or f"EPSG:{self.target_epsg}"

        if transform is None:
            print("   ⚠️ No geotransform available; skipping GeoTIFF save.")
            return None

        affine = Affine(*transform)
        height, width = arr.shape

        profile = {
            "driver": "GTiff",
            "height": int(height),
            "width": int(width),
            "count": 1,
            "dtype": arr.dtype,
            "crs": crs,
            "transform": affine,
            "nodata": np.nan,
        }

        if compress:
            profile.update({"tiled": True, "compress": "LZW", "blockxsize": 512, "blockysize": 512})

        out_path = out_path or os.path.join(self.output_dir, "aster_quartz.tif")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr, 1)

        print(f"-> Saved GeoTIFF: {out_path}")

    def execute_pipeline(self, lat, lon, vector_layers):
        bbox_str = self.calculate_bbox(lat, lon)
        feature_payload = {"vector": {}, "raster": {}}
        
        for v_layer in vector_layers:
            gdf = self.pull_vector_features(bbox_str, v_layer)
            if gdf is not None:
                feature_payload["vector"][v_layer] = gdf
        
        arr_info = self.pull_aster_ga_wcs(bbox_str)
        if arr_info is not None:
            feature_payload["raster"]["aster_quartz"] = arr_info
                
        return feature_payload

if __name__ == "__main__":
    pipeline = WAExplorationPipeline(target_epsg=7851)
    
    print("\n--- 🕵️ STEP 1: DISCOVERING HIDDEN DMIRS LAYER NAMES ---")
    print("Searching for mineral-occurrence layer...")
    mineral_layer = pipeline.select_mineral_occurrence_layer()
    print("\nSearching for Fault layers...")
    fault_matches = pipeline.discover_wfs_layers("fault")

    v_targets = [mineral_layer] if mineral_layer else []

    target_lat = -30.7489
    target_lon = 121.4658

    target_lat = -30.7766
    target_lon = 121.5065
    
    print(f"\n--- 🚀 STEP 2: RUNNING PIPELINE FOR TARGET ({target_lat}, {target_lon}) ---")
    dataset_package = pipeline.execute_pipeline(target_lat, target_lon, v_targets)
    
    print("\n--- 📊 EXTRACTED FEATURE MATRIX SUMMARY ---")
    if v_targets and v_targets[0] in dataset_package["vector"]:
        occ_df = dataset_package["vector"][v_targets[0]]
        print(f"Mineral Occurrences   : {len(occ_df)} features detected")

    if not fault_matches:
        print("Secondary Vector Layer: No fault layer exposed by this server.")

    if "aster_quartz" in dataset_package["raster"]:
        raster_info = dataset_package["raster"]["aster_quartz"]
        aster_data = raster_info.get("array")
        print(f"ASTER Quartz Matrix   : Shape {aster_data.shape}")
        if np.isnan(aster_data).all():
            print("ASTER Quartz Matrix   : No data in this specific grid cell.")
        else:
            print(f"Mean Quartz Alteration: {np.nanmean(aster_data):.4f}")

    # Save a single packaged file for downstream models (in output/)
    pipeline.save_payload(dataset_package)
    # Also save the raster as a GeoTIFF for GIS/model pipelines (in output/)
    try:
        if "aster_quartz" in dataset_package["raster"]:
            pipeline.save_raster_geotiff(dataset_package["raster"]["aster_quartz"])
    except Exception as e:
        print(f"❌ Failed to save GeoTIFF: {e}")