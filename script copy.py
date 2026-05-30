import io
import requests
import re
import numpy as np
import geopandas as gpd
from pyproj import Transformer
import rasterio
from rasterio.io import MemoryFile

class WAExplorationPipeline:
    def __init__(self, target_epsg=7851):
        self.target_epsg = target_epsg
        self.wfs_url = "https://geossdi.dmp.wa.gov.au/services/wfs"
        self.raster_resolution_m = 30
        self.raster_wcs_url = "https://services.ga.gov.au/gis/machine-learning-models/wcs"
        self.raster_coverage_id = "ml__radmap_v4_2019_filtered_ML_pctk"

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
                    return raster_array

        except Exception as e:
            print(f"❌ Raster Error: {e}")
            grid_size = int(round(2000 / self.raster_resolution_m))
            return np.full((grid_size, grid_size), np.nan)

    def execute_pipeline(self, lat, lon, vector_layers):
        bbox_str = self.calculate_bbox(lat, lon)
        feature_payload = {"vector": {}, "raster": {}}
        
        for v_layer in vector_layers:
            gdf = self.pull_vector_features(bbox_str, v_layer)
            if gdf is not None:
                feature_payload["vector"][v_layer] = gdf
                
        arr = self.pull_aster_ga_wcs(bbox_str)
        if arr is not None:
             feature_payload["raster"]["aster_quartz"] = arr
                
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
        aster_data = dataset_package["raster"]["aster_quartz"]
        print(f"ASTER Quartz Matrix   : Shape {aster_data.shape}")
        if np.isnan(aster_data).all():
             print("ASTER Quartz Matrix   : No data in this specific grid cell.")
        else:
             print(f"Mean Quartz Alteration: {np.nanmean(aster_data):.4f}")