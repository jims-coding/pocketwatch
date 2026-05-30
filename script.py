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
        # DMIRS WFS Endpoint for Vector Data
        self.wfs_url = "https://geossdi.dmp.wa.gov.au/services/wfs"
        self.raster_resolution_m = 30
        
        # Coordinate Transformers
        self.to_meters = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)
        self.to_degrees = Transformer.from_crs(f"EPSG:{target_epsg}", "EPSG:4326", always_xy=True)

    def _get_wfs_capability_layers(self):
        """Fetches the master list of all available layers from the server."""
        params = {"service": "WFS", "version": "1.0.0", "request": "GetCapabilities"}
        response = requests.get(self.wfs_url, params=params, timeout=15)
        response.raise_for_status()

        layer_names = set(re.findall(r"<Name>(.*?)</Name>", response.text))
        return sorted(name for name in layer_names if ":" in name)

    def select_mineral_occurrence_layer(self):
        """Hunts for the most likely mineral occurrence layer."""
        layer_names = self._get_wfs_capability_layers()
        preferred = ["mo:MinOccView", "gsml:MappedFeature", "er:MiningFeatureOccurrence", "dmp:mineral_occurrences"]
        for layer_name in preferred:
            if layer_name in layer_names:
                print(f"   ✅ Selected mineral-occurrence layer: {layer_name}")
                return layer_name

        print("   ⚠️ No explicit mineral-occurrence layer found")
        return None

    def discover_wfs_layers(self, search_term):
        """Hunts for layers matching a specific string (e.g., 'structure')."""
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
        """Converts Lat/Lon center to a 2000m x 2000m geographic bounding box."""
        x_center, y_center = self.to_meters.transform(lon, lat)
        x_min, x_max = x_center - buffer_meters, x_center + buffer_meters
        y_min, y_max = y_center - buffer_meters, y_center + buffer_meters
        lon_min, lat_min = self.to_degrees.transform(x_min, y_min)
        lon_max, lat_max = self.to_degrees.transform(x_max, y_max)
        
        bbox_str = f"{lon_min},{lat_min},{lon_max},{lat_max}"
        return bbox_str

    def pull_vector_features(self, bbox_str, layer_name):
        """Pulls vector attributes from DMIRS, handling XML exceptions silently."""
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
            
            # Intercept XML server errors masquerading as successful JSON responses
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
        """Pulls ASTER Silica (Quartz) Index directly from Geoscience Australia."""
        print("-> Pulling ASTER raster via Geoscience Australia API...")
        
        # Official GA endpoint specifically for ASTER Map of Australia Silica Index
        wcs_url = "https://services.ga.gov.au/gis/services/ASTER_Map_of_Australia_Silica_Index/MapServer/WCSServer"

        params = {
            "service": "WCS",
            "version": "1.0.0",
            "request": "GetCoverage",
            "coverage": "1", # Standard GA MapServer coverage ID
            "crs": "EPSG:4326",
            "bbox": bbox_str,
            "format": "GeoTIFF",
            "resx": "0.00027",
            "resy": "0.00027",
        }

        try:
            response = requests.get(wcs_url, params=params, timeout=45)

            if response.status_code == 200:
                try:
                    with MemoryFile(response.content) as memfile:
                        with memfile.open() as dataset:
                            raster_array = dataset.read(1).astype(float)
                            nodata = dataset.nodata
                            if nodata is not None:
                                raster_array = np.where(raster_array == nodata, np.nan, raster_array)
                            return raster_array
                except Exception as parse_e:
                    print(f"   ❌ Failed to parse TIF: {parse_e}")
            else:
                print(f"   ❌ GA API Error {response.status_code}: {response.text[:200]}")

            print("   ⚠️ Raster service unavailable; returning an empty NaN grid.")
            grid_size = int(round(2000 / self.raster_resolution_m))
            return np.full((grid_size, grid_size), np.nan)
                    
        except Exception as e:
             print(f"❌ Raster Error: {e}")
             grid_size = int(round(2000 / self.raster_resolution_m))
             return np.full((grid_size, grid_size), np.nan)

    def execute_pipeline(self, lat, lon, vector_layers):
        """Orchestrates full data ingestion and vector-raster alignment loop."""
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


# --- Execution Entry Point ---
if __name__ == "__main__":
    pipeline = WAExplorationPipeline(target_epsg=7851)
    
    print("\n--- 🕵️ STEP 1: DISCOVERING HIDDEN DMIRS LAYER NAMES ---")
    print("Searching for mineral-occurrence layer...")
    mineral_layer = pipeline.select_mineral_occurrence_layer()
    
    print("\nSearching for Structure layers...")
    structure_matches = pipeline.discover_wfs_layers("structure")

    # Build target array dynamically based on what the server actually has
    v_targets = []
    if mineral_layer:
        v_targets.append(mineral_layer)
        
    # Grab the first structure layer if it exists (usually gsml:ShearDisplacementStructureView)
    if structure_matches:
        v_targets.append(structure_matches[0])
        print(f"   ✅ Auto-selecting structure layer: {structure_matches[0]}")
    
    # Target Point
    target_lat = -30.7489
    target_lon = 121.4658
    
    print(f"\n--- 🚀 STEP 2: RUNNING PIPELINE FOR TARGET ({target_lat}, {target_lon}) ---")
    dataset_package = pipeline.execute_pipeline(target_lat, target_lon, v_targets)
    
    print("\n--- 📊 EXTRACTED FEATURE MATRIX SUMMARY ---")
    
    # Output verification dynamically based on what was injected
    if len(v_targets) > 0 and v_targets[0] in dataset_package["vector"]:
        occ_df = dataset_package["vector"][v_targets[0]]
        print(f"Target 1 ({v_targets[0]:<40}) : {len(occ_df)} features detected")

    if len(v_targets) > 1 and v_targets[1] in dataset_package["vector"]:
        struct_df = dataset_package["vector"][v_targets[1]]
        print(f"Target 2 ({v_targets[1]:<40}) : {len(struct_df)} features detected")
    elif not structure_matches:
        print("Secondary Vector Layer: No structure layer exposed by this server.")
        
    if "aster_quartz" in dataset_package["raster"]:
        aster_data = dataset_package["raster"]["aster_quartz"]
        print(f"ASTER Quartz Matrix                                : Shape {aster_data.shape}")
        if np.isnan(aster_data).all():
             print("ASTER Quartz Matrix                                : No data in this specific grid cell.")
        else:
             print(f"Mean Quartz Alteration                             : {np.nanmean(aster_data):.4f}")