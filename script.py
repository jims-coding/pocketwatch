import io
import requests
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from pyproj import Transformer
import rasterio
from rasterio.io import MemoryFile

class WAExplorationPipeline:
    def __init__(self, target_epsg=7851):
        """
        Initializes the pipeline orchestrator.
        Defaults to EPSG:7851 (GDA2020 / MGA Zone 51) for the WA Goldfields.
        """
        self.target_epsg = target_epsg
        self.wfs_url = "http://geossdi.dmp.wa.gov.au/services/wfs"
        self.wcs_url = "http://geossdi.dmp.wa.gov.au/services/wcs"
        
        # Setup coordinate transformers
        self.to_meters = Transformer.from_crs("EPSG:4326", f"EPSG:{target_epsg}", always_xy=True)
        self.to_degrees = Transformer.from_crs(f"EPSG:{target_epsg}", "EPSG:4326", always_xy=True)

    def calculate_bbox(self, lat, lon, buffer_meters=1000):
        """Calculates metric coordinates and returns degree-based BBOX string."""
        x_center, y_center = self.to_meters.transform(lon, lat)
        
        x_min, x_max = x_center - buffer_meters, x_center + buffer_meters
        y_min, y_max = y_center - buffer_meters, y_center + buffer_meters
        
        lon_min, lat_min = self.to_degrees.transform(x_min, y_min)
        lon_max, lat_max = self.to_degrees.transform(x_max, y_max)
        
        return f"{lon_min},{lat_min},{lon_max},{lat_max}", (x_min, y_min, x_max, y_max)

    def pull_vector_features(self, bbox_str, layer_name):
        """Queries WFS server for vector attributes within the bounding box."""
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": layer_name,
            "bbox": bbox_str,
            "outputFormat": "application/json"
        }
        try:
            print(f"-> Pulling vector layer: {layer_name}")
            response = requests.get(self.wfs_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if not data.get("features"):
                return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{self.target_epsg}")
                
            gdf = gpd.read_file(io.StringIO(response.text))
            return gdf.to_crs(epsg=self.target_epsg)
        except Exception as e:
            print(f"❌ WFS Error [{layer_name}]: {e}")
            return None

    def pull_raster_coverage(self, bbox_str, layer_name):
        """Queries WCS server for raster grids (ASTER/Geophysics) within the bounding box."""
        params = {
            "service": "WCS",
            "version": "1.0.0",
            "request": "GetCoverage",
            "coverage": layer_name,
            "crs": "EPSG:4326",
            "bbox": bbox_str,
            "format": "GeoTIFF",
            "resx": "0.00027", # ~30m cell resolution for SWIR bands
            "resy": "0.00027"
        }
        try:
            print(f"-> Pulling raster coverage: {layer_name}")
            response = requests.get(self.wcs_url, params=params, timeout=45)
            response.raise_for_status()
            
            with MemoryFile(response.content) as memfile:
                with memfile.open() as dataset:
                    raster_array = dataset.read(1)
                    nodata = dataset.nodata
                    if nodata is not None:
                        raster_array = np.where(raster_array == nodata, np.nan, raster_array)
                    return raster_array, dataset.transform
        except Exception as e:
            print(f"❌ WCS Error [{layer_name}]: {e}")
            return None, None

    def execute_pipeline(self, lat, lon, vector_layers, raster_layers):
        """Orchestrates full data ingestion and vector-raster alignment loop."""
        bbox_str, metric_bounds = self.calculate_bbox(lat, lon)
        feature_payload = {"vector": {}, "raster": {}}
        
        # 1. Fetch Vectors
        for v_layer in vector_layers:
            gdf = self.pull_vector_features(bbox_str, v_layer)
            if gdf is not None:
                feature_payload["vector"][v_layer] = gdf
                
        # 2. Fetch Rasters
        for r_layer in raster_layers:
            arr, transform = self.pull_raster_coverage(bbox_str, r_layer)
            if arr is not None:
                feature_payload["raster"][r_layer] = {"array": arr, "transform": transform}
                
        print("\n✅ Execution completed successfully.")
        return feature_payload

# --- Execution Entry Point ---
if __name__ == "__main__":
    # Point configuration (Example: Kalgoorlie exploration window)
    target_lat = -30.7489
    target_lon = 121.4658
    
    # Selected layers to engineer modeling vectors
    v_targets = ["geonode:mineral_occurrences", "geonode:faults_and_lineaments"]
    r_targets = ["aster:quartz_index"] 
    
    # Instantiation and run
    pipeline = WAExplorationPipeline(target_epsg=7851)
    dataset_package = pipeline.execute_pipeline(target_lat, target_lon, v_targets, r_targets)
    
    # Structural Verification Output
    if "geonode:mineral_occurrences" in dataset_package["vector"]:
        occ_df = dataset_package["vector"]["geonode:mineral_occurrences"]
        print(f"\nExtracted {len(occ_df)} mineral occurrence vectors within bounding matrix.")
        
    if "aster:quartz_index" in dataset_package["raster"]:
        aster_data = dataset_package["raster"]["aster:quartz_index"]["array"]
        print(f"Extracted ASTER Raster Matrix. Shape: {aster_data.shape} | Area Mean: {np.nanmean(aster_data):.4f}")