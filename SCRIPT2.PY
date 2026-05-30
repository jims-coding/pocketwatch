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
            matches = [name for name in