import io
import os
import requests
import re
import time
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

    def calculate_bbox(self, lat, lon, buffer_meters=100000):
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

    def list_wcs_coverages(self):
        """Return a list of available coverage IDs from the GA WCS GetCapabilities."""
        try:
            resp = requests.get(self.raster_wcs_url, params={"service": "WCS", "request": "GetCapabilities", "version": "2.0.1"}, timeout=90)
            resp.raise_for_status()
            txt = resp.text
            covs = set(re.findall(r'<wcs:CoverageId>(.*?)</wcs:CoverageId>', txt))
            if not covs:
                covs = set(re.findall(r'<CoverageId>(.*?)</CoverageId>', txt))
            if not covs:
                covs = set(re.findall(r'coverageId="(.*?)"', txt))
            return sorted(list(covs))
        except Exception as e:
            print(f"❌ WCS GetCapabilities error: {e}")
            return [self.raster_coverage_id]


    def pull_wcs_coverage(self, bbox_str, coverage_id):
        """Pull a single GA WCS coverage by ID and return array/transform/crs."""
        print(f"-> Pulling WCS coverage: {coverage_id}")
        lon_min, lat_min, lon_max, lat_max = bbox_str.split(",")

        # Try to discover coverage CRS and axis labels via DescribeCoverage
        axis_labels = None
        target_crs_code = None
        try:
            desc = requests.get(self.raster_wcs_url, params={"service": "WCS", "request": "DescribeCoverage", "version": "2.0.1", "coverageId": coverage_id}, timeout=30)
            desc.raise_for_status()
            txt = desc.text
            # Look for gml:Envelope srsName and axisLabels
            m = re.search(r'<gml:Envelope[^>]*srsName="([^"]+)"[^>]*axisLabels="([^"]+)"', txt)
            if m:
                srs = m.group(1)
                axis_labels = m.group(2).split()
                # try to extract EPSG code
                m2 = re.search(r'EPSG(?:/0/|/)(\d+)', srs)
                if m2:
                    target_crs_code = int(m2.group(1))
        except Exception:
            axis_labels = None

        # Build candidate subset parameter combinations to try (order matters)
        attempts = []

        # helper to build base params with optional subset tuples
        def base_params(subs=None):
            p = [
                ("service", "WCS"),
                ("version", "2.0.1"),
                ("request", "GetCoverage"),
                ("coverageId", coverage_id),
                ("format", "image/tiff"),
            ]
            if subs:
                p.extend(subs)
            return p

        # If we discovered axis labels and a coverage CRS, prepare transformed numeric bounds
        transformed_bounds = None
        if axis_labels and target_crs_code is not None:
            try:
                from pyproj import Transformer as _Transformer
                transformer = _Transformer.from_crs("EPSG:4326", f"EPSG:{target_crs_code}", always_xy=True)
                x1, y1 = transformer.transform(float(lon_min), float(lat_min))
                x2, y2 = transformer.transform(float(lon_max), float(lat_max))
                a_min = min(x1, x2)
                a_max = max(x1, x2)
                b_min = min(y1, y2)
                b_max = max(y1, y2)
                transformed_bounds = (a_min, a_max, b_min, b_max)
            except Exception:
                transformed_bounds = None

        # Common candidate subsets (prefer coverage axisLabels when available)
        if axis_labels and transformed_bounds:
            a_min, a_max, b_min, b_max = transformed_bounds
            # Primary: axis labels in discovered order
            subs = [("subset", f"{axis_labels[0]}({a_min},{a_max})"), ("subset", f"{axis_labels[1]}({b_min},{b_max})")]
            attempts.append(base_params(subs))
            # swapped order
            subs_swapped = [("subset", f"{axis_labels[1]}({b_min},{b_max})"), ("subset", f"{axis_labels[0]}({a_min},{a_max})")]
            attempts.append(base_params(subs_swapped))

        # Try common Lat/Long variants using geographic coords
        latlong = [("subset", f"Lat({lat_min},{lat_max})"), ("subset", f"Long({lon_min},{lon_max})")]
        attempts.append(base_params(latlong))
        attempts.append(base_params(list(reversed(latlong))))
        attempts.append(base_params([("subset", f"lat({lat_min},{lat_max})"), ("subset", f"long({lon_min},{lon_max})")]))

        # If discovered axis labels but no transformation (e.g., axisLabels 'Lat Long'), try using numeric degrees with those labels
        if axis_labels and not transformed_bounds:
            try:
                subs_deg = [("subset", f"{axis_labels[0]}({lat_min},{lat_max})"), ("subset", f"{axis_labels[1]}({lon_min},{lon_max})")]
                attempts.append(base_params(subs_deg))
            except Exception:
                pass

        # Fallback: no subset (full coverage)
        attempts.append(base_params(None))

        # Also attempt lowercase axis names for discovered labels
        if axis_labels:
            lower = [lbl.lower() for lbl in axis_labels]
            try:
                if transformed_bounds:
                    subs = [("subset", f"{lower[0]}({a_min},{a_max})"), ("subset", f"{lower[1]}({b_min},{b_max})")]
                    attempts.append(base_params(subs))
            except Exception:
                pass

        last_error = None
        for idx, params_try in enumerate(attempts):
            try:
                # small backoff for repeated attempts
                if idx > 0:
                    time.sleep(min(1.0, 0.2 * idx))

                response = requests.get(self.raster_wcs_url, params=params_try, timeout=120)
                # if server returns XML error, we may want to inspect and continue
                if response.status_code >= 400:
                    txt = response.text or ""
                    # if InvalidAxisLabel, try next attempt
                    if "InvalidAxisLabel" in txt or "Invalid axis label" in txt:
                        last_error = txt.strip()[:300]
                        continue
                    else:
                        last_error = f"HTTP {response.status_code}: {txt.strip()[:300]}"
                        continue

                content_type = response.headers.get("content-type", "").lower()
                if not content_type.startswith("image/"):
                    last_error = f"Unexpected content-type: {content_type}"
                    # if XML with exception, try next
                    continue

                # Got an image payload; try to read it
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
                last_error = str(e)
                continue

        # If we reach here, all attempts failed. Try to find a direct link in DescribeCoverage text (common for radmap S3 GeoTIFF staging)
        try:
            desc = requests.get(self.raster_wcs_url, params={"service": "WCS", "request": "DescribeCoverage", "version": "2.0.1", "coverageId": coverage_id}, timeout=30)
            desc.raise_for_status()
            urls = re.findall(r'https?://[^"\s]+\\.tif(?:f)?', desc.text)
            for u in urls:
                try:
                    r2 = requests.get(u, timeout=120)
                    r2.raise_for_status()
                    with MemoryFile(r2.content) as memfile:
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
                except Exception:
                    continue
        except Exception:
            pass

        print(f"❌ Raster Error for {coverage_id}: {last_error}")
        return None

    def save_payload(self, payload):
        """Save vector GeoJSONs and raster GeoTIFFs for downstream steps.

        Vector layers are written under `output/vectors/` as GeoJSON files.
        Raster coverages are written under `output/rasters/` as GeoTIFFs.
        """
        layer_names = []
        vectors_dir = os.path.join(self.output_dir, "vectors")
        rasters_dir = os.path.join(self.output_dir, "rasters")
        os.makedirs(vectors_dir, exist_ok=True)
        os.makedirs(rasters_dir, exist_ok=True)

        for lname, gdf in payload.get("vector", {}).items():
            layer_names.append(lname)
            file_name = f"{lname.replace(':', '__')}.geojson"
            out_path = os.path.join(vectors_dir, file_name)
            try:
                if gdf is None or gdf.empty:
                    geojson = "{}"
                else:
                    geojson = gdf.to_crs(epsg=4326).to_json()
            except Exception:
                geojson = "{}"
            with open(out_path, "w", encoding="utf-8") as handle:
                handle.write(geojson)
            print(f"-> Saved GeoJSON: {out_path}")

        # Save all raster coverages found in the payload; keep the configured primary coverage
        raster_dict = payload.get("raster", {})
        primary = self.raster_coverage_id

        # Also store each coverage explicitly under a raster__<id> key for downstream use
        for cov_id, info in raster_dict.items():
            try:
                if info is None:
                    continue
                try:
                    tif_path = os.path.join(rasters_dir, f"{cov_id.replace(':', '__')}.tif")
                    self.save_raster_geotiff(info, out_path=tif_path)
                except Exception:
                    pass
            except Exception:
                continue

        print(f"-> Saved {len(layer_names)} vector layer(s) to {vectors_dir}")

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
        
        # Discover available coverages and pull each one (may be large)
        coverages = self.list_wcs_coverages()
        for cov in coverages:
            info = self.pull_wcs_coverage(bbox_str, cov)
            if info is not None:
                feature_payload["raster"][cov] = info

        # For backward compatibility, expose the configured quartz coverage under the key 'aster_quartz'
        if self.raster_coverage_id in feature_payload["raster"]:
            feature_payload["raster"]["aster_quartz"] = feature_payload["raster"][self.raster_coverage_id]
                
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

    #kalgoolie
    target_lat = -30.7766
    target_lon = 121.5065

    #leonora
    #target_lat = -28.8851
    #target_lon = 121.3283

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