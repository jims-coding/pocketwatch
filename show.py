import os
import io
import json
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.transform import array_bounds
from pyproj import Transformer
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    import folium
    from folium.raster_layers import ImageOverlay
except Exception:
    folium = None


OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PKG = os.path.join(OUTPUT_DIR, "dataset_package.npz")
TIFF_FILE = os.path.join(OUTPUT_DIR, "aster_quartz.tif")
PNG_OVERLAY = os.path.join(OUTPUT_DIR, "aster_quartz_overlay.png")
MAP_FILE = os.path.join(OUTPUT_DIR, "map.html")


def create_image_overlay_from_array(arr, transform_tuple, crs, out_png=PNG_OVERLAY):
    # Compute bounds in raster CRS
    transform = Affine(*transform_tuple)
    height, width = arr.shape
    minx, miny, maxx, maxy = array_bounds(height, width, transform)

    # Convert bounds to WGS84 if needed
    if crs is None or crs.upper().startswith("EPSG:4326"):
        west, south, east, north = minx, miny, maxx, maxy
    else:
        try:
            src_crs = crs
            transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
            west, south = transformer.transform(minx, miny)
            east, north = transformer.transform(maxx, maxy)
        except Exception:
            # If transformation fails, fallback to reading from TIFF if present
            west = south = east = north = None

    # Normalize array to 0-1 ignoring NaN
    arr_float = arr.astype(float)
    mask = np.isnan(arr_float)
    if np.all(mask):
        raise ValueError("Raster contains only NaN")

    vmin = np.nanmin(arr_float)
    vmax = np.nanmax(arr_float)
    if vmin == vmax:
        vmin = 0
    norm = (arr_float - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)

    cmap = cm.get_cmap("viridis")
    rgba = cmap(norm)
    rgba[..., 3] = np.where(mask, 0.0, rgba[..., 3])
    rgba_u8 = (rgba * 255).astype(np.uint8)

    img = Image.fromarray(rgba_u8, mode="RGBA")
    img.save(out_png)
    return out_png, (south, west, north, east)


def build_map(npz_path=DATA_PKG, tiff_path=TIFF_FILE, out_map=MAP_FILE):
    if folium is None:
        raise RuntimeError("folium is not installed. Install with: pip install folium")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"{npz_path} not found")

    data = np.load(npz_path, allow_pickle=True)

    # Load raster either from package or from TIFF file
    raster = None
    raster_transform = None
    raster_crs = None

    if "raster" in data:
        raster = data["raster"]
        raster_transform = data.get("raster_transform", np.array([]))
        raster_crs = data.get("raster_crs", np.array([None]))
        raster_crs = raster_crs.tolist() if hasattr(raster_crs, "tolist") else raster_crs
        if isinstance(raster_crs, (list, tuple, np.ndarray)):
            raster_crs = raster_crs[0] if len(raster_crs) > 0 else None
    elif os.path.exists(tiff_path):
        with rasterio.open(tiff_path) as ds:
            raster = ds.read(1)
            raster_transform = np.array([ds.transform.a, ds.transform.b, ds.transform.c, ds.transform.d, ds.transform.e, ds.transform.f])
            raster_crs = str(ds.crs)

    if raster is None:
        raise RuntimeError("No raster found in package or TIFF file")

    # If transform is empty, try reading from TIFF
    if raster_transform is None or (isinstance(raster_transform, np.ndarray) and raster_transform.size == 0):
        if os.path.exists(tiff_path):
            with rasterio.open(tiff_path) as ds:
                raster_transform = np.array([ds.transform.a, ds.transform.b, ds.transform.c, ds.transform.d, ds.transform.e, ds.transform.f])
                raster_crs = str(ds.crs)
        else:
            raise RuntimeError("No valid raster transform available")

    # Ensure raster_crs is like 'EPSG:XXXX' or similar
    if raster_crs is None:
        raster_crs = f"EPSG:4326"

    # Create PNG overlay and get bounds
    png_file, bounds = create_image_overlay_from_array(raster, raster_transform.tolist() if isinstance(raster_transform, np.ndarray) else list(raster_transform), raster_crs)
    south, west, north, east = bounds

    # Center map
    lat_center = (south + north) / 2.0
    lon_center = (west + east) / 2.0

    m = folium.Map(location=[lat_center, lon_center], zoom_start=12, tiles="OpenStreetMap")

    # Add image overlay
    image_overlay = ImageOverlay(name="ASTER Quartz", image=png_file, bounds=[[south, west], [north, east]], opacity=0.7, interactive=True, cross_origin=False, zindex=1)
    image_overlay.add_to(m)

    # Add vector layers
    vector_layers = data.get("vector_layers", np.array([], dtype=object))
    for lname in vector_layers.tolist():
        key = f"vector__{lname.replace(':', '__')}"
        if key in data:
            geojson_str = data[key].tolist() if hasattr(data[key], 'tolist') else data[key]
            try:
                geojson_obj = json.loads(geojson_str)
                folium.GeoJson(geojson_obj, name=lname).add_to(m)
            except Exception:
                # If parsing fails, skip
                pass

    folium.LayerControl().add_to(m)
    m.save(out_map)
    print(f"Saved interactive map to {out_map}")


if __name__ == "__main__":
    try:
        build_map()
    except Exception as e:
        print(f"Error: {e}")
        print("If folium is missing, install: pip install folium")
        print("For better PNG rendering install pillow and matplotlib: pip install pillow matplotlib")
