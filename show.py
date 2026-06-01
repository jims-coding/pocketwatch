import os
import numpy as np
import rasterio
import requests
import geopandas as gpd
from pyproj import Transformer
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from bokeh.plotting import figure
from bokeh.io import output_file, save
from bokeh.models import ColumnDataSource, CheckboxGroup, HoverTool, LinearColorMapper, Div, WMTSTileSource, CustomJS
from bokeh.layouts import column
from bokeh.palettes import Turbo256


OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RASTER_DIR = os.path.join(OUTPUT_DIR, "rasters")
VECTOR_DIR = os.path.join(OUTPUT_DIR, "vectors")
PROB_TIF = os.path.join(OUTPUT_DIR, "probability_grid.tif")
TP_TIF = os.path.join(OUTPUT_DIR, "high_confidence_grid.tif")

# Canonical CRS objects (avoid hardcoded string comparisons)
from rasterio.crs import CRS
GEO_CRS = CRS.from_string("EPSG:4326")
DISPLAY_CRS = CRS.from_string("EPSG:3857")
# reuse transformer for lon/lat -> web mercator
_WEB_MERCATOR_TRANSFORMER = Transformer.from_crs(GEO_CRS, DISPLAY_CRS, always_xy=True)


def lonlat_to_web_mercator(lon, lat):
    return _WEB_MERCATOR_TRANSFORMER.transform(lon, lat)


def load_raster_layer(path):
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        if ds.nodata is not None:
            arr = np.where(arr == ds.nodata, np.nan, arr)
        src_crs = ds.crs if ds.crs is not None else GEO_CRS
        src_transform = ds.transform
        # compute original bounds in 4326 for external queries
        try:
            src_bounds_4326 = transform_bounds(src_crs, GEO_CRS, ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top, densify_pts=21)
        except Exception:
            src_bounds_4326 = None
        reprojected = False
        if src_crs != DISPLAY_CRS:
            dst_transform, dst_w, dst_h = calculate_default_transform(src_crs, DISPLAY_CRS, ds.width, ds.height, *ds.bounds)
            dst = np.full((dst_h, dst_w), np.nan, np.float32)
            reproject(source=arr, destination=dst, src_transform=src_transform, src_crs=src_crs, dst_transform=dst_transform, dst_crs=DISPLAY_CRS, resampling=Resampling.bilinear, src_nodata=ds.nodata, dst_nodata=np.nan)
            arr3857 = dst
            transform3857 = dst_transform
            reprojected = True
        else:
            arr3857 = arr
            transform3857 = src_transform

        # If we reprojected the array, its CRS is now the display CRS
        out_crs = DISPLAY_CRS if reprojected else src_crs

    if np.isfinite(arr3857).sum() == 0:
        return None

    minx, miny, maxx, maxy = array_bounds(arr3857.shape[0], arr3857.shape[1], transform3857)

    return {
        "name": os.path.splitext(os.path.basename(path))[0],
        "raw": arr3857,
        "img": np.flipud(arr3857),
        "transform": transform3857,
        "crs": out_crs,
        "x": minx,
        "y": miny,
        "dw": maxx - minx,
        "dh": maxy - miny,
        "bounds_4326": src_bounds_4326,
    }


def load_all_layers():
    layers = []
    if not os.path.isdir(RASTER_DIR):
        return layers
    for fn in sorted(os.listdir(RASTER_DIR)):
        if fn.lower().endswith((".tif", ".tiff")):
            layer = load_raster_layer(os.path.join(RASTER_DIR, fn))
            if layer is not None:
                layers.append(layer)
    return layers


def build_overlap(raster_layers):
    if not raster_layers:
        return None
    base = raster_layers[0]
    base_crs = base.get('crs', DISPLAY_CRS)
    mask = np.isfinite(base["raw"]).astype(np.uint8)
    for lyr in raster_layers[1:]:
        src = np.where(np.isfinite(lyr["raw"]), 1, 0).astype(np.uint8)
        warped = np.zeros_like(mask)
        src_crs = lyr.get('crs', DISPLAY_CRS)
        dst_crs = base_crs
        reproject(source=src, destination=warped, src_transform=lyr["transform"], src_crs=src_crs, dst_transform=base["transform"], dst_crs=dst_crs, resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
        mask = mask & (warped > 0)
    if mask.sum() == 0:
        return None
    return {
        "name": "overlap_mask",
        "raw": mask.astype(np.float32),
        "img": np.flipud(mask.astype(np.float32)),
        "transform": base["transform"],
        "x": base["x"],
        "y": base["y"],
        "dw": base["dw"],
        "dh": base["dh"],
    }


def load_tp_points(path):
    if not os.path.exists(path):
        return ColumnDataSource(data={"x": [], "y": [], "value": []})
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        tr = ds.transform
        crs = ds.crs
    mask = np.isfinite(arr) & (arr > 0)
    rows, cols = np.where(mask)
    xs, ys = [], []
    transformer = None
    if crs is not None and crs != GEO_CRS:
        transformer = Transformer.from_crs(crs, GEO_CRS, always_xy=True)
    for r, c in zip(rows, cols):
        xraw, yraw = rasterio.transform.xy(tr, r, c, offset="center")
        if transformer is not None:
            lon, lat = transformer.transform(xraw, yraw)
        else:
            lon, lat = xraw, yraw
        xw, yw = lonlat_to_web_mercator(lon, lat)
        xs.append(xw)
        ys.append(yw)
    return ColumnDataSource(data={"x": xs, "y": ys})


def fetch_gold(bbox4326):
    west, south, east, north = bbox4326
    params = {"service": "WFS", "version": "1.0.0", "request": "GetFeature", "typeName": "mo:MinOccView", "bbox": f"{west},{south},{east},{north}", "outputFormat": "application/json", "srsName": "EPSG:4326"}
    try:
        r = requests.get("https://geossdi.dmp.wa.gov.au/services/wfs", params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()
    except Exception:
        return ColumnDataSource(data={"x": [], "y": [], "name": [], "commodity": [], "id": []})
    xs, ys, names, comms, ids = [], [], [], [], []
    for f in payload.get("features", []):
        props = f.get("properties", {}) or {}
        commodity = str(props.get("commodity", ""))
        if not ("Au" in commodity or "gold" in commodity.lower()):
            continue
        geom = f.get("geometry", {}) or {}
        coords = geom.get("coordinates")
        if not coords or len(coords) < 2:
            continue
        lon, lat = coords[0], coords[1]
        xw, yw = lonlat_to_web_mercator(lon, lat)
        xs.append(xw); ys.append(yw)
        names.append(props.get("name", ""))
        comms.append(commodity)
        ids.append(props.get("id", ""))
    return ColumnDataSource(data={"x": xs, "y": ys, "name": names, "commodity": comms, "id": ids})


def build_map(out_map=os.path.join(OUTPUT_DIR, "map.html"), tif_path=TP_TIF):
    """Build an interactive map.

    Accepts `tif_path` for compatibility with pipeline.py.
    """
    # normalize kw name used by other callers
    tp_tif = tif_path

    raster_layers = load_all_layers()
    # Prefer the probability grid as the canonical base for display alignment
    if os.path.exists(PROB_TIF):
        prob_layer = load_raster_layer(PROB_TIF)
        if prob_layer is not None:
            # ensure prob_layer is first in the list for alignment
            # remove any existing entry with the same name
            raster_layers = [l for l in raster_layers if l.get('name') != prob_layer.get('name')]
            raster_layers.insert(0, prob_layer)
    if not raster_layers:
        # fallback: try to reproject probability tif for display
        if os.path.exists(PROB_TIF):
            arr, x, y, dw, dh, bbox = load_prob_for_display(PROB_TIF)
            raster_layers = [{"name": os.path.splitext(os.path.basename(PROB_TIF))[0], "raw": np.flipud(arr), "img": arr, "transform": None, "x": x, "y": y, "dw": dw, "dh": dh}]
        else:
            raster_layers = []

    # Align all raster layers to a common base grid (first layer) in DISPLAY_CRS
    if raster_layers:
        base = raster_layers[0]
        base_transform = base['transform']
        base_shape = base['raw'].shape
        base_transform_tuple = tuple(base_transform)
        for lyr in raster_layers[1:]:
            try:
                # If already aligned, skip
                lyr_transform_tuple = tuple(lyr['transform']) if not isinstance(lyr['transform'], tuple) else lyr['transform']
                if lyr_transform_tuple == base_transform_tuple and lyr['raw'].shape == base_shape:
                    continue
                dst = np.full(base_shape, np.nan, dtype=np.float32)
                src_crs = lyr.get('crs', DISPLAY_CRS)
                src_nodata = None
                # attempt to reuse nodata when present
                if hasattr(lyr, 'get'):
                    # lyr may be a dict from load_raster_layer; it doesn't include nodata, so default to None
                    src_nodata = None
                reproject(
                    source=lyr['raw'],
                    destination=dst,
                    src_transform=lyr['transform'],
                    src_crs=src_crs,
                    dst_transform=base_transform,
                    dst_crs=base.get('crs', DISPLAY_CRS),
                    resampling=Resampling.bilinear,
                    src_nodata=src_nodata,
                    dst_nodata=np.nan,
                )
                lyr['raw'] = dst
                lyr['img'] = np.flipud(dst)
                lyr['transform'] = base_transform
                lyr['x'] = base['x']; lyr['y'] = base['y']; lyr['dw'] = base['dw']; lyr['dh'] = base['dh']
            except Exception:
                # best-effort: leave layer as-is if reproject fails
                continue

    overlap = build_overlap(raster_layers)
    bbox = raster_layers[0].get("bounds_4326") if raster_layers else (-180, -90, 180, 90)

    tp_src = load_tp_points(tp_tif) if os.path.exists(tp_tif) else ColumnDataSource(data={"x": [], "y": []})
    gold_src = fetch_gold(bbox)

    output_file(out_map, title="Map")
    p = figure(x_axis_type="mercator", y_axis_type="mercator", width=1000, height=700, tools="pan,wheel_zoom,reset,save")
    p.add_tile(WMTSTileSource(url="https://tile.openstreetmap.org/{Z}/{X}/{Y}.png"))

    raster_renderers = []
    labels = []
    for lyr in raster_layers:
        img = lyr["img"]
        finite = img[np.isfinite(img)]
        if finite.size == 0:
            continue
        low = float(np.nanpercentile(finite, 2))
        high = float(np.nanpercentile(finite, 98))
        mapper = LinearColorMapper(palette=Turbo256, low=low, high=high, nan_color="#00000000")
        r = p.image(image=[img], x=lyr["x"], y=lyr["y"], dw=lyr["dw"], dh=lyr["dh"], color_mapper=mapper, alpha=0.6, visible=False)
        raster_renderers.append(r)
        labels.append(lyr["name"])

    overlap_renderer = None
    if overlap is not None:
        overlap_renderer = p.image(image=[overlap["img"]], x=overlap["x"], y=overlap["y"], dw=overlap["dw"], dh=overlap["dh"], color_mapper=LinearColorMapper(palette=["#00000000","#00c853"], low=0, high=1), alpha=0.25, visible=True)

    tp_r = p.scatter(x="x", y="y", source=tp_src, marker="diamond", size=8, color="#d81b60")
    gold_r = p.scatter(x="x", y="y", source=gold_src, marker="circle", size=8, color="#ffd700")

    labels_all = labels + (["Overlap mask"] if overlap_renderer is not None else []) + ["TP points", "Gold WFS"]
    base = len(labels)
    active = []
    if overlap_renderer is not None:
        active.append(base)
    active.extend([base + (1 if overlap_renderer is not None else 0), base + (2 if overlap_renderer is not None else 1)])

    controls = CheckboxGroup(labels=labels_all, active=active)
    controls.js_on_change("active", CustomJS(args=dict(raster_renderers=raster_renderers, overlap_renderer=overlap_renderer, tp=tp_r, gold=gold_r), code="""
        for (let i=0;i<raster_renderers.length;i++){ raster_renderers[i].visible = cb_obj.active.includes(i); }
        let base = raster_renderers.length;
        if (overlap_renderer !== null){ overlap_renderer.visible = cb_obj.active.includes(base); base += 1; }
        tp.visible = cb_obj.active.includes(base);
        gold.visible = cb_obj.active.includes(base+1);
    """))

    info = Div(text=f"<b>Raster layers:</b> {len(labels)} &nbsp; <b>Exports:</b> output/probability_grid.tif, output/high_confidence_grid.tif, output/overlap_mask.tif", width=1000)

    save(column(info, controls, p))


def load_prob_for_display(tif_path):
    # minimal helper: reproject probability tif to mercator for display
    with rasterio.open(tif_path) as ds:
        arr = ds.read(1).astype(np.float32)
        if ds.nodata is not None:
            arr = np.where(arr == ds.nodata, np.nan, arr)
        crs = ds.crs
        if crs != DISPLAY_CRS:
            dst_transform, dst_w, dst_h = calculate_default_transform(crs, DISPLAY_CRS, ds.width, ds.height, *ds.bounds)
            dst = np.full((dst_h, dst_w), np.nan, np.float32)
            reproject(source=arr, destination=dst, src_transform=ds.transform, src_crs=crs, dst_transform=dst_transform, dst_crs=DISPLAY_CRS, resampling=Resampling.nearest, src_nodata=ds.nodata, dst_nodata=np.nan)
            arrm = np.flipud(dst)
            minx, miny, maxx, maxy = rasterio.transform.array_bounds(dst_h, dst_w, dst_transform)
        else:
            arrm = np.flipud(arr)
            minx, miny, maxx, maxy = rasterio.transform.array_bounds(ds.height, ds.width, ds.transform)
        return arrm, minx, miny, maxx - minx, maxy - miny, transform_bounds(crs, GEO_CRS, *ds.bounds)


if __name__ == "__main__":
    build_map()
