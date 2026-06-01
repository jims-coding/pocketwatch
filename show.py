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


def lonlat_to_web_mercator(lon, lat):
    tr = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return tr.transform(lon, lat)


def load_raster_layer(path):
    if not os.path.exists(path):
        return None
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype(np.float32)
        if ds.nodata is not None:
            arr = np.where(arr == ds.nodata, np.nan, arr)
        src_crs = ds.crs
        src_transform = ds.transform
        if str(src_crs).upper() != "EPSG:3857":
            dst_transform, dst_w, dst_h = calculate_default_transform(src_crs, "EPSG:3857", ds.width, ds.height, *ds.bounds)
            dst = np.full((dst_h, dst_w), np.nan, np.float32)
            reproject(source=arr, destination=dst, src_transform=src_transform, src_crs=src_crs, dst_transform=dst_transform, dst_crs="EPSG:3857", resampling=Resampling.bilinear, src_nodata=ds.nodata, dst_nodata=np.nan)
            arr3857 = dst
            transform3857 = dst_transform
        else:
            arr3857 = arr
            transform3857 = src_transform

    if np.isfinite(arr3857).sum() == 0:
        return None

    minx, miny, maxx, maxy = array_bounds(arr3857.shape[0], arr3857.shape[1], transform3857)
    return {
        "name": os.path.splitext(os.path.basename(path))[0],
        "raw": arr3857,
        "img": np.flipud(arr3857),
        "transform": transform3857,
        "x": minx,
        "y": miny,
        "dw": maxx - minx,
        "dh": maxy - miny,
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
    mask = np.isfinite(base["raw"]).astype(np.uint8)
    for lyr in raster_layers[1:]:
        src = np.where(np.isfinite(lyr["raw"]), 1, 0).astype(np.uint8)
        warped = np.zeros_like(mask)
        reproject(source=src, destination=warped, src_transform=lyr["transform"], src_crs="EPSG:3857", dst_transform=base["transform"], dst_crs="EPSG:3857", resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
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
    if crs is not None and str(crs).upper() != "EPSG:4326":
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
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
    if not raster_layers:
        # fallback: try to reproject probability tif for display
        if os.path.exists(PROB_TIF):
            arr, x, y, dw, dh, bbox = load_prob_for_display(PROB_TIF)
            raster_layers = [{"name": os.path.splitext(os.path.basename(PROB_TIF))[0], "raw": np.flipud(arr), "img": arr, "transform": None, "x": x, "y": y, "dw": dw, "dh": dh}]
        else:
            raster_layers = []

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
        if str(crs).upper() != "EPSG:3857":
            dst_transform, dst_w, dst_h = calculate_default_transform(crs, "EPSG:3857", ds.width, ds.height, *ds.bounds)
            dst = np.full((dst_h, dst_w), np.nan, np.float32)
            reproject(source=arr, destination=dst, src_transform=ds.transform, src_crs=crs, dst_transform=dst_transform, dst_crs="EPSG:3857", resampling=Resampling.nearest, src_nodata=ds.nodata, dst_nodata=np.nan)
            arrm = np.flipud(dst)
            minx, miny, maxx, maxy = rasterio.transform.array_bounds(dst_h, dst_w, dst_transform)
        else:
            arrm = np.flipud(arr)
            minx, miny, maxx, maxy = rasterio.transform.array_bounds(ds.height, ds.width, ds.transform)
        return arrm, minx, miny, maxx - minx, maxy - miny, transform_bounds(str(crs), "EPSG:4326", *ds.bounds)


if __name__ == "__main__":
    build_map()
