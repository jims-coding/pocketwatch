import io
import json
import os

import geopandas as gpd
import numpy as np
import rasterio
import requests
from bokeh.io import output_file, save
from bokeh.layouts import column
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    CheckboxGroup,
    HoverTool,
    LinearColorMapper,
    Div,
    WMTSTileSource,
    CustomJS,
)
from bokeh.palettes import Turbo256
from bokeh.plotting import figure
from pyproj import Transformer
from rasterio.transform import Affine, array_bounds


OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROB_TIF_FILE = os.path.join(OUTPUT_DIR, "probability_grid.tif")
MAP_FILE = os.path.join(OUTPUT_DIR, "map.html")
TP_ONLY_TIF_FILE = os.path.join(OUTPUT_DIR, "high_confidence_grid.tif")
TP_ONLY_MAP_FILE = os.path.join(OUTPUT_DIR, "map_tp_only.html")
TP_POINTS_MAP_FILE = os.path.join(OUTPUT_DIR, "map_tp_points.html")
DATA_PKG = os.path.join(OUTPUT_DIR, "dataset_package.npz")
VECTOR_DIR = os.path.join(OUTPUT_DIR, "vectors")

GOV_WFS_URL = "https://geossdi.dmp.wa.gov.au/services/wfs"


def lonlat_to_web_mercator(lon, lat):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return transformer.transform(lon, lat)


def bounds_to_web_mercator(west, south, east, north):
    x0, y0 = lonlat_to_web_mercator(west, south)
    x1, y1 = lonlat_to_web_mercator(east, north)
    return x0, y0, x1, y1


def load_probability_grid(tif_path):
    if not os.path.exists(tif_path):
        raise FileNotFoundError(tif_path)

    with rasterio.open(tif_path) as ds:
        arr = ds.read(1).astype(float)
        if ds.nodata is not None:
            arr = np.where(arr == ds.nodata, np.nan, arr)
        transform = ds.transform
        crs = str(ds.crs) if ds.crs is not None else "EPSG:4326"

    height, width = arr.shape
    minx, miny, maxx, maxy = array_bounds(height, width, transform)

    if crs.upper() != "EPSG:4326":
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        west, south = transformer.transform(minx, miny)
        east, north = transformer.transform(maxx, maxy)
    else:
        west, south, east, north = minx, miny, maxx, maxy

    x0, y0, x1, y1 = bounds_to_web_mercator(west, south, east, north)
    mercator_arr = np.flipud(arr)
    return mercator_arr, x0, y0, x1 - x0, y1 - y0, (west, south, east, north)


def load_tp_points(tif_path):
    if not os.path.exists(tif_path):
        raise FileNotFoundError(tif_path)

    with rasterio.open(tif_path) as ds:
        arr = ds.read(1)
        transform = ds.transform
        crs = ds.crs

    mask = np.isfinite(arr) & (arr > 0)
    rows, cols = np.where(mask)
    if rows.size == 0:
        return ColumnDataSource(data={"x": [], "y": [], "value": []})

    xs = []
    ys = []
    values = []
    transformer = None
    if crs is not None and str(crs).upper() != "EPSG:4326":
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    for row, col in zip(rows.tolist(), cols.tolist()):
        x_raw, y_raw = rasterio.transform.xy(transform, row, col, offset="center")
        if transformer is not None:
            lon, lat = transformer.transform(x_raw, y_raw)
        else:
            lon, lat = x_raw, y_raw
        x_web, y_web = lonlat_to_web_mercator(lon, lat)
        xs.append(x_web)
        ys.append(y_web)
        values.append(float(arr[row, col]))

    return ColumnDataSource(data={"x": xs, "y": ys, "value": values})


def fetch_gold_points(bbox4326):
    west, south, east, north = bbox4326
    params = {
        "service": "WFS",
        "version": "1.0.0",
        "request": "GetFeature",
        "typeName": "mo:MinOccView",
        "bbox": f"{west},{south},{east},{north}",
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
    }
    response = requests.get(GOV_WFS_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    xs = []
    ys = []
    names = []
    commodities = []
    ids = []

    for feature in payload.get("features", []):
        props = feature.get("properties", {}) or {}
        commodity = str(props.get("commodity", ""))
        if not ("Au" in commodity or "gold" in commodity.lower()):
            continue

        geometry = feature.get("geometry", {}) or {}
        coords = geometry.get("coordinates")
        if not coords or len(coords) < 2:
            continue

        lon, lat = coords[0], coords[1]
        x, y = lonlat_to_web_mercator(lon, lat)
        xs.append(x)
        ys.append(y)
        names.append(props.get("name", ""))
        commodities.append(commodity)
        ids.append(props.get("id", feature.get("id", "")))

    return ColumnDataSource(
        data={
            "x": xs,
            "y": ys,
            "name": names,
            "commodity": commodities,
            "id": ids,
        }
    )


def load_local_dataset_points(vectors_dir):
    if not os.path.isdir(vectors_dir):
        return ColumnDataSource(data={"x": [], "y": [], "name": [], "commodity": [], "id": []})

    xs = []
    ys = []
    names = []
    commodities = []
    ids = []

    for fn in sorted(os.listdir(vectors_dir)):
        if not fn.lower().endswith((".geojson", ".json")):
            continue
        path = os.path.join(vectors_dir, fn)
        try:
            gdf = gpd.read_file(path)
        except Exception:
            continue
        if gdf.empty:
            continue
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        else:
            gdf = gdf.to_crs(epsg=4326)

        if "commodity" in gdf.columns:
            commodity_series = gdf["commodity"].astype(str)
            gold_mask = commodity_series.str.contains(r"\bAu\b|gold", case=False, regex=True, na=False)
        else:
            gold_mask = np.ones(len(gdf), dtype=bool)

        gold_gdf = gdf.loc[gold_mask].copy()
        if gold_gdf.empty:
            continue

        for _, row in gold_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            xs.append(geom.x)
            ys.append(geom.y)
            names.append(row.get("name", ""))
            commodities.append(row.get("commodity", ""))
            ids.append(row.get("id", ""))

    if xs:
        xs, ys = zip(*[lonlat_to_web_mercator(x, y) for x, y in zip(xs, ys)])
    else:
        xs, ys = [], []

    return ColumnDataSource(
        data={
            "x": list(xs),
            "y": list(ys),
            "name": names,
            "commodity": commodities,
            "id": ids,
        }
    )


def build_map(out_map=TP_POINTS_MAP_FILE, tif_path=TP_ONLY_TIF_FILE):
    prob_arr, x, y, dw, dh, bbox4326 = load_probability_grid(tif_path)
    tp_points_source = load_tp_points(tif_path)
    gold_source = fetch_gold_points(bbox4326)
    local_source = load_local_dataset_points(VECTOR_DIR)

    output_file(out_map, title="TP-only Prospectivity Map")

    p = figure(
        x_axis_type="mercator",
        y_axis_type="mercator",
        width=1200,
        height=900,
        title="TP-only Grid and Known Gold Occurrences",
        tools="pan,wheel_zoom,reset,save",
        active_scroll="wheel_zoom",
    )
    p.add_tile(WMTSTileSource(url="https://tile.openstreetmap.org/{Z}/{X}/{Y}.png"))

    mapper = LinearColorMapper(palette=Turbo256, low=0.0, high=1.0, nan_color="#00000000")
    tp_image_renderer = p.image(image=[prob_arr], x=x, y=y, dw=dw, dh=dh, color_mapper=mapper, alpha=0.55)

    tp_points_renderer = p.scatter(
        x="x",
        y="y",
        source=tp_points_source,
        marker="diamond",
        size=8,
        color="#d81b60",
        line_color="#880e4f",
        fill_alpha=0.9,
        legend_label="High-confidence TP points",
    )

    gold_renderer = p.scatter(
        x="x",
        y="y",
        source=gold_source,
        marker="circle",
        size=8,
        color="#ffd700",
        line_color="#7a5f00",
        fill_alpha=0.95,
        legend_label="Known gold occurrences (live WFS)",
    )

    local_renderer = p.scatter(
        x="x",
        y="y",
        source=local_source,
        marker="triangle",
        size=7,
        color="#00b894",
        line_color="#00695c",
        fill_alpha=0.80,
        legend_label="Local dataset gold points",
    )

    hover = HoverTool(
        renderers=[gold_renderer, local_renderer],
        tooltips=[
            ("Name", "@name"),
            ("Commodity", "@commodity"),
            ("ID", "@id"),
        ],
    )
    p.add_tools(hover)

    color_bar = ColorBar(color_mapper=mapper, ticker=BasicTicker(desired_num_ticks=6), label_standoff=10)
    p.add_layout(color_bar, "right")

    layer_controls = CheckboxGroup(
        labels=["TP raster", "TP points", "Live gold", "Local gold"],
        active=[0, 1, 2, 3],
    )
    layer_controls.js_on_change(
        "active",
        CustomJS(
            args=dict(
                tp_image_renderer=tp_image_renderer,
                tp_points_renderer=tp_points_renderer,
                gold_renderer=gold_renderer,
                local_renderer=local_renderer,
            ),
            code="""
                tp_image_renderer.visible = cb_obj.active.includes(0);
                tp_points_renderer.visible = cb_obj.active.includes(1);
                gold_renderer.visible = cb_obj.active.includes(2);
                local_renderer.visible = cb_obj.active.includes(3);
            """,
        ),
    )

    info = Div(
        text=f"""
        <div style="font-family: sans-serif; font-size: 13px; line-height: 1.4;">
                    <b>Reference grid:</b> high_confidence_grid.tif<br>
                                        <b>Local dataset:</b> output/vectors/*.geojson as the green triangle layer<br>
          <b>Government layer:</b> live WFS `mo:MinOccView` filtered to Au/gold within the raster bounds<br>
          <b>Map file:</b> {out_map}
        </div>
        """,
        width=1200,
    )

    layout = column(info, layer_controls, p)
    save(layout)
    print(f"Saved layered map to {out_map}")


if __name__ == "__main__":
    build_map()
