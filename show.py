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
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

    raster_renderers = []
    raster_labels = []
    for layer in raster_layers:
        display_array = layer["array"]
        finite = display_array[np.isfinite(display_array)]
        if finite.size == 0:
            continue
        low = float(np.nanpercentile(finite, 2))
        high = float(np.nanpercentile(finite, 98))
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            low = float(np.nanmin(finite))
            high = float(np.nanmax(finite))
        mapper = LinearColorMapper(palette=Turbo256, low=low, high=high, nan_color="#00000000")
        renderer = p.image(
            image=[display_array],
            x=layer["x"],
            y=layer["y"],
            dw=layer["dw"],
            dh=layer["dh"],
            color_mapper=mapper,
            alpha=0.55,
            visible=False,
        )
        raster_renderers.append(renderer)
        raster_labels.append(layer["name"])

    overlap_renderer = None
    if overlap_layer is not None:
        overlap_renderer = p.image(
            image=[overlap_layer["array"]],
            x=overlap_layer["x"],
            y=overlap_layer["y"],
            dw=overlap_layer["dw"],
            dh=overlap_layer["dh"],
            color_mapper=LinearColorMapper(palette=["#00000000", "#00c853"], low=0.0, high=1.0, nan_color="#00000000"),
            alpha=0.22,
            visible=True,
        )

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

    # Build default active indices: make overlap mask visible by default,
    # keep TP points, Live gold and Local gold visible.
    labels = raster_labels + (["Overlap mask"] if overlap_renderer is not None else []) + ["TP points", "Live gold", "Local gold"]
    active = []
    base = len(raster_labels)
    if overlap_renderer is not None:
        # overlap index is `base`
        active.append(base)
    # indices for TP points and the two gold layers come after the raster labels and optional overlap
    tp_index = base + (1 if overlap_renderer is not None else 0)
    active.extend([tp_index, tp_index + 1, tp_index + 2])
    layer_controls = CheckboxGroup(labels=labels, active=active)
    layer_controls.js_on_change(
        "active",
        CustomJS(
            args=dict(
                raster_renderers=raster_renderers,
                overlap_renderer=overlap_renderer,
                tp_points_renderer=tp_points_renderer,
                gold_renderer=gold_renderer,
                local_renderer=local_renderer,
            ),
            code="""
                for (let i = 0; i < raster_renderers.length; i++) {
                    raster_renderers[i].visible = cb_obj.active.includes(i);
                }
                let base = raster_renderers.length;
                if (overlap_renderer !== null) {
                    overlap_renderer.visible = cb_obj.active.includes(base);
                    base += 1;
                }
                tp_points_renderer.visible = cb_obj.active.includes(base);
                gold_renderer.visible = cb_obj.active.includes(base + 1);
                local_renderer.visible = cb_obj.active.includes(base + 2);
            """,
        ),
    )
        "name": os.path.splitext(os.path.basename(tif_path))[0],
        "raw_array": dst_arr,
        "array": np.flipud(dst_arr),
        "transform_3857": dst_transform,
        "crs_3857": "EPSG:3857",
        "x": minx,
        "y": miny,
        "dw": maxx - minx,
        "dh": maxy - miny,
        "bounds_4326": bounds_4326,
    }


def load_all_raster_layers(rasters_dir):
    layers = []
    if not os.path.isdir(rasters_dir):
        return layers

    for fn in sorted(os.listdir(rasters_dir)):
        if not fn.lower().endswith((".tif", ".tiff")):
            continue
        layer = None
        try:
            layer = load_raster_layer(os.path.join(rasters_dir, fn))
        except Exception:
            layer = None
        if layer is not None:
            layers.append(layer)
    return layers


def build_overlap_mask_layer(raster_layers):
    if not raster_layers:
        return None

    template = raster_layers[0]
    template_arr = template.get("raw_array", template["array"])
    template_shape = template_arr.shape
    template_transform = template["transform_3857"]

    overlap_mask = np.isfinite(template_arr).astype(np.uint8)
    for layer in raster_layers[1:]:
        src = np.where(np.isfinite(layer.get("raw_array", layer["array"])), 1, 0).astype(np.uint8)
        warped = np.zeros(template_shape, dtype=np.uint8)
        try:
            reproject(
                source=src,
                destination=warped,
                src_transform=layer["transform_3857"],
                src_crs=layer.get("crs_3857", "EPSG:3857"),
                dst_transform=template_transform,
                dst_crs="EPSG:3857",
                resampling=Resampling.nearest,
                src_nodata=0,
                dst_nodata=0,
            )
        except Exception:
            return None
        overlap_mask = overlap_mask & (warped > 0)

    if overlap_mask.sum() == 0:
        return None

    mask_arr = overlap_mask.astype(np.float32)
    finite = mask_arr[np.isfinite(mask_arr)]
    minx, miny, maxx, maxy = array_bounds(template_shape[0], template_shape[1], template_transform)
    return {
        "name": "overlap_mask",
        "raw_array": mask_arr,
        "array": np.flipud(mask_arr),
        "transform_3857": template_transform,
        "crs_3857": "EPSG:3857",
        "x": minx,
        "y": miny,
        "dw": maxx - minx,
        "dh": maxy - miny,
        "bounds_4326": template.get("bounds_4326"),
    }


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
    raster_layers = load_all_raster_layers(RASTER_DIR)
    if not raster_layers:
        prob_arr, x, y, dw, dh, bbox4326 = load_probability_grid(tif_path)
        raster_layers = [
            {
                "name": os.path.splitext(os.path.basename(tif_path))[0],
                "raw_array": np.flipud(prob_arr),
                "array": prob_arr,
                "x": x,
                "y": y,
                "dw": dw,
                "dh": dh,
                "bounds_4326": bbox4326,
            }
        ]

    overlap_layer = build_overlap_mask_layer(raster_layers)
    bbox4326 = raster_layers[0]["bounds_4326"]
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

    raster_renderers = []
    raster_labels = []
    for layer in raster_layers:
        display_array = layer["array"]
        finite = display_array[np.isfinite(display_array)]
        if finite.size == 0:
            continue
        low = float(np.nanpercentile(finite, 2))
        high = float(np.nanpercentile(finite, 98))
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            low = float(np.nanmin(finite))
            high = float(np.nanmax(finite))
        mapper = LinearColorMapper(palette=Turbo256, low=low, high=high, nan_color="#00000000")
            renderer = p.image(
            image=[display_array],
            x=layer["x"],
            y=layer["y"],
            dw=layer["dw"],
            dh=layer["dh"],
            color_mapper=mapper,
            alpha=0.55,
                visible=False,
        )
        raster_renderers.append(renderer)
        raster_labels.append(layer["name"])

    overlap_renderer = None
    if overlap_layer is not None:
        overlap_renderer = p.image(
            image=[overlap_layer["array"]],
            x=overlap_layer["x"],
            y=overlap_layer["y"],
            dw=overlap_layer["dw"],
            dh=overlap_layer["dh"],
            color_mapper=LinearColorMapper(palette=["#00000000", "#00c853"], low=0.0, high=1.0, nan_color="#00000000"),
            alpha=0.22,
                visible=True,
        )

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

        # Build default active indices: make overlap mask visible by default,
        # keep TP points, Live gold and Local gold visible.
        labels = raster_labels + ([("Overlap mask")] if overlap_renderer is not None else []) + ["TP points", "Live gold", "Local gold"]
        active = []
        base = len(raster_labels)
        if overlap_renderer is not None:
            # overlap index is `base`
            active.append(base)
        # indices for TP points and the two gold layers come after the raster labels and optional overlap
        tp_index = base + (1 if overlap_renderer is not None else 0)
        active.extend([tp_index, tp_index + 1, tp_index + 2])
        layer_controls = CheckboxGroup(labels=labels, active=active)
    layer_controls.js_on_change(
        "active",
        CustomJS(
            args=dict(
                raster_renderers=raster_renderers,
                overlap_renderer=overlap_renderer,
                <b>Exported files:</b> output/probability_grid.tif, output/high_confidence_grid.tif, output/overlap_mask.tif<br>
                tp_points_renderer=tp_points_renderer,
                gold_renderer=gold_renderer,
                local_renderer=local_renderer,
            ),
            code="""
                for (let i = 0; i < raster_renderers.length; i++) {
                    raster_renderers[i].visible = cb_obj.active.includes(i);
                }
                let base = raster_renderers.length;
                if (overlap_renderer !== null) {
                    overlap_renderer.visible = cb_obj.active.includes(base);
                    base += 1;
                }
                tp_points_renderer.visible = cb_obj.active.includes(base);
                gold_renderer.visible = cb_obj.active.includes(base + 1);
                local_renderer.visible = cb_obj.active.includes(base + 2);
            """,
        ),
    )

        info = Div(
                text=f"""
                <div style="font-family: sans-serif; font-size: 13px; line-height: 1.4;">
                                        <b>Raster layers:</b> {len(raster_labels)} local GeoTIFFs from output/rasters<br>
                                        <b>Overlap mask:</b> intersection of pixels where every raster layer has data<br>
                                        <b>Exported files:</b> output/probability_grid.tif, output/high_confidence_grid.tif, output/overlap_mask.tif<br>
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
