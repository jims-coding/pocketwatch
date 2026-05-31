from script import WAExplorationPipeline
import os

pipeline = WAExplorationPipeline(target_epsg=7851)
# target location used previously
lat, lon = -30.7766, 121.5065
bbox = pipeline.calculate_bbox(lat, lon)

# selected coverages (medians)
desired = [
    "ml_nat_conductivity__national_0_4m_conductivity_prediction_median",
    "ml_nat_conductivity__national_30m_conductivity_prediction_median",
]

# discover oxide median coverages from GetCapabilities and pick those with _prediction_median
all_covs = pipeline.list_wcs_coverages()
for cov in all_covs:
    if cov.startswith("ml_oxides__") and cov.endswith("_prediction_median"):
        desired.append(cov)

os.makedirs('output/rasters', exist_ok=True)

# pull each coverage and save
for cov in desired:
    print('Fetching', cov)
    info = pipeline.pull_wcs_coverage(bbox, cov)
    if info is None:
        print(' -> failed:', cov)
        continue
    out_path = os.path.join('output','rasters', f"{cov}.tif")
    try:
        pipeline.save_raster_geotiff(info, out_path=out_path)
    except Exception as e:
        print(' -> save failed', e)

# save occurrences as geojson
payload = pipeline.execute_pipeline(lat, lon, [pipeline.select_mineral_occurrence_layer()])
if 'vector' in payload and payload['vector']:
    for lname, gdf in payload['vector'].items():
        safe = lname.replace(':','__')
        outv = os.path.join('output', f"vector_{safe}.geojson")
        try:
            if gdf is None or gdf.empty:
                continue
            gdf.to_file(outv, driver='GeoJSON')
            print('Saved vector', outv)
        except Exception as e:
            print('Failed to save vector', e)
