import show, numpy as np
from rasterio.transform import array_bounds

raster_layers = show.load_all_layers()
if show.PROB_TIF and show.os.path.exists(show.PROB_TIF):
    prob_layer = show.load_raster_layer(show.PROB_TIF)
    if prob_layer is not None:
        raster_layers = [l for l in raster_layers if l.get('name') != prob_layer.get('name')]
        raster_layers.insert(0, prob_layer)

print('Initial count:', len(raster_layers))
base = raster_layers[0] if raster_layers else None
if base:
    base_transform = base['transform']
    base_shape = base['raw'].shape
    base_transform_tuple = tuple(base_transform)
    for lyr in raster_layers[1:]:
        print('\nProcessing layer:', lyr['name'])
        try:
            lyr_transform_tuple = tuple(lyr['transform']) if not isinstance(lyr['transform'], tuple) else lyr['transform']
            print(' lyr transform tuple length:', len(lyr_transform_tuple))
            if lyr_transform_tuple == base_transform_tuple and lyr['raw'].shape == base_shape:
                print('  already aligned, skip')
                continue
            dst = np.full(base_shape, np.nan, dtype=np.float32)
            src_crs = lyr.get('crs', show.DISPLAY_CRS)
            src_nodata = None
            if hasattr(lyr, 'get'):
                src_nodata = None
            show.reproject(
                source=lyr['raw'],
                destination=dst,
                src_transform=lyr['transform'],
                src_crs=src_crs,
                dst_transform=base_transform,
                dst_crs=base.get('crs', show.DISPLAY_CRS),
                resampling=show.Resampling.bilinear,
                src_nodata=src_nodata,
                dst_nodata=np.nan,
            )
            print('  reproject succeeded, dst shape', dst.shape, 'finite', int(np.isfinite(dst).sum()))
            lyr['raw'] = dst
            lyr['img'] = np.flipud(dst)
            lyr['transform'] = base_transform
            lyr['x'] = base['x']; lyr['y'] = base['y']; lyr['dw'] = base['dw']; lyr['dh'] = base['dh']
        except Exception as e:
            print('  reproject failed:', e)
            continue

# Now compute labels like build_map
labels = []
for lyr in raster_layers:
    img = lyr['img']
    finite = img[np.isfinite(img)]
    print('Layer', lyr['name'], 'finite pixels:', finite.size)
    if finite.size == 0:
        print('  -> skipped (no finite)')
        continue
    labels.append(lyr['name'])

print('\nFinal labels count:', len(labels))
print('Labels:', labels)
