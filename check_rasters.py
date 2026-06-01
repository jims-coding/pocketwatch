import rasterio, numpy as np, glob, os
files=sorted(glob.glob('output/rasters/*.tif'))
if not files:
    print('No rasters found')
for p in files:
    try:
        with rasterio.open(p) as ds:
            arr=ds.read(1)
            nod=ds.nodata
            finite=np.isfinite(arr) & (~np.isnan(arr))
            count=int(finite.sum())
            total=arr.size
            print(os.path.basename(p), 'crs=', ds.crs, 'shape=', arr.shape, 'nodata=', nod, 'finite=', count, f'{count/total:.2%}')
    except Exception as e:
        print('ERR', p, e)
