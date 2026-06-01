import show, numpy as np
layers = show.load_all_layers()
for l in layers:
    img = l['img']
    finite_count = int(np.isfinite(img).sum())
    total = img.size
    print(l['name'], 'img_shape=', img.shape, 'finite=', finite_count, f'{finite_count/total:.2%}', 'crs=', l.get('crs'))
