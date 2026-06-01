import show
layers = show.load_all_layers()
print('Loaded layers:')
for l in layers:
    print('-', l['name'], 'crs=', l.get('crs'), 'shape=', l['raw'].shape)
# Also simulate build_map's label extraction without writing file
raster_layers = layers
if show.PROB_TIF and show.os.path.exists(show.PROB_TIF):
    prob = show.load_raster_layer(show.PROB_TIF)
    if prob:
        raster_layers = [l for l in raster_layers if l.get('name') != prob.get('name')]
        raster_layers.insert(0, prob)
labels = [l['name'] for l in raster_layers]
print('Labels:', labels)
