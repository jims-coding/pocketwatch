import io,sys
p='output/map_pipeline.html'
try:
    with open(p,'r',encoding='utf-8') as f:
        txt=f.read()
except Exception as e:
    print('ERR open',e); sys.exit(1)
for name in ['ml_oxides__iron_oxide_prediction_median','ml_oxides__silicon_oxide_prediction_median']:
    print(name, 'found' if name in txt else 'MISSING')
# print first 2000 chars for manual inspection
print('\n--- snippet ---\n')
print(txt[:2000])
