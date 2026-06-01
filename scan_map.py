p='output/map_pipeline.html'
with open(p,'r',encoding='utf-8') as f:
    txt=f.read()
names=['probability_grid','aster_quartz','ml__radmap_v4_2019_filtered_ML_KThU_merged_3band','ml__radmap_v4_2019_filtered_ML_pctk','ml_nat_conductivity__national_0_4m_conductivity_prediction_median','ml_oxides__iron_oxide_prediction_median','ml_oxides__silicon_oxide_prediction_median']
for n in names:
    print(n, 'FOUND' if n in txt else 'MISSING')
