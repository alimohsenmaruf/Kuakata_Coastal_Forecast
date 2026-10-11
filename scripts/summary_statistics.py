import geopandas as gpd

rate_gdf = gpd.read_file("data/processed/dsas_rate_results.gpkg")

# Exclude extreme estuarine bar outliers for mean calculation
beach_gdf = rate_gdf[(rate_gdf['lrr_m_yr'] >= -20) & (rate_gdf['lrr_m_yr'] <= 20)]

total_transects = len(rate_gdf)
eroding = len(rate_gdf[rate_gdf['lrr_m_yr'] < -1.0])
stable = len(rate_gdf[(rate_gdf['lrr_m_yr'] >= -1.0) & (rate_gdf['lrr_m_yr'] <= 1.0)])
accreting = len(rate_gdf[rate_gdf['lrr_m_yr'] > 1.0])

print("KUAKATA SHORELINE ANALYSIS SUMMARY METRICS")
print(f"Total Evaluated Transects : {total_transects}")
print(f"Eroding Coastline        : {eroding} ({eroding/total_transects*100:.1f}%)")
print(f"Stable Coastline         : {stable} ({stable/total_transects*100:.1f}%)")
print(f"Accreted Coastline       : {accreting} ({accreting/total_transects*100:.1f}%)")
print(f"Average Beach LRR Rate   : {beach_gdf['lrr_m_yr'].mean():.2f} m/year")
print(f"Average Model R² Fit     : {beach_gdf['r2'].mean():.3f}")