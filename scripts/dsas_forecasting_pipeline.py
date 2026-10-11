import os
import glob
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
from scipy.signal import savgol_filter
from scipy.stats import linregress

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
AOI_PATH = os.path.join(PROJECT_DIR, "data", "boundry", "kuakata_study_area.gpkg")

TRANSECT_SPACING = 100.0  # meters
TRANSECT_LENGTH = 1800.0   # meters

def generate_ocean_baseline(aoi_gdf, shoreline_2026_path):
    """
    Generates a clean ocean baseline bounded strictly to the open ocean coast,
    eliminating western spit loops and eastern channel crossovers.
    """
    shoreline_2026 = gpd.read_file(shoreline_2026_path)
    shoreline_clipped = gpd.clip(shoreline_2026, aoi_gdf)
    
    combined_geom = shoreline_clipped.union_all()
    if combined_geom.geom_type == 'MultiLineString':
        main_line = max(list(combined_geom.geoms), key=lambda l: l.length)
    else:
        main_line = combined_geom

    smoothed_line = main_line.simplify(150.0, preserve_topology=True)
    baseline_geom = smoothed_line.parallel_offset(600, side='right', join_style=2)
    
    coords = list(baseline_geom.coords)
    if coords[0][0] > coords[-1][0]:
        coords = coords[::-1]
        
    full_line = LineString(coords)
    total_len = full_line.length
    
    # Topology Domain Bounding:
    # 0.28 start_dist: Trims western spit river-mouth loop
    # 0.93 end_dist: Covers full eastern beach front without crossing the tidal inlet
    start_dist = total_len * 0.28
    end_dist = total_len * 0.93
    
    num_samples = int((end_dist - start_dist) // 50)
    trimmed_pts = [full_line.interpolate(start_dist + i * 50) for i in range(num_samples + 1)]
    ocean_baseline = LineString([(p.x, p.y) for p in trimmed_pts])
        
    return gpd.GeoDataFrame([{'id': 1, 'geometry': ocean_baseline}], crs=aoi_gdf.crs)

def generate_transects(baseline_gdf, spacing=100.0, length=1800.0):
    """Casts perpendicular landward transects along the open-ocean baseline domain."""
    baseline = baseline_gdf.geometry.iloc[0]
    total_length = baseline.length
    num_transects = int(total_length // spacing)
    
    transects, transect_ids = [], []
    for i in range(num_transects):
        distance = i * spacing
        point = baseline.interpolate(distance)
        
        pt1 = baseline.interpolate(max(0, distance - 10))
        pt2 = baseline.interpolate(min(total_length, distance + 10))
        dx, dy = pt2.x - pt1.x, pt2.y - pt1.y
        
        angle = np.arctan2(dy, dx) + np.pi / 2.0
        x2 = point.x + length * np.cos(angle)
        y2 = point.y + length * np.sin(angle)
        
        transects.append(LineString([point, Point(x2, y2)]))
        transect_ids.append(i + 1)
        
    return gpd.GeoDataFrame({'transect_id': transect_ids, 'geometry': transects}, crs=baseline_gdf.crs)

def build_smooth_forecast_line(df_pts):
    """Constructs a continuous, smooth ocean shoreline geometry without loops."""
    df_clean = df_pts.dropna(subset=['pt']).sort_values('transect_id').reset_index(drop=True)
    
    if len(df_clean) < 4:
        return LineString()

    pts = df_clean['pt'].tolist()
    x = [p.x for p in pts]
    y = [p.y for p in pts]
    
    # Savitzky-Golay cartographic line smoothing
    window = min(15, len(x) if len(x) % 2 != 0 else len(x) - 1)
    if window >= 5:
        x = savgol_filter(x, window_length=window, polyorder=2)
        y = savgol_filter(y, window_length=window, polyorder=2)
        
    smooth_pts = [Point(x[i], y[i]) for i in range(len(x))]
    return LineString(smooth_pts)

def run_dsas_analysis(transects_gdf, aoi_gdf):
    shoreline_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "shoreline_*.gpkg")))
    
    shorelines = []
    for f in shoreline_files:
        try:
            yr = int(os.path.basename(f).split("_")[1].split(".")[0])
            gdf = gpd.read_file(f)
            gdf_clean = gpd.clip(gdf, aoi_gdf)
            gdf_clean["year"] = yr
            if not gdf_clean.empty:
                shorelines.append(gdf_clean)
        except Exception:
            continue
            
    print(f"Loaded {len(shorelines)} shoreline epochs.")
    results = []
    
    for idx, transect_row in transects_gdf.iterrows():
        t_id = transect_row['transect_id']
        t_geom = transect_row['geometry']
        base_pt = Point(t_geom.coords[0])
        
        intersections = []
        pt_2026 = None
        
        for sgdf in shorelines:
            yr = sgdf['year'].iloc[0]
            union_shore = sgdf.union_all()
            
            if t_geom.intersects(union_shore):
                intersect_geom = t_geom.intersection(union_shore)
                if intersect_geom.geom_type == 'Point':
                    d = base_pt.distance(intersect_geom)
                    intersections.append({'year': yr, 'distance': d})
                    if yr == 2026:
                        pt_2026 = intersect_geom
                elif intersect_geom.geom_type in ['MultiPoint', 'GeometryCollection']:
                    pts = [p for p in intersect_geom.geoms if p.geom_type == 'Point']
                    if pts:
                        nearest_pt = min(pts, key=lambda p: base_pt.distance(p))
                        d = base_pt.distance(nearest_pt)
                        intersections.append({'year': yr, 'distance': d})
                        if yr == 2026:
                            pt_2026 = nearest_pt
        
        if len(intersections) < 3 or pt_2026 is None:
            continue
            
        df_int = pd.DataFrame(intersections).sort_values('year')
        
        slope, intercept, r_val, p_val, std_err = linregress(df_int['year'], df_int['distance'])
        nsm = df_int['distance'].iloc[-1] - df_int['distance'].iloc[0]
        
        dx = t_geom.coords[1][0] - t_geom.coords[0][0]
        dy = t_geom.coords[1][1] - t_geom.coords[0][1]
        t_len = t_geom.length
        ux, uy = dx / t_len, dy / t_len
        
        slope_capped = np.clip(slope, -15.0, 15.0)
        dt_2035 = slope_capped * (2035 - 2026)
        dt_2050 = slope_capped * (2050 - 2026)
        
        p35 = Point(pt_2026.x + ux * dt_2035, pt_2026.y + uy * dt_2035)
        p50 = Point(pt_2026.x + ux * dt_2050, pt_2026.y + uy * dt_2050)
        
        results.append({
            'transect_id': t_id,
            'lrr_m_yr': slope,
            'r2': r_val**2,
            'nsm_m': nsm,
            'valid_pts': len(df_int),
            'pt_2035': p35,
            'pt_2050': p50,
            'geometry': t_geom
        })
        
    analysis_gdf = gpd.GeoDataFrame(results, crs=transects_gdf.crs)
    
    df_35 = pd.DataFrame({'transect_id': analysis_gdf['transect_id'], 'pt': analysis_gdf['pt_2035']})
    df_50 = pd.DataFrame({'transect_id': analysis_gdf['transect_id'], 'pt': analysis_gdf['pt_2050']})
    
    line_2035 = build_smooth_forecast_line(df_35)
    line_2050 = build_smooth_forecast_line(df_50)
    
    fc_2035_gdf = gpd.GeoDataFrame([{'year': 2035, 'geometry': line_2035}], crs=transects_gdf.crs)
    fc_2050_gdf = gpd.GeoDataFrame([{'year': 2050, 'geometry': line_2050}], crs=transects_gdf.crs)
    
    analysis_gdf = analysis_gdf.drop(columns=['pt_2035', 'pt_2050'])
    return analysis_gdf, fc_2035_gdf, fc_2050_gdf

if __name__ == "__main__":
    print("Running Final DSAS Pipeline...")
    aoi_gdf = gpd.read_file(AOI_PATH)
    sh_2026_path = os.path.join(PROCESSED_DIR, "shoreline_2026.gpkg")
    
    baseline_gdf = generate_ocean_baseline(aoi_gdf, sh_2026_path)
    baseline_gdf.to_file(os.path.join(PROCESSED_DIR, "dsas_baseline.gpkg"), driver="GPKG")
    
    transects_gdf = generate_transects(baseline_gdf, spacing=TRANSECT_SPACING, length=TRANSECT_LENGTH)
    transects_gdf.to_file(os.path.join(PROCESSED_DIR, "dsas_transects.gpkg"), driver="GPKG")
    
    analysis_gdf, fc_2035_gdf, fc_2050_gdf = run_dsas_analysis(transects_gdf, aoi_gdf)
    
    analysis_gdf.to_file(os.path.join(PROCESSED_DIR, "dsas_rate_results.gpkg"), driver="GPKG")
    fc_2035_gdf.to_file(os.path.join(PROCESSED_DIR, "forecast_shoreline_2035.gpkg"), driver="GPKG")
    fc_2050_gdf.to_file(os.path.join(PROCESSED_DIR, "forecast_shoreline_2050.gpkg"), driver="GPKG")
    
    print("Completed! forecast layers generated.")