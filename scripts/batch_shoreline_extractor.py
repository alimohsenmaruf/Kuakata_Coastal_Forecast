import os
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping
from rasterio.features import shapes
from pystac_client import Client
import planetary_computer as pc

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
AOI_PATH = os.path.join(PROJECT_DIR, "data", "boundry", "kuakata_study_area.gpkg")

# Kuakata Bounding Box [min_lon, min_lat, max_lon, max_lat]
BBOX = [90.0, 21.7, 90.4, 22.1]

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def fetch_best_scene(year):
    """Finds the lowest cloud-cover winter scene for a given year."""
    client = Client.open(STAC_URL)
    start_date = f"{year-1}-12-01"
    end_date = f"{year}-03-01"
    
    search = client.search(
        collections=["landsat-c2-l2"],
        bbox=BBOX,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": 20}}
    )
    items = list(search.items())
    if not items:
        print(f"No scenes < 20% cloud found for {year}")
        return None
    
    # Sort by cloud cover ascending
    items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
    # Sign item for SAS token authorization
    return pc.sign(items[0])

def extract_shoreline_for_year(year):
    item = fetch_best_scene(year)
    if not item:
        return
    
    scene_id = item.id
    print(f"\n🚀 Processing {year} (Scene: {scene_id})...")
    
    # STAC asset keys for Green and SWIR1 (1.6um) across Landsat Collection 2
    green_key = "green"
    swir_key = "swir16"
        
    green_href = item.assets[green_key].href
    swir_href = item.assets[swir_key].href
    
    # Load study area geometry
    aoi_gdf = gpd.read_file(AOI_PATH)
    
    # Process MNDWI directly from HTTP cloud sources
    with rasterio.open(green_href) as src_g, rasterio.open(swir_href) as src_s:
        # Reproject AOI to raster CRS if needed
        aoi_proj = aoi_gdf.to_crs(src_g.crs)
        geoms = [mapping(g) for g in aoi_proj.geometry]
        
        green_img, transform = mask(src_g, geoms, crop=True)
        swir_img, _ = mask(src_s, geoms, crop=True)
        
        green = green_img[0].astype(np.float32)
        swir = swir_img[0].astype(np.float32)
        
        # Calculate MNDWI
        denominator = green + swir
        denominator[denominator == 0] = np.nan
        mndwi = (green - swir) / denominator
        
        # Water Mask Threshold (Water = 1, Land = 0)
        land_mask = np.where(mndwi <= 0, 1, 0).astype(np.uint8)
        
        # Polygonize Land Pixels
        results = (
            {'properties': {'DN': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(land_mask, mask=(land_mask == 1), transform=transform))
        )
        
        geoms_list = list(results)
        if not geoms_list:
            print(f"Extraction failed for {year}: No land geometry generated.")
            return

        # Create GeoDataFrame for Land Polygons
        land_gdf = gpd.GeoDataFrame.from_features(geoms_list, crs=src_g.crs)
        
        # Convert Polygons to Boundary Lines (Shorelines)
        shoreline_gdf = land_gdf.copy()
        shoreline_gdf["geometry"] = shoreline_gdf.geometry.boundary
        shoreline_gdf["year"] = year
        
        # Ensure target CRS is EPSG:32646 (UTM Zone 46N)
        shoreline_gdf = shoreline_gdf.to_crs("EPSG:32646")
        
        # Export Shoreline GeoPackage
        out_path = os.path.join(PROCESSED_DIR, f"shoreline_{year}.gpkg")
        shoreline_gdf.to_file(out_path, driver="GPKG")
        print(f"Successfully exported: shoreline_{year}.gpkg")

# ---------------------------------------------------------
# BATCH EXECUTION LOOP (1988 - 2026)
# ---------------------------------------------------------
if __name__ == "__main__":
    years_to_process = list(range(1988, 2027))
    print(f"Starting batch extraction for {len(years_to_process)} annual epochs...")
    
    for yr in years_to_process:
        try:
            extract_shoreline_for_year(yr)
        except Exception as e:
            print(f"Failed processing year {yr}: {e}")