# basic packages
import os
import glob
import urllib
from numpy import newaxis
#import xarray as xr
import numpy as np
import pandas as pd
import warnings

from shapely.geometry import shape

# rasterio
import rasterio
import rasterio.plot
#from rasterio import features
#from rasterio.windows import from_bounds
from rasterio.plot import show
#from rasterio.enums import Resampling
#from rasterio.mask import mask
#from rasterio.fill import fillnodata
#from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.features import rasterize
from rasterio.features import shapes

# gis
#from pysheds.grid import Grid
#from scipy import ndimage
import geopandas as gpd
#from rasterio.crs import CRS

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import colors

def dem_from_mml(out_fd, subset, apikey, layer='korkeusmalli_2m', form='image/tiff', scalefactor=0.125, plot=True, save_in='asc'):

    '''Downloads a raster from MML database and writes it to dirpath folder in local memory

        Parameters:
        subset = boundary coordinates [minx, miny, maxx, maxy] (list)
        layer = the layer wanted to fetch e.g. 'korkeusmalli_2m' or 'korkeusmalli_10m' (str)
        form = form of the raster e.g 'image/tiff' (str)
        plot = whether or not to plot the created raster, True/False
        '''

    # The base url for maanmittauslaitos
    url = 'https://avoin-karttakuva.maanmittauslaitos.fi/ortokuvat-ja-korkeusmallit/wcs/v2?'
    scalefactorstr = f'SCALEFACTOR={scalefactor}'

    # Defining the latter url code
    params = dict(service='service=WCS',
                  version='version=2.0.1',
                  request='request=GetCoverage',
                  CoverageID=f'CoverageID={layer}',
                  SUBSET=f'SUBSET=E({subset[0]},{subset[2]})&SUBSET=N({subset[1]},{subset[3]})',
                  outformat=f'format={form}',
                  compression='geotiff:compression=LZW',
                  #scalefactor=scalefactorstr,
                  api=f'api-key={apikey}')
    if scalefactor < 1.:
        params['scalefactor'] = scalefactorstr

    if not os.path.exists(out_fd):
        # Create a new directory because it does not exist
        os.makedirs(out_fd)
    
    par_url = ''
    for par in params.keys():
        par_url += params[par] + '&'
    par_url = par_url[0:-1]
    new_url = (url + par_url)
    #print(new_url)
    # Putting the whole url together
    r = urllib.request.urlretrieve(new_url)

    # Open the file with the url:
    raster = rasterio.open(r[0])

    del r
    res = int(2/scalefactor) # !! WATCHOUT 2 IS HARD CODED
    layer = f'korkeusmalli_{res}m'

    if save_in=='tif':
        out_fp = os.path.join(out_fd, layer) + '.tif'
    elif save_in=='asc':
        out_fp = os.path.join(out_fd, layer) + '.asc'
        
    # Copy the metadata
    out_meta = raster.meta.copy()

    # Update the metadata
    out_meta.update({"driver": "GTiff",
                     "height": raster.height,
                     "width": raster.width,
                     "transform": raster.meta['transform'],
                     "crs": raster.meta['crs'],
                     "nodata":-9999,
                         }
                    )
    if save_in=='asc':
            out_meta.update({"driver": "AAIGrid"})
        
    # Manipulating the data for writing purpose
    raster_dem = raster.read(1)
    raster_dem = raster_dem[newaxis, :, :]

    # Write the raster to disk
    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(raster_dem)

    raster_dem = rasterio.open(out_fp)

    if plot==True:
        show(raster_dem)
    
    return raster_dem, out_fp


def burn_water_into_dem(dem_path, water_mask_path, drop=1.0):
    """
    Lowers DEM elevation by `drop` meters where water bodies are present in the water mask.

    Parameters:
        dem_path (str): Path to the input DEM (.tif).
        water_mask_path (str): Path to the binary water mask raster (non-zero values indicate water).
        drop (float): Amount in meters to drop elevation in water-covered cells.

    Returns:
        str: Path to the modified DEM with "_water" suffix.
    """
    # Load DEM
    with rasterio.open(dem_path) as dem_src:
        dem_data = dem_src.read(1)
        dem_meta = dem_src.meta.copy()
        dem_nodata = dem_src.nodata

    # Load water mask
    with rasterio.open(water_mask_path) as water_src:
        water_data = water_src.read(1)

    # Identify water cells
    water_mask = water_data > 0

    # Modify DEM
    dem_modified = np.copy(dem_data)
    if dem_nodata is not None:
        valid_mask = (dem_data != dem_nodata)
        dem_modified[water_mask & valid_mask] -= drop
    else:
        dem_modified[water_mask] -= drop

    # Prepare metadata
    dem_meta.update({
        'dtype': 'float32',
        'nodata': dem_nodata,
        'compress': 'lzw'
    })

    # Construct output path
    base, ext = os.path.splitext(dem_path)
    output_path = f"{base}_water{ext}"

    # Save modified DEM
    with rasterio.open(output_path, 'w', **dem_meta) as dst:
        dst.write(dem_modified.astype('float32'), 1)

    print(f"Modified DEM saved to: {output_path}")
    return output_path


def rasterize_water_bodies(stream_file=None, river_file=None, lake_file=None, ref_raster=None):
    """
    Rasterize stream, river, and/or lake geometries into the grid of a reference DEM.
    Any of the vector layers (stream, river, lake) can be omitted by passing None.

    Parameters:
        stream_file (str or None): Path to stream shapefile (lines).
        river_file (str or None): Path to river shapefile (polygons).
        lake_file (str or None): Path to lake shapefile (polygons).
        ref_raster (str): Path to reference raster (e.g., DEM).

    Returns:
        str: Path to the output hydrological mask raster.
    """
    if ref_raster is None:
        raise ValueError("ref_raster must be provided.")

    base, _ = os.path.splitext(ref_raster)

    suffix_parts = []
    if stream_file: suffix_parts.append("streams")
    if river_file:  suffix_parts.append("rivers")
    if lake_file:   suffix_parts.append("lakes")
    suffix = "_".join(suffix_parts) or "water"

    out_raster = f"{base}_{suffix}.tif"

    # Load and collect vector data
    vector_layers = []
    for file in [stream_file, river_file, lake_file]:
        if file:
            gdf = gpd.read_file(file)
            vector_layers.append(gdf)

    if not vector_layers:
        raise ValueError("At least one of stream_file, river_file, or lake_file must be provided.")

    # Open reference raster
    with rasterio.open(ref_raster) as ref:
        meta = ref.meta.copy()
        out_shape = (ref.height, ref.width)
        transform = ref.transform
        crs = ref.crs

    # Reproject all to raster CRS
    for gdf in vector_layers:
        if gdf.crs != crs:
            gdf.to_crs(crs, inplace=True)

    # Collect all geometries
    combined_shapes = []
    for gdf in vector_layers:
        combined_shapes.extend([(geom, 1) for geom in gdf.geometry if geom is not None])

    # Rasterize
    rasterized = rasterize(
        combined_shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype='uint8'
    )

    # Update metadata
    meta.update({
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "compress": "lzw",
        "nodata": 0
    })

    with rasterio.open(out_raster, 'w', **meta) as dst:
        dst.write(rasterized, 1)

    print(f"Hydrological mask raster saved to: {out_raster}")
    return out_raster


def raster_to_watershed_shapefile(input_raster: str) -> str:
    """
    Convert a watershed raster to a polygon shapefile.

    The output shapefile path is generated by replacing the input raster's
    file extension with '.shp'.

    Parameters:
        input_raster (str): Path to the input watershed raster file.

    Returns:
        str: Path to the output shapefile.
    """
    output_shapefile = os.path.splitext(input_raster)[0] + '.shp'

    with rasterio.open(input_raster) as src:
        raster = src.read(1)
        mask = raster != src.nodata
        transform = src.transform
        crs = src.crs

    results = (
        {'geometry': shape(geom), 'properties': {'ws_id': int(value)}}
        for geom, value in shapes(raster, mask=mask, transform=transform)
    )

    gdf = gpd.GeoDataFrame.from_features(results, crs=crs)
    gdf.to_file(output_shapefile)

    print(f"Watershed shapefile saved to: {output_shapefile}")
    return output_shapefile