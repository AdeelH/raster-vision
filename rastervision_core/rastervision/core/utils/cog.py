from typing import TYPE_CHECKING
import os

import rasterio
from rasterio.enums import Resampling
import rasterio.shutil

from rastervision.pipeline.file_system import (download_or_copy,
                                               upload_or_copy)

if TYPE_CHECKING:
    from collections.abc import Sequence

COGIFY = 'COGIFY'

DEFAULT_BLOCK_SIZE = 512
DEFAULT_RESAMPLE_METHOD = 'near'
DEFAULT_COMPRESSION = 'deflate'
DEFAULT_OVERVIEWS = [2, 4, 8, 16, 32]

# GDAL (gdaladdo) resampling method names --> rasterio Resampling.
RESAMPLE_METHODS = {
    'near': Resampling.nearest,
    'nearest': Resampling.nearest,
    'bilinear': Resampling.bilinear,
    'cubic': Resampling.cubic,
    'cubicspline': Resampling.cubic_spline,
    'cubic_spline': Resampling.cubic_spline,
    'lanczos': Resampling.lanczos,
    'average': Resampling.average,
    'mode': Resampling.mode,
    'gauss': Resampling.gauss,
    'rms': Resampling.rms,
}


def _get_resampling(resample_method: str) -> Resampling:
    try:
        return RESAMPLE_METHODS[resample_method.lower()]
    except KeyError:
        raise ValueError(
            f'Unsupported resample_method: "{resample_method}". '
            f'Must be one of: {list(RESAMPLE_METHODS.keys())}.') from None


def create_cog(source_uri: str,
               dest_uri: str,
               local_dir: str,
               block_size: int = DEFAULT_BLOCK_SIZE,
               resample_method: str = DEFAULT_RESAMPLE_METHOD,
               compression: str = DEFAULT_COMPRESSION,
               overviews: 'Sequence[int] | None' = None) -> None:
    """Convert a raster to a Cloud Optimized GeoTIFF (COG).

    The source raster is first copied to a tiled GeoTIFF to which overviews
    are added at the requested levels. This is then converted to a COG
    (using GDAL's COG driver via rasterio) that reuses those overviews.

    Args:
        source_uri: URI of the source raster.
        dest_uri: URI to write the COG to.
        local_dir: Local directory to use for intermediate files.
        block_size: Tile size (in pixels) used for the full-resolution image
            and the overviews. Defaults to 512.
        resample_method: GDAL resampling method name (as accepted by
            ``gdaladdo -r``) used to build the overviews. One of: 'near',
            'nearest', 'bilinear', 'cubic', 'cubicspline', 'lanczos',
            'average', 'mode', 'gauss', 'rms'. Defaults to 'near'.
        compression: Compression method (e.g. 'deflate', 'lzw', 'zstd',
            'jpeg') or 'none' to disable compression. Defaults to 'deflate'.
        overviews: Overview decimation levels. If None or empty, defaults to
            [2, 4, 8, 16, 32].

    Raises:
        ValueError: If resample_method is not supported.
    """
    resampling = _get_resampling(resample_method)
    if not overviews:
        overviews = DEFAULT_OVERVIEWS
    overviews = [int(x) for x in overviews]
    compression = compression.upper()

    local_path = download_or_copy(source_uri, local_dir)
    fname = os.path.splitext(os.path.basename(local_path))[0]
    translate_path = os.path.join(local_dir, f'{fname}-translate.tif')
    output_path = os.path.join(local_dir, f'{fname}-cog.tif')

    # Step 1: Copy to a tiled GeoTIFF. Lossless compression is used for this
    # intermediate file to avoid compounding losses from lossy methods.
    rasterio.shutil.copy(
        local_path,
        translate_path,
        driver='GTiff',
        TILED='YES',
        BIGTIFF='IF_SAFER',
        COMPRESS='NONE' if compression == 'NONE' else 'DEFLATE')

    # Step 2: Add overviews.
    with rasterio.open(translate_path, 'r+') as ds:
        ds.build_overviews(overviews, resampling)

    # Step 3: Convert to COG, reusing the overviews built above.
    rasterio.shutil.copy(
        translate_path,
        output_path,
        driver='COG',
        BLOCKSIZE=block_size,
        OVERVIEWS='FORCE_USE_EXISTING',
        BIGTIFF='IF_SAFER',
        COMPRESS=compression)
    os.remove(translate_path)

    upload_or_copy(output_path, dest_uri)
