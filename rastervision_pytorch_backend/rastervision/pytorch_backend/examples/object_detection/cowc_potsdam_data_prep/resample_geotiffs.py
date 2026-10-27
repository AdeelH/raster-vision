import os
import glob

import click
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject

from rv.utils import make_empty_dir


def resample_geotiff(input_path, output_path, resolution):
    """Resample a GeoTIFF to the given resolution, keeping its CRS.

    Mimics ``gdalwarp -tr <res> <res> -r bilinear -setci -co ALPHA=NO``.
    """
    with rasterio.open(input_path) as src:
        # Same output extent and size computation as gdalwarp -tr.
        t, w, h = calculate_default_transform(src.crs, src.crs, src.width,
                                              src.height, *src.bounds)
        left, top = t.c, t.f
        right, bottom = left + w * t.a, top + h * t.e
        width = max(int((right - left + resolution / 2) / resolution), 1)
        height = max(int((top - bottom + resolution / 2) / resolution), 1)
        transform = from_origin(left, top, resolution, resolution)
        profile = dict(
            driver='GTiff',
            width=width,
            height=height,
            count=src.count,
            dtype=src.dtypes[0],
            crs=src.crs,
            transform=transform,
            nodata=src.nodata,
            ALPHA='NO')
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.colorinterp = src.colorinterp
            reproject(
                source=rasterio.band(src, src.indexes),
                destination=rasterio.band(dst, dst.indexes),
                resampling=Resampling.bilinear)


@click.command()
@click.argument('input_dir')
@click.argument('output_dir')
@click.option(
    '--resolution', default=0.3, help='Output resolution in meters/pixel')
def resample_geotiffs(input_dir, output_dir, resolution):
    input_paths = glob.glob(os.path.join(input_dir, '*.tif'))
    make_empty_dir(output_dir)

    for input_path in input_paths:
        input_filename = os.path.basename(input_path)
        resample_geotiff(input_path, os.path.join(output_dir, input_filename),
                         resolution)


if __name__ == '__main__':
    resample_geotiffs()
