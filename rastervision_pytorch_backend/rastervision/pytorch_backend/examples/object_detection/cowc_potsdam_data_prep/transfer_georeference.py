#!/usr/bin/env python

import sys

import rasterio
import rasterio.shutil
from rasterio.transform import from_bounds


def transfer_georeference(input_rgb, input_label, output_label):
    """Copy input_label to output_label, georeferenced like input_rgb.

    The output gets the CRS of input_rgb and a transform that maps the
    label's pixel grid onto the bounds of input_rgb.
    """
    with rasterio.open(input_rgb) as ref:
        crs = ref.crs
        left, bottom, right, top = ref.bounds

    rasterio.shutil.copy(input_label, output_label, driver='GTiff')
    with rasterio.open(output_label, 'r+') as dst:
        dst.crs = crs
        dst.transform = from_bounds(left, bottom, right, top, dst.width,
                                    dst.height)


if __name__ == '__main__':
    if not len(sys.argv) >= 4:
        print('Usage: {} <input_rgb.tif> <input_label.tif> <output_label.tif>'.
              format(sys.argv[0]))
        exit()

    transfer_georeference(sys.argv[1], sys.argv[2], sys.argv[3])
