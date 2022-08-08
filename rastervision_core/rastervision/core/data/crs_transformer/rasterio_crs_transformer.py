from typing import Optional
from pyproj import Transformer

import rasterio as rio
from rasterio.transform import (rowcol, xy)
from rasterio import Affine

from rastervision.core.data.crs_transformer import (CRSTransformer,
                                                    IdentityCRSTransformer)


class RasterioCRSTransformer(CRSTransformer):
    """Transformer for a RasterioRasterSource."""

    def __init__(self, transform, image_crs, map_crs='epsg:4326'):
        """Constructor.

        Args:
            transform: Rasterio affine transform
            image_crs: CRS of image in format that PyProj can handle eg. wkt or init
                string
            map_crs: CRS of the labels
        """

        if (image_crs is None) or (image_crs == map_crs):
            self.map2image = lambda *args, **kws: args[:2]
            self.image2map = lambda *args, **kws: args[:2]
        else:
            self.map2image = Transformer.from_crs(
                map_crs, image_crs, always_xy=True).transform
            self.image2map = Transformer.from_crs(
                image_crs, map_crs, always_xy=True).transform

        super().__init__(transform, image_crs, map_crs)

    def map_to_pixel(self, map_point):
        """Transform point from map to pixel-based coordinates.

        Args:
            map_point: (x, y) tuple in map coordinates

        Returns:
            (x, y) tuple in pixel coordinates
        """
        image_point = self.map2image(*map_point)
        pixel_point = rowcol(self.transform, image_point[0], image_point[1])
        pixel_point = (pixel_point[1], pixel_point[0])
        return pixel_point

    def pixel_to_map(self, pixel_point):
        """Transform point from pixel to map-based coordinates.

        Args:
            pixel_point: (x, y) tuple in pixel coordinates

        Returns:
            (x, y) tuple in map coordinates
        """
        image_point = xy(self.transform, int(pixel_point[1]),
                         int(pixel_point[0]))
        map_point = self.image2map(*image_point)
        return map_point

    @classmethod
    def from_dataset(cls, dataset, map_crs: Optional[str] = 'epsg:4326'
                     ) -> 'RasterioCRSTransformer':
        transform = dataset.transform
        image_crs = None if dataset.crs is None else dataset.crs.wkt
        map_crs = image_crs if map_crs is None else map_crs

        no_crs_tf = (image_crs is None) or (image_crs == map_crs)
        no_affine_tf = (transform is None) or (transform == Affine.identity())
        if no_crs_tf and no_affine_tf:
            return IdentityCRSTransformer()

        if transform is None:
            transform = Affine.identity()

        return cls(transform, image_crs, map_crs)

    @classmethod
    def from_uri(cls, uri: str, map_crs: Optional[str] = 'epsg:4326'
                 ) -> 'RasterioCRSTransformer':
        with rio.open(uri) as ds:
            return cls.from_dataset(ds, map_crs=map_crs)
