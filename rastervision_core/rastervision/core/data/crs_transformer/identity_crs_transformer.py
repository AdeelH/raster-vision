from rastervision.core.data.crs_transformer import CRSTransformer
from rastervision.core.data.crs_transformer.crs_transformer import PointT


class IdentityCRSTransformer(CRSTransformer):
    """Transformer for when map coordinates are already in pixel coordinates.

    This is useful for non-georeferenced imagery.
    """

    def _map_to_pixel_point(self, map_point: PointT) -> PointT:
        """Identity function.

        Args:
            map_point: (x, y) tuple in pixel coordinates

        Returns:
            (x, y) tuple in pixel coordinates
        """
        return map_point

    def _pixel_to_map_point(self, pixel_point: PointT) -> PointT:
        """Identity function.

        Args:
            pixel_point: (x, y) tuple in pixel coordinates

        Returns:
            (x, y) tuple in pixel coordinates
        """
        return pixel_point
