from typing import TYPE_CHECKING, Sequence

from pydantic import NonNegativeInt as NonNegInt
import numpy as np
from pystac import Item
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.windows import Window, transform as window_transform

from rastervision.core.box import Box
from rastervision.core.data.crs_transformer import RasterioCRSTransformer
from rastervision.core.data.raster_source import RasterSource, RasterioSource
from rastervision.core.data.raster_source.stac_config import subset_assets
from rastervision.core.data.utils import all_equal

if TYPE_CHECKING:
    from typing import Self
    from rastervision.core.data import RasterTransformer, CRSTransformer

# Tolerance (in pixels) for considering two pixel grids to be aligned.
GRID_ALIGNMENT_TOL = 1e-6


class MultiRasterSource(RasterSource):
    """
    Merge multiple ``RasterSources`` by concatenating along the channel dim.

    Windows are specified in the pixel coordinates of the primary raster
    source. The chips read from the other raster sources are aligned with the
    pixel grid of the primary raster source: if the raster sources are
    :class:`.RasterioSource`\\ s with different CRSs, resolutions, or pixel
    alignments, their data is reprojected onto the primary raster source's
    pixel grid (using bilinear resampling).
    """

    def __init__(self,
                 raster_sources: Sequence[RasterSource],
                 primary_source_idx: NonNegInt = 0,
                 channel_order: Sequence[NonNegInt] | None = None,
                 raster_transformers: Sequence['RasterTransformer'] = [],
                 bbox: Box | None = None):
        """Constructor.

        Args:
            raster_sources: Sequence of RasterSources.
            primary_source_idx: Index of the raster source whose CRS, dtype,
                and other attributes will override those of the other raster
                sources.
            channel_order: Channel ordering that will be used by
                :meth:`.MultiRasterSource.get_chip`. Defaults to ``None``.
            raster_transformers: List of transformers. Defaults to ``[]``.
            bbox: User-specified crop of the extent. If specified, the primary
                raster source's bbox is set to this. If ``None``, the full
                extent available in the source file of the primary raster
                source is used.
        """
        dtype_raw = raster_sources[primary_source_idx].dtype

        num_channels_raw = sum(rs.num_channels for rs in raster_sources)
        if not channel_order:
            num_channels = sum(rs.num_channels for rs in raster_sources)
            channel_order = list(range(num_channels))

        # validate primary_source_idx
        if not (0 <= primary_source_idx < len(raster_sources)):
            raise IndexError('primary_source_idx must be in range '
                             '[0, len(raster_sources)].')

        if bbox is None:
            bbox = raster_sources[primary_source_idx].bbox
        else:
            raster_sources[primary_source_idx].set_bbox(bbox)

        super().__init__(
            channel_order=channel_order,
            num_channels_raw=num_channels_raw,
            dtype_raw=dtype_raw,
            bbox=bbox,
            raster_transformers=raster_transformers)

        self.raster_sources = raster_sources
        self.primary_source_idx = primary_source_idx
        self.non_primary_sources = [
            rs for i, rs in enumerate(raster_sources)
            if i != primary_source_idx
        ]
        self.validate_raster_sources()

    @classmethod
    def from_stac(
            cls,
            item: Item,
            assets: list[str] | None,
            primary_source_idx: NonNegInt = 0,
            raster_transformers: list['RasterTransformer'] = [],
            channel_order: Sequence[int] | None = None,
            bbox: Box | tuple[int, int, int, int] | None = None,
            bbox_map_coords: Box | tuple[int, int, int, int] | None = None,
            allow_streaming: bool = False) -> 'Self':
        """Construct a ``MultiRasterSource`` from a STAC Item.

        This creates a :class:`.RasterioSource` for each asset and puts all
        the raster sources together into a ``MultiRasterSource``. If ``assets``
        is not specified, all the assets in the STAC item are used.

        Only assets that are readable by rasterio are supported.

        Args:
            item: STAC Item.
            assets: List of names of assets to use. If ``None``, all assets
                present in the item will be used. Defaults to ``None``.
            primary_source_idx: Index of the raster source whose CRS, dtype,
                and other attributes will override those of the other raster
                sources.
            raster_transformers: RasterTransformers to use to transform chips
                after they are read. Defaults to ``[]``.
            channel_order: List of indices of channels to extract from raw
                imagery. Can be a subset of the available channels. If None,
                all channels available in the image will be read.
                Defaults to None.
            bbox: User-specified crop of the extent. Can be :class:`.Box` or
                (ymin, xmin, ymax, xmax) tuple. If None, the full extent
                available in the source file is used. Mutually exclusive with
                ``bbox_map_coords``. Defaults to ``None``.
            bbox_map_coords: User-specified bbox in EPSG:4326 coords. Can be
                :class:`.Box` or (ymin, xmin, ymax, xmax) tuple. Useful for
                cropping the raster source so that only part of the raster is
                read from. Mutually exclusive with ``bbox``.
                Defaults to ``None``.
            allow_streaming: Passed to :class:`.RasterioSource`. If ``False``,
                assets will be downloaded. Defaults to ``True``.
        """
        if bbox is not None and bbox_map_coords is not None:
            raise ValueError('Specify either bbox or bbox_map_coords, '
                             'but not both.')

        if assets is not None:
            item = subset_assets(item, assets)

        uris = [asset.href for asset in item.assets.values()]
        raster_sources = [
            RasterioSource(uri, allow_streaming=allow_streaming)
            for uri in uris
        ]

        crs_transformer = raster_sources[primary_source_idx].crs_transformer
        if bbox_map_coords is not None:
            bbox_map_coords = Box(*bbox_map_coords)
            bbox = crs_transformer.map_to_pixel(bbox_map_coords)
        elif bbox is not None:
            bbox = Box(*bbox)

        raster_source = MultiRasterSource(
            raster_sources,
            primary_source_idx=primary_source_idx,
            raster_transformers=raster_transformers,
            channel_order=channel_order,
            bbox=bbox)
        return raster_source

    def validate_raster_sources(self) -> None:
        """Validate sub-``RasterSources``.

        Checks if all raster sources have the same dtype.
        """
        dtypes = [rs.dtype for rs in self.raster_sources]
        if not all_equal(dtypes):
            raise ValueError(
                'dtypes of all sub raster sources must be the same. '
                f'Got: {dtypes}.')

    @property
    def primary_source(self) -> RasterSource:
        """Primary sub-``RasterSource``"""
        return self.raster_sources[self.primary_source_idx]

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the raster as a (..., H, W, C) tuple."""
        *shape, _ = self.primary_source.shape
        return (*shape, self.num_channels)

    @property
    def crs_transformer(self) -> 'CRSTransformer':
        return self.primary_source.crs_transformer

    def _get_sub_chips(self,
                       window: Box,
                       out_shape: tuple[int, int] | None = None
                       ) -> list[np.ndarray]:
        """Return chips from sub raster sources as a list.

        The chip from the primary sub raster source is read using the window.
        The chips from the remaining sub raster sources are read such that
        they are aligned with this chip. This is done as follows:

        - If a sub raster source's pixel grid is aligned with that of the
          primary sub raster source (same CRS and resolution, and offset by
          a whole number of pixels), the corresponding window is read from
          it directly.
        - Otherwise, if both are georeferenced :class:`.RasterioSource`\\ s,
          its data is reprojected onto the pixel grid of the primary chip
          (see :meth:`.RasterioSource.get_chip_reprojected`). This accounts
          for differences in CRS, resolution, and pixel alignment.
        - Otherwise, the window is converted to map coordinates using the
          primary sub raster source's CRS transformer and then to pixel
          coordinates using the sub raster source's CRS transformer, and the
          resulting window is read and resized to the shape of the primary
          chip. Note that this is only approximately aligned if the pixel
          grids differ.

        Args:
            window: The window for which to get the chip, in pixel coordinates.
            out_shape: (height, width) to resize the chip to.

        Returns:
            List of chips from each sub raster source.
        """
        primary_rs = self.primary_source
        primary_sub_chip = primary_rs.get_chip(window, out_shape=out_shape)
        if out_shape is None:
            out_shape = primary_sub_chip.shape[-3:-1]

        sub_chips = [
            self._get_aligned_sub_chip(rs, window, out_shape)
            for rs in self.non_primary_sources
        ]
        sub_chips.insert(self.primary_source_idx, primary_sub_chip)
        return sub_chips

    def _get_aligned_sub_chip(self, rs: RasterSource, window: Box,
                              out_shape: tuple[int, int]) -> np.ndarray:
        """Get chip from a non-primary sub raster source aligned with the
        chip from the primary sub raster source for the same window."""
        primary_rs = self.primary_source
        primary_grid = _get_grid(primary_rs)
        rs_grid = _get_grid(rs)

        if primary_grid is not None and rs_grid is not None:
            offset = _get_pixel_offset(primary_grid, rs_grid)
            if offset is not None:
                # Same pixel grid: read the corresponding window directly.
                dy, dx = offset
                rs_window = window.to_global_coords(primary_rs.bbox)
                rs_window = rs_window.translate(dy, dx)
                rs_window = rs_window.to_local_coords(rs.bbox)
                return rs.get_chip(rs_window, out_shape=out_shape)
            if isinstance(rs, RasterioSource):
                crs, transform = primary_grid
                transform = _get_window_transform(
                    transform, window.to_global_coords(primary_rs.bbox),
                    out_shape)
                return rs.get_chip_reprojected(crs, transform, out_shape)

        window_map_coords = primary_rs.crs_transformer.pixel_to_map(
            window, bbox=primary_rs.bbox)
        rs_window = rs.crs_transformer.map_to_pixel(
            window_map_coords, bbox=rs.bbox)
        # Ensure the window is at least 1 pixel in size. It can otherwise be
        # empty if rs has a much coarser resolution than the primary source.
        ymin, xmin, ymax, xmax = rs_window
        rs_window = Box(ymin, xmin, max(ymax, ymin + 1), max(xmax, xmin + 1))
        return rs.get_chip(rs_window, out_shape=out_shape)

    def _get_chip(self, window: Box,
                  out_shape: tuple[int, int] | None = None) -> np.ndarray:
        """Get chip w/o applying channel_order and transformers.

        Args:
            window (Box): The window for which to get the chip, in pixel
                coordinates.
            out_shape (tuple[int, int] | None): (height, width) to resize
                the chip to.

        Returns:
            Array of shape (height, width, channels).
        """
        sub_chips = self._get_sub_chips(window, out_shape=out_shape)
        chip = np.concatenate(sub_chips, axis=-1)
        return chip

    def get_chip(self, window: Box,
                 out_shape: tuple[int, int] | None = None) -> np.ndarray:
        """Return the transformed chip in the window.

        Get processed chips from sub raster sources (with their respective
        channel orders and transformations applied), concatenate them along the
        channel dimension, apply channel_order, followed by transformations.

        Args:
            window: The window for which to get the chip, in pixel coordinates.
            out_shape: (height, width) to resize the chip to.

        Returns:
            Array of shape (height, width, channels).
        """
        sub_chips = self._get_sub_chips(window, out_shape=out_shape)
        chip = np.concatenate(sub_chips, axis=-1)
        chip = chip[..., self.channel_order]

        for transformer in self.raster_transformers:
            chip = transformer.transform(chip)

        return chip


def _get_grid(rs: RasterSource) -> tuple[CRS, Affine] | None:
    """Return the (CRS, transform) of the raster source's pixel grid, or
    ``None`` if it is not georeferenced."""
    crs_transformer = rs.crs_transformer
    if not isinstance(crs_transformer, RasterioCRSTransformer):
        return None
    if crs_transformer.image_crs is None:
        return None
    crs = CRS.from_user_input(crs_transformer.image_crs)
    return crs, crs_transformer.transform


def _get_pixel_offset(grid_a: tuple[CRS, Affine],
                      grid_b: tuple[CRS, Affine]) -> tuple[int, int] | None:
    """Return the (row, col) offset of grid_a's pixels in grid_b, if the two
    grids are aligned, i.e., have the same CRS, the same resolution and
    orientation, and differ only by a whole number of pixels. Otherwise,
    return None."""
    crs_a, tf_a = grid_a
    crs_b, tf_b = grid_b
    if crs_a != crs_b:
        return None
    # position of grid_a's origin in grid_b's pixel coords
    col, row = ~tf_b * (tf_a.c, tf_a.f)
    dx, dy = round(col), round(row)
    tf_a_in_b = Affine.translation(dx, dy)
    # the transform from grid_a pixel coords to grid_b pixel coords should be
    # a pure whole-pixel translation
    a_to_b = ~tf_b * tf_a
    if not a_to_b.almost_equals(tf_a_in_b, precision=GRID_ALIGNMENT_TOL):
        return None
    return dy, dx


def _get_window_transform(transform: Affine, window: Box,
                          out_shape: tuple[int, int]) -> Affine:
    """Return the transform for a chip of shape out_shape read from window
    (in global pixel coords) of a raster with the given transform."""
    ymin, xmin, ymax, xmax = window
    win_h, win_w = ymax - ymin, xmax - xmin
    win_transform = window_transform(
        Window(xmin, ymin, win_w, win_h), transform)
    out_h, out_w = out_shape
    if (out_h, out_w) != (win_h, win_w):
        win_transform = win_transform * Affine.scale(win_w / out_w,
                                                     win_h / out_h)
    return win_transform
