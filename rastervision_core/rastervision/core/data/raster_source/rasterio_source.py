from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union
import logging
import os
import subprocess

import numpy as np
import rasterio
from rasterio.enums import (ColorInterp, MaskFlags, Resampling)

from rastervision.pipeline import rv_config
from rastervision.pipeline.file_system import download_if_needed
from rastervision.core.box import Box
from rastervision.core.data.crs_transformer import RasterioCRSTransformer
from rastervision.core.data.raster_source import (RasterSource, CropOffsets)
from rastervision.core.data.utils import listify_uris

if TYPE_CHECKING:
    from rasterio.io import DatasetReader
    from rastervision.core.data import RasterTransformer

log = logging.getLogger(__name__)


def build_vrt(vrt_path: str, image_paths: List[str]) -> None:
    """Build a VRT for a set of TIFF files."""
    log.info('Building VRT...')
    cmd = ['gdalbuildvrt', vrt_path]
    cmd.extend(image_paths)
    subprocess.run(cmd)


def download_and_build_vrt(image_uris: List[str],
                           vrt_dir: str,
                           stream: bool = False) -> str:
    if not stream:
        image_uris = [download_if_needed(uri) for uri in image_uris]
    vrt_path = os.path.join(vrt_dir, 'index.vrt')
    build_vrt(vrt_path, image_uris)
    return vrt_path


def load_window(
        image_dataset: 'DatasetReader',
        bands: Optional[Union[int, Sequence[int]]] = None,
        window: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        is_masked: bool = False,
        out_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """Load a window of an image using Rasterio.

    Args:
        image_dataset: a Rasterio dataset.
        bands (Optional[Union[int, Sequence[int]]]): Band index or indices to
            read. Must be 1-indexed.
        window (Optional[Tuple[Tuple[int, int], Tuple[int, int]]]):
            ((row_start, row_stop), (col_start, col_stop)) or
            ((y_min, y_max), (x_min, x_max)). If None, reads the entire raster.
            Defaults to None.
        is_masked (bool): If True, read a masked array from rasterio.
            Defaults to False.
        out_shape (Optional[Tuple[int, int]]): (hieght, width) of the output
            chip. If None, no resizing is done. Defaults to None.

    Returns:
        np.ndarray of shape (height, width, channels) where channels is the
            number of channels in the image_dataset.
    """
    if bands is not None:
        bands = tuple(bands)
    im = image_dataset.read(
        indexes=bands,
        window=window,
        boundless=True,
        masked=is_masked,
        out_shape=out_shape,
        resampling=Resampling.bilinear)

    if is_masked:
        im = np.ma.filled(im, fill_value=0)

    # Handle non-zero NODATA values by setting the data to 0.
    if bands is None:
        for channel, nodataval in enumerate(image_dataset.nodatavals):
            if nodataval is not None and nodataval != 0:
                im[channel, im[channel] == nodataval] = 0
    else:
        for channel, src_band in enumerate(bands):
            src_band_0_indexed = src_band - 1
            nodataval = image_dataset.nodatavals[src_band_0_indexed]
            if nodataval is not None and nodataval != 0:
                im[channel, im[channel] == nodataval] = 0

    im = np.transpose(im, axes=[1, 2, 0])
    return im


def fill_overflow(extent: Box,
                  window: Box,
                  arr: np.ndarray,
                  fill_value: int = 0) -> np.ndarray:
    """Given a window and corresponding array of values, if the window
    overflows the extent, fill the overflowing regions with fill_value.
    """
    top_overflow = max(0, extent.ymin - window.ymin)
    bottom_overflow = max(0, window.ymax - extent.ymax)
    left_overflow = max(0, extent.xmin - window.xmin)
    right_overflow = max(0, window.xmax - extent.xmax)

    h, w = arr.shape[:2]
    arr[:top_overflow] = fill_value
    arr[h - bottom_overflow:] = fill_value
    arr[:, :left_overflow] = fill_value
    arr[:, w - right_overflow:] = fill_value
    return arr


def get_channel_order_from_dataset(
        image_dataset: 'DatasetReader') -> List[int]:
    colorinterp = image_dataset.colorinterp
    if colorinterp:
        channel_order = [
            i for i, color_interp in enumerate(colorinterp)
            if color_interp != ColorInterp.alpha
        ]
    else:
        channel_order = list(range(0, image_dataset.count))
    return channel_order


class RasterioSource(RasterSource):
    """A rasterio-based RasterSource.

    This RasterSource can read any file that can be opened by Rasterio/GDAL
    including georeferenced formats such as GeoTIFF and non-georeferenced
    formats such as JPG. See https://www.gdal.org/formats_list.html for more
    details.

    If channel_order is None, then use non-alpha channels. This also sets any
    masked or NODATA pixel values to be zeros.
    """

    def __init__(self,
                 uris: Union[str, List[str]],
                 raster_transformers: List['RasterTransformer'] = [],
                 tmp_dir: Optional[str] = None,
                 allow_streaming: bool = False,
                 channel_order: Optional[Sequence[int]] = None,
                 extent_crop: Optional[CropOffsets] = None):
        """Constructor.

        Args:
            uris (Union[str, List[str]]): One or more URIs of images. If more
                than one, the images will be mosaiced together using GDAL.
            raster_transformers (List['RasterTransformer']): RasterTransformers
                to use to trasnform chips after they are read.
            tmp_dir (Optional[str]): Directory to use for downloading data
                and/or store VRT. If None, will be auto-generated. Defaults to
                None.
            allow_streaming (bool): If True, read data without downloading the
                entire file first. Defaults to False.
            channel_order (Optional[Sequence[int]]): List of indices of
                channels to extract from raw imagery. Can be a subset of the
                available channels. If None, all channels available in the
                image will be read. Defaults to None.
            extent_crop (Optional[CropOffsets], optional): Relative
                offsets (top, left, bottom, right) for cropping the extent.
                Useful for using splitting a scene into different datasets.
                Defaults to None i.e. no cropping.
        """
        self.uris = listify_uris(uris)
        self.tmp_dir = rv_config.get_tmp_dir() if tmp_dir is None else tmp_dir
        self.image_dataset = None
        self.allow_streaming = allow_streaming
        self.extent_crop = extent_crop

        self.imagery_path = self.download_data(
            self.tmp_dir, stream=self.allow_streaming)
        self.image_dataset = rasterio.open(self.imagery_path)
        self.crs_transformer = RasterioCRSTransformer.from_dataset(
            self.image_dataset)
        self._dtype = None

        self.height = self.image_dataset.height
        self.width = self.image_dataset.width

        num_channels_raw = self.image_dataset.count
        if channel_order is None:
            channel_order = get_channel_order_from_dataset(self.image_dataset)
        self.bands_to_read = np.array(channel_order, dtype=int) + 1

        mask_flags = self.image_dataset.mask_flag_enums
        self.is_masked = any(
            [m for m in mask_flags if m != MaskFlags.all_valid])

        super().__init__(channel_order, num_channels_raw, raster_transformers)

    @property
    def shape(self) -> Tuple[int, int, int]:
        ymin, xmin, ymax, xmax = self.get_extent()
        return ymax - ymin, xmax - xmin, self.num_channels

    @property
    def dtype(self) -> Tuple[int, int, int]:
        if self._dtype is None:
            # Read 1x1 chip to determine dtype
            test_chip = self.get_chip(Box.make_square(0, 0, 1))
            self._dtype = test_chip.dtype
        return self._dtype

    def download_data(self, tmp_dir: str, stream: bool = False) -> str:
        """Download any data needed for this Raster Source.

        Return a single local path representing the image or a VRT of the data.
        """
        if len(self.uris) == 1:
            if stream:
                return self.uris[0]
            else:
                return download_if_needed(self.uris[0], download_dir=tmp_dir)
        else:
            return download_and_build_vrt(self.uris, tmp_dir, stream=stream)

    def get_crs_transformer(self) -> RasterioCRSTransformer:
        return self.crs_transformer

    def get_extent(self) -> Box:
        h, w = self.height, self.width
        if self.extent_crop is not None:
            skip_top, skip_left, skip_bottom, skip_right = self.extent_crop
            ymin, xmin = int(h * skip_top), int(w * skip_left)
            ymax, xmax = h - int(h * skip_bottom), w - int(w * skip_right)
            return Box(ymin, xmin, ymax, xmax)
        return Box(0, 0, h, w)

    def _get_chip(self,
                  window: Box,
                  bands: Optional[Sequence[int]] = None,
                  out_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        chip = load_window(
            self.image_dataset,
            bands=bands,
            window=window.rasterio_format(),
            is_masked=self.is_masked,
            out_shape=out_shape)
        if self.extent_crop is not None:
            chip = fill_overflow(self.get_extent(), window, chip)
        return chip

    def get_chip(self,
                 window: Box,
                 bands: Optional[Union[Sequence[int], slice]] = None,
                 out_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """Read a chip specified by a window from the file.

        Args:
            window (Box): Bounding box of chip in pixel coordinates.
            bands (Optional[Union[Sequence[int], slice]], optional): Subset of
                bands to read. Note that this will be applied on top of the
                channel_order (if specified). So if this is an RGB image and
                channel_order=[2, 1, 0], then using bands=[0] will return the
                B-channel. Defaults to None.
            out_shape (Optional[Tuple[int, ...]], optional): (hieght, width) of
            the output chip. If None, no resizing is done. Defaults to None.

        Returns:
            np.ndarray: A chip of shape (height, width, channels).
        """
        bands_to_read = self.bands_to_read
        if bands is not None:
            bands_to_read = bands_to_read[bands]
        chip = self._get_chip(window, out_shape=out_shape, bands=bands_to_read)
        for transformer in self.raster_transformers:
            chip = transformer.transform(chip, self.channel_order)
        return chip

    def __getitem__(self, key: Any) -> 'np.ndarray':
        if isinstance(key, Box):
            return self.get_chip(key)
        elif isinstance(key, slice):
            key = [key]
        elif isinstance(key, tuple):
            pass
        else:
            raise TypeError('Unsupported key type.')

        slices = list(key)
        assert 1 <= len(slices) <= 3
        assert all(s is not None for s in slices)
        assert isinstance(slices[0], slice)
        if len(slices) == 1:
            h, = slices
            w = slice(None, None)
            c = None
        elif len(slices) == 2:
            assert isinstance(slices[1], slice)
            h, w = slices
            c = None
        else:
            h, w, c = slices

        if any(x is not None and x < 0
               for x in [h.start, h.stop, w.start, w.stop]):
            raise NotImplementedError()

        ymin, xmin, ymax, xmax = self.get_extent()
        _ymin = ymin if h.start is None else h.start + ymin
        _xmin = xmin if w.start is None else w.start + xmin
        _ymax = ymax if h.stop is None else min(h.stop + ymin, ymax)
        _xmax = xmax if w.stop is None else min(w.stop + xmin, xmax)
        window = Box(_ymin, _xmin, _ymax, _xmax)

        out_shape = None
        if h.step is not None or w.step is not None:
            if h.step is not None:
                out_h = (ymax - ymin) // h.step
            else:
                out_h = ymax - ymin
            if w.step is not None:
                out_w = (xmax - xmin) // w.step
            else:
                out_w = xmax - xmin
            out_shape = (int(out_h), int(out_w))

        chip = self.get_chip(window, bands=c, out_shape=out_shape)
        return chip
