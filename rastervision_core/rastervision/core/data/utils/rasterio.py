from typing import TYPE_CHECKING, Sequence
from dataclasses import dataclass
import os
from os.path import join
import logging
import math
import xml.etree.ElementTree as ET

import numpy as np
import rasterio as rio
import rasterio.windows as rio_windows
from rasterio.transform import from_origin
from rasterio.dtypes import dtype_rev, typename_fwd
from rasterio.enums import (ColorInterp, MaskFlags, Resampling)
from rasterio.errors import RasterioIOError
from rasterio.session import AWSSession

from rastervision.pipeline.file_system.utils import (
    file_to_json, get_local_path, get_tmp_dir, make_dir, upload_or_copy,
    download_if_needed, uri_to_vsi_path)
from rastervision.core.box import Box

if TYPE_CHECKING:
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.io import DatasetReader
    from rasterio.session import Session

log = logging.getLogger(__name__)


def write_window(dataset: 'DatasetReader',
                 arr: np.ndarray,
                 window: Box | None = None) -> None:
    """Write a (H, W[, C]) array out to a rasterio dataset.

    Args:
        dataset (DatasetReader): Rasterio dataset, opened for writing.
        arr (np.ndarray): Array to write.
        window (Box | None): Window (in pixel coords) to write to.
            Defaults to None.
    """
    if window is not None:
        window = window.rasterio_format()
    if arr.ndim == 2:
        dataset.write_band(1, arr, window=window)
    else:
        arr_chw = arr.transpose(2, 0, 1)
        for i, band in enumerate(arr_chw, start=1):
            dataset.write_band(i, band, window=window)


def write_bbox(path: str, arr: np.ndarray, bbox: Box, crs_wkt: str, **kwargs):
    """Write a (H, W[, C]) array to a GeoTIFF, georeferenced to the given bbox.

    Args:
        path (str): GeoTiff path.
        arr (np.ndarray): (H, W[, C]) Array to write.
        bbox (Box): Bounding box in map coords to georeference the GeoTiff to.
        crs_wkt (str): CRS in WKT format.
    """
    if arr.ndim == 2:
        h_arr, w_arr = arr.shape
        num_channels = 1
    else:
        h_arr, w_arr, num_channels = arr.shape
    h_bbox, w_bbox = bbox.size
    resolution = w_bbox / w_arr, h_bbox / h_arr
    transform = from_origin(bbox.xmin, bbox.ymax, *resolution)
    out_profile = dict(
        driver='GTiff',
        height=h_arr,
        width=w_arr,
        crs=crs_wkt,
        count=num_channels,
        dtype=arr.dtype,
        transform=transform,
    )
    out_profile.update(kwargs)
    with rio.open(path, 'w', **out_profile) as ds:
        write_window(ds, arr)


def write_geotiff_like_geojson(path: str,
                               arr: np.ndarray,
                               geojson_path: str,
                               crs: str | None = None,
                               **kwargs) -> None:
    """Write array to GeoTIFF, georeferenced to same bbox as the given GeoJSON.

    Args:
        path (str): GeoTiff path.
        arr (np.ndarray): (H, W[, C]) Array to write.
        geojson_path (str): GeoJSON path.
        crs (str): CRS name. If None, read from the GeoJSON. If not specified
            in the GeoJSON, use "EPSG:4326". Defaults to None.
    """
    from rastervision.core.data.utils.geojson import geojson_to_geoms
    import pyproj
    from shapely.ops import unary_union

    geojson = file_to_json(geojson_path)
    if crs is None:
        try:
            crs = geojson['crs']['properties']['name']
        except KeyError:
            crs = 'epsg:4326'
    crs_wkt = pyproj.CRS(crs).to_wkt()
    geoms = unary_union(list(geojson_to_geoms(geojson)))
    bbox = Box.from_shapely(geoms)
    write_bbox(path, arr, bbox=bbox, crs_wkt=crs_wkt, **kwargs)


def crop_geotiff(src_uri: str, window: Box, dst_uri: str):
    """Create a new GeoTIFF from a crop of an existing GeoTIFF.

    Args:
        src_uri (str): Source GeoTIFF URI to read from.
        window (Box): Window specifying the crop bounds.
        dst_uri (str): Crop GeoTIFF URI to write to.
    """
    rio_window = window.rasterio_format()

    with rio.open(src_uri) as src_ds, get_tmp_dir() as tmp_dir:
        crop_path = get_local_path(dst_uri, tmp_dir)
        make_dir(crop_path, use_dirname=True)

        meta = src_ds.meta
        colorinterp = src_ds.colorinterp
        img_cropped = src_ds.read(window=rio_window)

        meta['height'], meta['width'] = window.size
        meta['transform'] = rio_windows.transform(rio_window, src_ds.transform)

        with rio.open(crop_path, 'w', **meta) as dst_ds:
            dst_ds.colorinterp = colorinterp
            dst_ds.write(img_cropped)

        upload_or_copy(crop_path, dst_uri)


@dataclass
class _VRTSourceInfo:
    """Properties of an input raster needed to add it to a mosaic VRT."""
    path: str
    crs: 'CRS | None'
    transform: 'Affine'
    width: int
    height: int
    dtypes: tuple[str, ...]
    nodatavals: tuple[float | None, ...]
    colorinterp: tuple[ColorInterp, ...]
    scales: tuple[float, ...]
    offsets: tuple[float, ...]
    colormaps: dict[int, dict[int, tuple[int, ...]]]
    block_shape: tuple[int, int]
    has_mask_band: tuple[bool, ...]
    has_dataset_mask: bool
    overview_factors: list[int]

    @property
    def count(self) -> int:
        return len(self.dtypes)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(minx, miny, maxx, maxy), computed the same way as GDAL does."""
        tf = self.transform
        minx, maxy = tf.c, tf.f
        maxx = minx + self.width * tf.a
        miny = maxy + self.height * tf.e
        return minx, miny, maxx, maxy

    @classmethod
    def from_dataset(cls, path: str, ds: 'DatasetReader') -> '_VRTSourceInfo':
        colormaps = {}
        for bidx, ci in enumerate(ds.colorinterp, start=1):
            if ci == ColorInterp.palette:
                try:
                    colormaps[bidx] = ds.colormap(bidx)
                except ValueError:
                    pass
        # Same logic as gdalbuildvrt for deciding whether a source band
        # should be read with its mask band.
        mask_flags = [set(flags) for flags in ds.mask_flag_enums]
        has_mask_band = tuple(
            (flags != {MaskFlags.all_valid} and flags != {MaskFlags.nodata})
            or ci == ColorInterp.alpha
            for flags, ci in zip(mask_flags, ds.colorinterp))
        has_dataset_mask = (len(mask_flags) > 0
                            and MaskFlags.per_dataset in mask_flags[0])
        block_shape = ds.block_shapes[0] if ds.count > 0 else (0, 0)
        overview_factors = []
        if ds.count > 0:
            overview_factors = _get_vrt_overview_factors(
                ds.overviews(1), ds.width, ds.height)
        return cls(
            path=path,
            crs=ds.crs,
            transform=ds.transform,
            width=ds.width,
            height=ds.height,
            dtypes=tuple(ds.dtypes),
            nodatavals=tuple(ds.nodatavals),
            colorinterp=tuple(ds.colorinterp),
            scales=tuple(ds.scales),
            offsets=tuple(ds.offsets),
            colormaps=colormaps,
            block_shape=tuple(block_shape),
            has_mask_band=has_mask_band,
            has_dataset_mask=has_dataset_mask,
            overview_factors=overview_factors)


def _get_vrt_overview_factors(factors: list[int], width: int,
                              height: int) -> list[int]:
    """Get the power-of-2 overview factors usable for virtual overviews.

    Mirrors gdalbuildvrt: stop at the first overview smaller than 128x128 or
    at the first factor that does not follow the 2, 4, 8, ... sequence.
    """
    out = []
    expected_factor = 2
    for factor in factors:
        if (math.ceil(width / factor) < 128
                and math.ceil(height / factor) < 128):
            break
        if factor != expected_factor:
            break
        out.append(factor)
        expected_factor *= 2
    return out


def _check_vrt_source(src: _VRTSourceInfo,
                      ref: _VRTSourceInfo | None) -> str | None:
    """Return the reason why src can't be mosaiced with ref, if any."""
    tf = src.transform
    if tf.b != 0 or tf.d != 0:
        return 'Rotated geotransforms are not supported.'
    if tf.e >= 0:
        return ('Ungeoreferenced images and images with a positive '
                'north-south resolution are not supported.')
    if src.count == 0:
        return 'Dataset has no bands.'
    if ref is None:
        return None
    if src.crs != ref.crs:
        return (f'Heterogeneous projections are not supported: expected '
                f'{ref.crs}, got {src.crs}.')
    if src.count != ref.count:
        return (f'Heterogeneous band numbers are not supported: expected '
                f'{ref.count}, got {src.count}.')
    if src.dtypes != ref.dtypes:
        return (f'Heterogeneous band data types are not supported: expected '
                f'{ref.dtypes}, got {src.dtypes}.')
    if src.colorinterp != ref.colorinterp:
        exp = [ci.name for ci in ref.colorinterp]
        got = [ci.name for ci in src.colorinterp]
        return ('Heterogeneous band color interpretations are not '
                f'supported: expected {exp}, got {got}.')
    if src.scales != ref.scales or src.offsets != ref.offsets:
        return ('Heterogeneous band scales/offsets are not supported: '
                f'expected {ref.scales}/{ref.offsets}, '
                f'got {src.scales}/{src.offsets}.')
    return None


def _get_vrt_src_dst_rects(
        src: _VRTSourceInfo, we_res: float, ns_res: float, minx: float,
        maxy: float, vrt_width: int,
        vrt_height: int) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """Compute the SrcRect and DstRect of a source within the VRT.

    Mirrors GetSrcDstWin() in gdalbuildvrt for the case where the VRT
    extent is the union of the source extents.
    """
    tf = src.transform
    src_xoff, src_yoff = 0., 0.
    src_xsize, src_ysize = float(src.width), float(src.height)
    dst_xoff = (tf.c - minx) / we_res
    dst_yoff = (maxy - tf.f) / -ns_res
    src_to_dst_x = tf.a / we_res
    src_to_dst_y = tf.e / ns_res
    dst_xsize = src_xsize * src_to_dst_x
    dst_ysize = src_ysize * src_to_dst_y
    if dst_xoff + dst_xsize > vrt_width:
        dst_xsize = vrt_width - dst_xoff
        src_xsize = dst_xsize / src_to_dst_x
    if dst_yoff + dst_ysize > vrt_height:
        dst_ysize = vrt_height - dst_yoff
        src_ysize = dst_ysize / src_to_dst_y
    if min(src_xsize, src_ysize, dst_xsize, dst_ysize) <= 0:
        return None
    src_rect = (src_xoff, src_yoff, src_xsize, src_ysize)
    dst_rect = (dst_xoff, dst_yoff, dst_xsize, dst_ysize)
    # GDAL rounds values that are within 1e-3 of an integer
    src_rect = tuple(_round_if_close_to_int(v) for v in src_rect)
    dst_rect = tuple(_round_if_close_to_int(v) for v in dst_rect)
    return src_rect, dst_rect


def _round_if_close_to_int(value: float) -> float:
    closest_int = math.floor(value + 0.5)
    return float(closest_int) if abs(value - closest_int) < 1e-3 else value


def _gdal_dtype_name(dtype: str) -> str:
    return typename_fwd[dtype_rev[dtype]]


def _format_vrt_nodata(value: float) -> str:
    if math.isnan(value):
        return 'nan'
    return '%.18g' % value


def _add_vrt_source(band_el: ET.Element,
                    src: _VRTSourceInfo,
                    bidx: int,
                    src_rect: tuple[float, ...],
                    dst_rect: tuple[float, ...],
                    nodata: float | None = None,
                    use_mask_band: bool = False,
                    is_mask: bool = False) -> None:
    """Add a SimpleSource or ComplexSource element to a VRTRasterBand."""
    is_complex = nodata is not None or use_mask_band
    tag = 'ComplexSource' if is_complex else 'SimpleSource'
    src_el = ET.SubElement(band_el, tag)
    ET.SubElement(src_el, 'SourceFilename', relativeToVRT='0').text = src.path
    source_band = f'mask,{bidx}' if is_mask else str(bidx)
    ET.SubElement(src_el, 'SourceBand').text = source_band
    block_ysize, block_xsize = src.block_shape
    ET.SubElement(
        src_el,
        'SourceProperties',
        RasterXSize=str(src.width),
        RasterYSize=str(src.height),
        DataType=_gdal_dtype_name(src.dtypes[bidx - 1]),
        BlockXSize=str(block_xsize),
        BlockYSize=str(block_ysize))
    rect_keys = ('xOff', 'yOff', 'xSize', 'ySize')
    src_rect_attrs = {k: '%.15g' % v for k, v in zip(rect_keys, src_rect)}
    dst_rect_attrs = {k: '%.15g' % v for k, v in zip(rect_keys, dst_rect)}
    ET.SubElement(src_el, 'SrcRect', **src_rect_attrs)
    ET.SubElement(src_el, 'DstRect', **dst_rect_attrs)
    if use_mask_band:
        ET.SubElement(src_el, 'UseMaskBand').text = 'true'
    if nodata is not None:
        ET.SubElement(src_el, 'NODATA').text = _format_vrt_nodata(nodata)


def _get_vrt_colorinterp_name(ci: ColorInterp) -> str:
    """Map rasterio's ColorInterp to the name GDAL uses in VRTs."""
    special_names = {
        'Y': 'YCbCr_Y',
        'Cb': 'YCbCr_Cb',
        'Cr': 'YCbCr_Cr',
        'other_ir': 'OtherIR',
    }
    # GDAL parses these case-insensitively, but capitalize them anyway to
    # match what gdalbuildvrt writes for the common cases (Gray, Red, etc.).
    return special_names.get(ci.name, ci.name.capitalize())


def _get_vrt_common_overview_factors(sources: list[_VRTSourceInfo],
                                     we_res: float,
                                     ns_res: float) -> list[int]:
    """Get overview factors for the VRT's virtual overviews.

    Mirrors gdalbuildvrt: virtual overviews are only added if all sources
    have the same resolution as the VRT and share the same overview factors.
    """
    factors = set()
    for src in sources:
        tf = src.transform
        if (abs(tf.a - we_res) > 1e-8 * abs(we_res)
                or abs(tf.e - ns_res) > 1e-8 * abs(ns_res)):
            return []
        factors.update(src.overview_factors)
    factors = sorted(factors)
    for src in sources:
        for f in list(factors):
            if src.width // f < 128 and src.height // f < 128:
                break
            if f not in src.overview_factors:
                factors.remove(f)
    return factors


def build_vrt(vrt_path: str, image_uris: list[str]) -> None:
    """Build a VRT for a set of TIFF files.

    This is a pure-Python (rasterio-based) equivalent of running
    ``gdalbuildvrt vrt_path *image_uris`` with default options. All inputs
    must have the same CRS, number of bands, data types, and color
    interpretation, and must have north-up geotransforms; any input that does
    not satisfy this is skipped with a warning. The VRT's extent is the union
    of the extents of the inputs and its resolution is the average of their
    resolutions.

    Args:
        vrt_path (str): Local path for the VRT to be created.
        image_uris (list[str]): Image URIs.

    Raises:
        ValueError: If none of the images can be added to the VRT.
    """
    log.info('Building VRT...')
    image_uris_vsi = [uri_to_vsi_path(uri) for uri in image_uris]

    sources: list[_VRTSourceInfo] = []
    # the average resolution is computed using a running mean like GDAL does
    we_res, ns_res = 0., 0.
    with rio.Env():
        for path in image_uris_vsi:
            try:
                with rio.open(path) as ds:
                    src = _VRTSourceInfo.from_dataset(path, ds)
            except RasterioIOError as e:
                log.warning(f'Cannot open {path}: {e} Skipping it.')
                continue
            ref = sources[0] if len(sources) > 0 else None
            err_msg = _check_vrt_source(src, ref)
            if err_msg is not None:
                log.warning(f'{err_msg} Skipping {path}.')
                continue
            sources.append(src)
            we_res += (src.transform.a - we_res) / len(sources)
            ns_res += (src.transform.e - ns_res) / len(sources)

    if len(sources) == 0:
        raise ValueError('Cannot build VRT: none of the input images could '
                         'be used.')

    all_bounds = np.array([src.bounds for src in sources])
    minx, miny = all_bounds[:, :2].min(axis=0).tolist()
    maxx, maxy = all_bounds[:, 2:].max(axis=0).tolist()
    vrt_width = max(1, int(0.5 + (maxx - minx) / we_res))
    vrt_height = max(1, int(0.5 + (maxy - miny) / -ns_res))

    rects = []
    for src in sources:
        src_dst_rects = _get_vrt_src_dst_rects(src, we_res, ns_res, minx, maxy,
                                               vrt_width, vrt_height)
        if src_dst_rects is None:
            log.debug(f'Skipping {src.path} as it does not intersect the '
                      'VRT extent.')
            continue
        rects.append((src, *src_dst_rects))

    ref = sources[0]
    vrt_el = ET.Element(
        'VRTDataset', rasterXSize=str(vrt_width), rasterYSize=str(vrt_height))
    if ref.crs is not None:
        ET.SubElement(vrt_el, 'SRS').text = ref.crs.to_wkt()
    geotransform = (minx, we_res, 0., maxy, 0., ns_res)
    ET.SubElement(vrt_el, 'GeoTransform').text = ','.join(
        f'{v:24.16e}' for v in geotransform)

    for bidx in range(1, ref.count + 1):
        band_el = ET.SubElement(
            vrt_el,
            'VRTRasterBand',
            dataType=_gdal_dtype_name(ref.dtypes[bidx - 1]),
            band=str(bidx))
        nodata = ref.nodatavals[bidx - 1]
        if nodata is not None:
            ET.SubElement(band_el,
                          'NoDataValue').text = _format_vrt_nodata(nodata)
        if ref.offsets[bidx - 1] != 0:
            ET.SubElement(band_el,
                          'Offset').text = '%.16g' % ref.offsets[bidx - 1]
        if ref.scales[bidx - 1] != 1:
            ET.SubElement(band_el,
                          'Scale').text = '%.16g' % ref.scales[bidx - 1]
        ci = ref.colorinterp[bidx - 1]
        if ci != ColorInterp.undefined:
            ET.SubElement(band_el,
                          'ColorInterp').text = _get_vrt_colorinterp_name(ci)
        if ci == ColorInterp.palette and bidx in ref.colormaps:
            colormap = ref.colormaps[bidx]
            ct_el = ET.SubElement(band_el, 'ColorTable')
            for i in sorted(colormap.keys()):
                c = (tuple(colormap[i]) + (255, ))[:4]
                entry_attrs = {f'c{j}': str(v) for j, v in enumerate(c, 1)}
                ET.SubElement(ct_el, 'Entry', **entry_attrs)
        for src, src_rect, dst_rect in rects:
            src_nodata = src.nodatavals[bidx - 1]
            _add_vrt_source(
                band_el,
                src,
                bidx,
                src_rect,
                dst_rect,
                nodata=src_nodata,
                use_mask_band=(src_nodata is None
                               and src.has_mask_band[bidx - 1]))

    if any(src.has_dataset_mask for src in sources):
        mask_el = ET.SubElement(vrt_el, 'MaskBand')
        mask_band_el = ET.SubElement(mask_el, 'VRTRasterBand', dataType='Byte')
        for src, src_rect, dst_rect in rects:
            _add_vrt_source(
                mask_band_el,
                src,
                1,
                src_rect,
                dst_rect,
                use_mask_band=True,
                is_mask=True)

    overview_factors = _get_vrt_common_overview_factors(
        [src for src, *_ in rects], we_res, ns_res)
    if len(overview_factors) > 0:
        ET.SubElement(
            vrt_el, 'OverviewList', resampling='nearest').text = ' '.join(
                str(f) for f in overview_factors)

    make_dir(vrt_path, use_dirname=True)
    ET.indent(vrt_el)
    vrt_xml = ET.tostring(vrt_el, encoding='unicode')
    with open(vrt_path, 'w', encoding='utf-8') as f:
        f.write(vrt_xml + '\n')


def download_and_build_vrt(image_uris: list[str],
                           vrt_dir: str,
                           stream: bool = False) -> str:
    """Download images (if needed) and build a VRT for a set of TIFF files.

    Args:
        image_uris (list[str]): Image URIs.
        vrt_dir (str): Dir where the VRT will be created.
        stream (bool): If true, do not download images.
            Defaults to False.

    Returns:
        str: The path to the created VRT file.
    """
    if not stream:
        image_uris = [download_if_needed(uri) for uri in image_uris]
    vrt_path = join(vrt_dir, 'index.vrt')
    build_vrt(vrt_path, image_uris)
    return vrt_path


def read_window(dataset: 'DatasetReader',
                bands: int | Sequence[int] | None = None,
                window: tuple[tuple[int, int], tuple[int, int]] | None = None,
                is_masked: bool = False,
                out_shape: tuple[int, ...] | None = None,
                session: 'Session | None' = None) -> np.ndarray:
    """Load a window of an image using Rasterio.

    Args:
        dataset: a Rasterio dataset.
        bands: Band index or indices to read. Must be 1-indexed.
        window: ``((row_start, row_stop), (col_start, col_stop))`` or
            ``((y_min, y_max), (x_min, x_max))``. If ``None``, reads the entire
            raster. Defaults to ``None``.
        is_masked: If ``True``, read a masked array from rasterio.
            Defaults to ``False``.
        out_shape: (height, width) of the output chip. If ``None``, no
            resizing is done. Defaults to ``None``.
        session: Rasterio :class:`.Session`.

    Returns:
        np.ndarray: array of shape (height, width, channels).
    """
    if bands is not None:
        bands = tuple(bands)
    with rio.Env(session=session):
        im = dataset.read(
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
        for channel, nodataval in enumerate(dataset.nodatavals):
            if nodataval is not None and nodataval != 0:
                im[channel, im[channel] == nodataval] = 0
    else:
        for channel, src_band in enumerate(bands):
            src_band_0_indexed = src_band - 1
            nodataval = dataset.nodatavals[src_band_0_indexed]
            if nodataval is not None and nodataval != 0:
                im[channel, im[channel] == nodataval] = 0

    im = np.transpose(im, axes=[1, 2, 0])
    return im


def get_channel_order_from_dataset(dataset: 'DatasetReader') -> list[int]:
    """Get channel order from rasterio image dataset.

    Accounts for dataset's ``colorinterp`` if defined.

    Args:
        dataset (DatasetReader): Rasterio image dataset.

    Returns:
        list[int]: List of channel indices.
    """
    colorinterp = dataset.colorinterp
    if colorinterp:
        channel_order = [
            i for i, color_interp in enumerate(colorinterp)
            if color_interp != ColorInterp.alpha
        ]
    else:
        channel_order = list(range(0, dataset.count))
    return channel_order


def is_masked(dataset: 'DatasetReader') -> bool:
    """Check if dataset has any masks defined."""
    mask_flags = dataset.mask_flag_enums
    is_masked = any(m for m in mask_flags if m != MaskFlags.all_valid)
    return is_masked


def get_aws_session() -> 'Session':
    """Build a rasterio AWS session from environment variables."""
    try:
        from rastervision.aws_s3 import S3FileSystem
        requester_pays = S3FileSystem.get_request_payer()
    except ModuleNotFoundError:
        requester_pays = os.getenv('AWS_REQUEST_PAYER',
                                   '').lower() == 'requestor'
    session = AWSSession.from_environ(requester_pays=requester_pays)
    return session
