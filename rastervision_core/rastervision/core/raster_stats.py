import json
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from tqdm import tqdm

from rastervision.pipeline.file_system import str_to_file, file_to_str

CHIP_SZ = 300

if TYPE_CHECKING:
    from rastervision.core.data import RasterSource


def parallel_variance(mean_a, count_a, var_a, mean_b, count_b, var_b):
    """Compute the variance based on stats from two partitions of the data.

    See "Parallel Algorithm" in
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    Args:
        mean_a: the mean of partition a
        count_a: the number of elements in partition a
        var_a: the variance of partition a
        mean_b: the mean of partition b
        count_b: the number of elements in partition b
        var_b: the variance of partition b

    Return:
        the variance of the two partitions if they were combined
    """
    delta = mean_b - mean_a
    m_a = var_a * (count_a - 1)
    m_b = var_b * (count_b - 1)
    M2 = m_a + m_b + delta**2 * count_a * count_b / (count_a + count_b)
    var = M2 / (count_a + count_b - 1)
    return var


def parallel_mean(mean_a, count_a, mean_b, count_b):
    """Compute the mean based on stats from two partitions of the data.

    See "Parallel Algorithm" in
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    Args:
        mean_a: the mean of partition a
        count_a: the number of elements in partition a
        mean_b: the mean of partition b
        count_b: the number of elements in partition b

    Return:
        the mean of the two partitions if they were combined
    """
    mean = (count_a * mean_a + count_b * mean_b) / (count_a + count_b)
    return mean


class RasterStats():
    def __init__(self):
        self.means = None
        self.stds = None

    def compute(self,
                raster_sources: Sequence['RasterSource'],
                sample_prob: Optional[float] = None) -> None:
        """Compute the mean and stds over all the raster_sources.

        This ignores NODATA values.

        If sample_prob is set, then a subset of each scene is used to compute stats which
        speeds up the computation. Roughly speaking, if sample_prob=0.5, then half the
        pixels in the scene will be used. More precisely, the number of chips is equal to
        sample_prob * (width * height / 300^2), or 1, whichever is greater. Each chip is
        uniformly sampled from the scene with replacement. Otherwise, it uses a sliding
        window over the entire scene to compute stats.

        Args:
            raster_sources: list of RasterSource
            sample_prob: (float or None) between 0 and 1
        """
        nb_channels = raster_sources[0].num_channels

        def get_chip(raster_source, window):
            """Return chip or None if all values are NODATA."""
            chip = raster_source.get_raw_chip(window).astype(float)
            # Convert shape from [h,w,c] to [c,h*w]
            chip = np.reshape(np.transpose(chip, [2, 0, 1]), (nb_channels, -1))

            # Ignore NODATA values.
            chip[chip == 0.0] = np.nan
            if np.any(~np.isnan(chip)):
                return chip
            return None

        def sliding_chip_stream():
            """Get stream of chips using a sliding window of size 300."""
            for raster_source in raster_sources:
                with raster_source.activate():
                    extent = raster_source.get_extent()
                    # ensure chip_sz doesn't exceed raster bounds
                    _chip_sz = min(CHIP_SZ, *extent.size)
                    windows = extent.get_windows(_chip_sz, stride=CHIP_SZ)
                    for window in windows:
                        chip = get_chip(raster_source, window)
                        if chip is not None:
                            yield chip

        def random_chip_stream():
            """Get random stream of chips."""
            for raster_source in raster_sources:
                with raster_source.activate():
                    extent = raster_source.get_extent()
                    # ensure chip_sz doesn't exceed raster bounds
                    _chip_sz = min(CHIP_SZ, *extent.size)
                    chip_area = _chip_sz * _chip_sz

                    num_chips = round(
                        sample_prob * (extent.get_area() / chip_area))
                    num_chips = max(1, num_chips)
                    windows = [
                        extent.make_random_square(_chip_sz)
                        for _ in range(num_chips)
                    ]
                    for window in windows:
                        chip = get_chip(raster_source, window)
                        if chip is not None:
                            yield chip

        # For each chip, compute the mean and var of that chip and then update the
        # running mean and var.
        count = 0
        mean = np.zeros((nb_channels, ))
        var = np.zeros((nb_channels, ))
        chip_stream = (sliding_chip_stream()
                       if sample_prob is None else random_chip_stream())

        with tqdm(chip_stream, desc='Analyzing chips') as bar:
            for chip in bar:
                chip_means = np.nanmean(chip, axis=1)
                chip_vars = np.nanvar(chip, axis=1)
                chip_count = np.sum(chip[0] != np.nan)

                var = parallel_variance(chip_means, chip_count, chip_vars,
                                        mean, count, var)
                mean = parallel_mean(chip_means, chip_count, mean, count)
                count += chip_count

        self.means = mean
        self.stds = np.sqrt(var)

    def save(self, stats_uri: str) -> None:
        # Ensure lists
        means = list(self.means)
        stds = list(self.stds)
        stats = {'means': means, 'stds': stds}
        str_to_file(json.dumps(stats), stats_uri)

    @staticmethod
    def load(stats_uri):
        stats_json = json.loads(file_to_str(stats_uri))
        stats = RasterStats()
        stats.means = stats_json['means']
        stats.stds = stats_json['stds']
        return stats
