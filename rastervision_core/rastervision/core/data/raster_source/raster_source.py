from abc import ABC, abstractmethod

from rastervision.core.box import Box


class ChannelOrderError(Exception):
    def __init__(self, channel_order, num_channels):
        self.channel_order = channel_order
        self.num_channels = num_channels
        msg = 'The channel_order={} contains a channel index >= num_channels={}'
        super().__init__(msg.format(str(channel_order), num_channels))


class RasterSource(ABC):
    """A source of raster data.

    This should be subclassed when adding a new source of raster data such as
    a set of files, an API, a TMS URI schema, etc.
    """

    def __init__(self, channel_order, num_channels, raster_transformers=[]):
        """Construct a new RasterSource.

        Args:
            channel_order: list of channel indices to use when extracting chip from
                raw imagery.
            num_channels: Number of channels in the raw imagery before applying
                channel_order.
            raster_transformers: RasterTransformers used to transform chips
                whenever they are retrieved.
        """
        self.channel_order = channel_order
        self.num_channels = num_channels
        self.raster_transformers = raster_transformers

    def validate_channel_order(self, channel_order, num_channels):
        for c in channel_order:
            if c >= num_channels:
                raise ChannelOrderError(channel_order, num_channels)

    @abstractmethod
    def get_extent(self):
        """Return the extent of the RasterSource.

        Returns:
            Box in pixel coordinates with extent
        """
        pass

    @abstractmethod
    def get_dtype(self):
        """Return the numpy.dtype of this scene"""
        pass

    @abstractmethod
    def get_crs_transformer(self):
        """Return the associated CRSTransformer."""
        pass

    @abstractmethod
    def _get_chip(self, window):
        """Return the raw chip located in the window.

        Args:
            window: Box

        Returns:
            [height, width, channels] numpy array
        """
        pass

    def __getitem__(self, window):
        return self.get_chip(window)

    def get_chip(self, window):
        """Return the transformed chip in the window.

        Get a raw chip, extract subset of channels using channel_order, and then apply
        transformations.

        Args:
            window: Box

        Returns:
            np.ndarray with shape [height, width, channels]
        """
        chip = self._get_chip(window)

        if self.channel_order:
            chip = chip[:, :, self.channel_order]

        for transformer in self.raster_transformers:
            chip = transformer.transform(chip, self.channel_order)

        return chip

    def get_raw_chip(self, window):
        """Return raw chip without using channel_order or applying transforms.

        Args:
            window: (Box) the window for which to get the chip

        Returns:
            np.ndarray with shape [height, width, channels]
        """
        return self._get_chip(window)

    def get_image_array(self):
        """Return entire transformed image array.

        Not safe to call on very large RasterSources.
        """
        return self.get_chip(self.get_extent())

    def get_raw_image_array(self):
        """Return entire raw image without using channel_order or applying transforms.

        Not safe to call on very large RasterSources.
        """
        return self.get_raw_chip(self.get_extent())

    def get_transformed_window(self, window: Box,
                               inverse: bool = False) -> Box:
        """Apply the CRS transform to the window.

        Args:
            window (Box): A window in pixel coordinates.

        Returns:
            Box: A window in world coordinates.
        """
        crs_transformer = self.get_crs_transformer()
        if inverse:
            tf = crs_transformer.map_to_pixel
        else:
            tf = crs_transformer.pixel_to_map
        ymin, xmin, ymax, xmax = window
        xmin, ymin = tf((xmin, ymin))
        xmax, ymax = tf((xmax, ymax))
        window = Box(ymin, xmin, ymax, xmax)
        return window
