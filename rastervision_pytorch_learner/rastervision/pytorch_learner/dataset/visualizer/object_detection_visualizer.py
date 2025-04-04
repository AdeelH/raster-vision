from collections.abc import Sequence

import torch

from rastervision.pytorch_learner.dataset.visualizer import Visualizer
from rastervision.pytorch_learner.object_detection_utils import (
    BoxList,
    collate_fn,
    draw_boxes,
)
from rastervision.pytorch_learner.utils import (
    channel_groups_to_imgs,
    plot_channel_groups,
)


class ObjectDetectionVisualizer(Visualizer):
    """Plots samples from object detection Datasets."""

    def get_collate_fn(self):
        return collate_fn

    def plot_xyz(
        self,
        axs: Sequence,
        x: torch.Tensor,
        y: BoxList | None = None,
        z: BoxList | None = None,
        plot_title: bool = True,
    ) -> None:
        channel_groups = self.get_channel_display_groups(x.shape[1])
        imgs = channel_groups_to_imgs(x, channel_groups)

        if y is not None or z is not None:
            y = y if z is None else z
            class_names = self.class_names
            class_colors = self.class_colors
            imgs = [
                draw_boxes(img, y, class_names, class_colors) for img in imgs
            ]

        plot_channel_groups(axs, imgs, channel_groups, plot_title=plot_title)
