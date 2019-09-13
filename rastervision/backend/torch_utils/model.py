import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from rastervision.backend.torch_utils.boxlist import BoxList


class MyFasterRCNN(nn.Module):
    """Adapter around torchvision Faster-RCNN.

    The purpose of the adapter is to use a different input and output format
    and inject bogus boxes to circumvent torchvision's inability to handle
    training examples with no ground truth boxes.
    """

    def __init__(self, num_labels, img_sz, pretrained=True):
        super().__init__()

        self.model = fasterrcnn_resnet50_fpn(
            pretrained=False,
            progress=True,
            num_classes=num_labels,
            pretrained_backbone=pretrained,
            min_size=img_sz,
            max_size=img_sz)
        self.subloss_names = [
            'total_loss', 'loss_box_reg', 'loss_classifier', 'loss_objectness',
            'loss_rpn_box_reg'
        ]

    def forward(self, input, targets=None):
        """Forward pass

        Args:
            input: tensor<n, 3, h, w> with batch of images
            targets: None or list<BoxList> of length n with boxes and labels

        Returns:
            if targets is None, returns list<BoxList> of length n, containing
            boxes, labels, and scores for boxes with score > 0.05. Further
            filtering based on score should be done before considering the
            prediction "final".

            if targets is a list, returns the losses as dict with keys from
            self.subloss_names.
        """
        if targets:
            # Add bogus background class box for each image to workaround
            # the inability of torchvision to train on images with
            # no ground truth boxes. This is important for being able
            # to handle negative chips generated by RV.
            new_targets = []
            for x, y in zip(input, targets):
                h, w = x.shape[1:]
                boxes = torch.cat(
                    [y.boxes, torch.tensor([[0., 0, h, w]], device=input.device)], dim=0)
                labels = torch.cat(
                    [y.get_field('labels'),
                     torch.tensor([0], device=input.device)], dim=0)
                bl = BoxList(boxes, labels=labels)
                new_targets.append(bl)
            targets = new_targets

            _targets = [bl.xyxy() for bl in targets]
            _targets = [{
                'boxes': bl.boxes,
                'labels': bl.get_field('labels')
            } for bl in _targets]
            loss_dict = self.model(input, _targets)
            loss_dict['total_loss'] = sum(list(loss_dict.values()))
            return loss_dict

        out = self.model(input)
        boxlists = [
            BoxList(
                _out['boxes'], labels=_out['labels'],
                scores=_out['scores']).yxyx() for _out in out
        ]

        # Remove bogus background boxes.
        new_boxlists = []
        for bl in boxlists:
            labels = bl.get_field('labels')
            non_zero_inds = labels != 0
            new_boxlists.append(bl.ind_filter(non_zero_inds))
        return new_boxlists
