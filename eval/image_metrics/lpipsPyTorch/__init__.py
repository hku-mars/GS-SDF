import torch

from .modules.lpips import LPIPS


class Lpips:
    def __init__(self, net_type: str = "alex", version: str = "0.1"):
        self.criterion = LPIPS(net_type, version).cuda()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        r"""Function that measures
        Learned Perceptual Image Patch Similarity (LPIPS).

        Arguments:
            x, y (torch.Tensor): the input tensors to compare.
            net_type (str): the network type to compare the features:
                            'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
            version (str): the version of LPIPS. Default: 0.1.
        """
        return self.criterion(x, y)
