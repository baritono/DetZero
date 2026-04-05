import torch.nn as nn

from ...structures import BatchDict


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict: BatchDict) -> BatchDict:
        """
        Args:
            batch_dict (BatchDict):
                encoded_spconv_tensor: SparseConvTensor, 3-D feature volume
                encoded_spconv_tensor_stride: int, spatial stride of the tensor
        Returns:
            batch_dict (BatchDict) with ``spatial_features`` (B, C*D, H, W) and
            ``spatial_features_stride`` added.

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.reshape(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
