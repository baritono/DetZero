from typing import Optional

import numpy as np
import torch

from detzero_utils.shape_types import Float, Int, Tensor, shape_checked


class ResidualCoder(object):
    def __init__(self, code_size=7, encode_angle_by_sincos=False, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        if self.encode_angle_by_sincos:
            self.code_size += 1

    @shape_checked
    def encode_torch(
        self,
        boxes: Float[Tensor, "*batch box_dim"],
        anchors: Float[Tensor, "*batch box_dim"],
    ) -> Float[Tensor, "*batch code_size"]:
        """
        Args:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:
            box_encodings: (N, code_size), code_size = 7 + C (+1 if encode_angle_by_sincos)
        """
        anchors[:, 3:6] = torch.clamp_min(anchors[:, 3:6], min=1e-5)
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / dza
        dxt = torch.log(dxg / dxa)
        dyt = torch.log(dyg / dya)
        dzt = torch.log(dzg / dza)
        if self.encode_angle_by_sincos:
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra]

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)

    @shape_checked
    def decode_torch(
        self,
        box_encodings: Float[Tensor, "*batch code_size"],
        anchors: Float[Tensor, "*batch box_dim"],
    ) -> Float[Tensor, "*batch box_dim"]:
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:
            boxes: same leading dims as the inputs, (..., 7 + C)
        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza

        if self.encode_angle_by_sincos:
            rg_cos = cost + torch.cos(ra)
            rg_sin = sint + torch.sin(ra)
            rg = torch.atan2(rg_sin, rg_cos)
        else:
            rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PreviousResidualDecoder(object):
    def __init__(self, code_size=7, **kwargs):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        Args:
            box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom values
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(lt) * dxa
        dyg = torch.exp(wt) * dya
        dzg = torch.exp(ht) * dza
        rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PreviousResidualRoIDecoder(object):
    def __init__(self, code_size=7, **kwargs):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        Args:
            box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom values
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(lt) * dxa
        dyg = torch.exp(wt) * dya
        dzg = torch.exp(ht) * dza
        rg = ra - rt

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PointResidualCoder(object):
    def __init__(self, code_size=8, use_mean_size=True, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.use_mean_size = use_mean_size
        if self.use_mean_size:
            self.mean_size = torch.from_numpy(np.array(kwargs['mean_size'])).cuda().float()
            assert self.mean_size.min() > 0

    @shape_checked
    def encode_torch(
        self,
        gt_boxes: Float[Tensor, "N box_dim"],
        points: Float[Tensor, "N 3"],
        gt_classes: Optional[Int[Tensor, " N"]] = None,
    ) -> Float[Tensor, "N code_size"]:
        """
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            points: (N, 3) [x, y, z]
            gt_classes: (N) [1, num_classes]
        Returns:
            box_coding: (N, 8 + C)
        """
        gt_boxes[:, 3:6] = torch.clamp_min(gt_boxes[:, 3:6], min=1e-5)

        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(gt_boxes, 1, dim=-1)
        xa, ya, za = torch.split(points, 1, dim=-1)

        if self.use_mean_size:
            assert gt_classes.max() <= self.mean_size.shape[0]
            point_anchor_size = self.mean_size[gt_classes - 1]
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xt = (xg - xa) / diagonal
            yt = (yg - ya) / diagonal
            zt = (zg - za) / dza
            dxt = torch.log(dxg / dxa)
            dyt = torch.log(dyg / dya)
            dzt = torch.log(dzg / dza)
        else:
            xt = (xg - xa)
            yt = (yg - ya)
            zt = (zg - za)
            dxt = torch.log(dxg)
            dyt = torch.log(dyg)
            dzt = torch.log(dzg)

        cts = [g for g in cgs]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, torch.cos(rg), torch.sin(rg), *cts], dim=-1)

    @shape_checked
    def decode_torch(
        self,
        box_encodings: Float[Tensor, "N code_size"],
        points: Float[Tensor, "N 3"],
        pred_classes: Optional[Int[Tensor, " N"]] = None,
    ) -> Float[Tensor, "N box_dim"]:
        """
        Args:
            box_encodings: (N, 8 + C) [x, y, z, dx, dy, dz, cos, sin, ...]
            points: (N, 3) [x, y, z]
            pred_classes: (N) [1, num_classes]
        Returns:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
        """
        xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)
        xa, ya, za = torch.split(points, 1, dim=-1)

        if self.use_mean_size:
            assert pred_classes.max() <= self.mean_size.shape[0]
            point_anchor_size = self.mean_size[pred_classes - 1]
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xg = xt * diagonal + xa
            yg = yt * diagonal + ya
            zg = zt * dza + za

            dxg = torch.exp(dxt) * dxa
            dyg = torch.exp(dyt) * dya
            dzg = torch.exp(dzt) * dza
        else:
            xg = xt + xa
            yg = yt + ya
            zg = zt + za
            dxg, dyg, dzg = torch.split(torch.exp(box_encodings[..., 3:6]), 1, dim=-1)

        rg = torch.atan2(sint, cost)

        cgs = [t for t in cts]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)
