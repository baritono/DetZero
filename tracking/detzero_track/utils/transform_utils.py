from typing import Union

import numpy as np

from detzero_utils.shape_types import Float, shape_checked


@shape_checked
def yaw_filter(
    yaw: Union[Float[np.ndarray, "*shape"], float, np.floating],
) -> Union[Float[np.ndarray, "*shape"], float, np.floating]:
    """
    filter the heading into -pi ~ pi
    Args:
        yaw: np.ndarray or float, raw heading
    Returns:
        yaw: np.ndarray or float, filtered heading
    """
    pi2 = np.pi * 2

    if isinstance(yaw, np.ndarray):
        mask = np.abs(yaw) >= pi2
        yaw[mask] = yaw[mask] - np.floor(yaw[mask]/pi2)*pi2
        yaw[yaw > np.pi] -= pi2
        yaw[yaw <= -np.pi] += pi2
    else:
        if np.abs(yaw) >= pi2:
            yaw = yaw - np.floor(yaw/pi2)*pi2
            if yaw > np.pi: yaw -= pi2
            if yaw <= -np.pi: yaw += pi2

    return yaw


@shape_checked
def get_inverse_transform_mat(src_pose: Float[np.ndarray, "4 4"]) -> Float[np.ndarray, "4 4"]:
    """ 
    Args:
        src_pose: 4*4 transform pose include rotate matrix and translation
    Returns:
        reverse_pose: 4*4 inverse of transform pose
    """
    reverse_pose = np.zeros((4, 4), dtype=np.float32)
    reverse_pose[:3, :3] = src_pose[:3, :3].T
    reverse_pose[:3, 3:] = -(src_pose[:3, :3].T @ src_pose[:3, 3:])
    reverse_pose[3, 3] = 1

    return reverse_pose


@shape_checked
def transform_boxes3d(
    boxes: Float[np.ndarray, "N box_dim"],
    pose: Float[np.ndarray, "4 4"],
    inverse: bool = False,
) -> Float[np.ndarray, "N 7"]:
    """
    Args:
        boxes: (N, 7 + C) x,y,z,dx,dy,dz,heading,...; only the first 7 columns are used
        pose: 4*4 transform pose include rotate matrix and translation
        inverse: using inverse of transform pose if True, Fasle otherwise
    Returns:
        transformed_boxes: (N, 7) x,y,z,dx,dy,dz,heading
    """
    center = boxes[:, :3]
    center = np.concatenate([center, np.ones((center.shape[0], 1))], axis=-1)
    if inverse:
        pose = get_inverse_transform_mat(pose)
    center = center @ pose.T
    heading = yaw_filter(boxes[:, [6]] + np.arctan2(pose[1, 0], pose[0, 0]))

    return np.concatenate([center[:, :3], boxes[:, 3:6], heading], axis=-1)
