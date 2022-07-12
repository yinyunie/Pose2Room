# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Helper functions for calculating 2D and 3D bounding box IoU.

Collected and written by Charles R. Qi
Last modified: Jul 2019
"""
from __future__ import print_function

import numpy as np
from scipy.spatial import ConvexHull
import torch

def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.

    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**

    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return (outputList)

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (rqi): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    corners1 = corners1[[7, 6, 2, 3, 4, 5, 1, 0]]
    corners2 = corners2[[7, 6, 2, 3, 4, 5, 1, 0]]

    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])
    inter_vol = inter_area * max(0.0, ymax-ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,3] without new malloc:
    [A,3] -> [A,1,3] -> [A,B,3]
    [B,3] -> [1,B,3] -> [A,B,3]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,6].
      box_b: (tensor) bounding boxes, Shape: [B,6].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xyz = torch.min(box_a[:, 3:].unsqueeze(1).expand(A, B, 3),
                        box_b[:, 3:].unsqueeze(0).expand(A, B, 3))
    min_xyz = torch.max(box_a[:, :3].unsqueeze(1).expand(A, B, 3),
                        box_b[:, :3].unsqueeze(0).expand(A, B, 3))
    inter = torch.clamp((max_xyz - min_xyz), min=0)

    return inter[..., 0] * inter[..., 1] * inter[..., 2]

def box3d_iou_cuda(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,6]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,6]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    box_a = torch.cat([torch.min(box_a, dim=1)[0], torch.max(box_a, dim=1)[0]], dim=-1)
    box_b = torch.cat([torch.min(box_b, dim=1)[0], torch.max(box_b, dim=1)[0]], dim=-1)

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 3] - box_a[:, 0]) * (box_a[:, 4] - box_a[:, 1]) * (box_a[:, 5] - box_a[:, 2])).unsqueeze(
        1).expand_as(inter)
    area_b = ((box_b[:, 3] - box_b[:, 0]) * (box_b[:, 4] - box_b[:, 1]) * (box_b[:, 5] - box_b[:, 2])).unsqueeze(
        0).expand_as(inter)

    union = area_a + area_b - inter

    return inter / (union + 1e-12)

def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                    [0,  1,  0],
                    [-s, 0,  c]])

def corners2params(box3d_pts_3d):
    batch_flag = True
    if len(box3d_pts_3d.shape) == 2:
        box3d_pts_3d = box3d_pts_3d[np.newaxis]
        batch_flag = False

    center = (box3d_pts_3d.max(axis=1) + box3d_pts_3d.min(axis=1)) / 2.
    vectors = np.array([(box3d_pts_3d[:, 1] - box3d_pts_3d[:, 0]) / 2.,
                        (box3d_pts_3d[:, 2] - box3d_pts_3d[:, 1]) / 2.,
                        (box3d_pts_3d[:, 4] - box3d_pts_3d[:, 0]) / 2.])
    vectors = vectors.transpose([1, 0, 2])
    box_size = np.linalg.norm(vectors, axis=2) * 2

    size_normalize = 1 / (box_size / 2)
    m, n = size_normalize.shape
    size_normalize_diag = np.zeros((m, n, n), dtype=size_normalize.dtype)
    size_normalize_diag.reshape(-1, n ** 2)[..., ::n + 1] = size_normalize
    R_mat = np.matmul(size_normalize_diag, vectors)

    # right-hand system
    yflags = R_mat[:, 1].dot(np.array([0., 1., 0.]))
    R_mat[yflags < 0, 1] *= -1

    z_flags = np.einsum('ij,ij->i', np.cross(R_mat[:, 0], R_mat[:, 1]), R_mat[:, 2])
    R_mat[z_flags<0, 2] *= -1

    if not batch_flag:
        box_size = box_size[0]
        R_mat = R_mat[0]
        center = center[0]

    return box_size, R_mat, center

