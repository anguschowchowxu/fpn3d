import torch
import numpy as np
import pdb

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 3] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 4] - ex_rois[:, 1] + 1.0
    ex_longs = ex_rois[:, 5] - ex_rois[:, 2] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
    ex_ctr_z = ex_rois[:, 2] + 0.5 * ex_longs

    gt_widths = gt_rois[:, 3] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 4] - gt_rois[:, 1] + 1.0
    gt_longs = gt_rois[:, 5] - gt_rois[:, 2] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights
    gt_ctr_z = gt_rois[:, 2] + 0.5 * gt_longs

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dz = (gt_ctr_z - ex_ctr_z) / ex_longs
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)
    targets_dl = torch.log(gt_longs / ex_longs)

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dz, targets_dw, targets_dh, targets_dl), 1)

    return targets

def bbox_transform_batch(ex_rois, gt_rois):

    if ex_rois.dim() == 2:
        ex_widths = ex_rois[:, 3] - ex_rois[:, 0] + 1.0
        ex_heights = ex_rois[:, 4] - ex_rois[:, 1] + 1.0
        ex_longs = ex_rois[:, 5] - ex_rois[:, 2] + 1.0
        ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
        ex_ctr_z = ex_rois[:, 2] + 0.5 * ex_longs

        gt_widths = gt_rois[:, :, 3] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 4] - gt_rois[:, :, 1] + 1.0
        gt_longs = gt_rois[:, :, 5] - gt_rois[:, :, 2] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights
        gt_ctr_z = gt_rois[:, :, 2] + 0.5 * gt_longs

        targets_dx = (gt_ctr_x - ex_ctr_x.view(1,-1).expand_as(gt_ctr_x)) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y.view(1,-1).expand_as(gt_ctr_y)) / ex_heights
        targets_dz = (gt_ctr_z - ex_ctr_z.view(1,-1).expand_as(gt_ctr_z)) / ex_longs
        targets_dw = torch.log(gt_widths / ex_widths.view(1,-1).expand_as(gt_widths))
        targets_dh = torch.log(gt_heights / ex_heights.view(1,-1).expand_as(gt_heights))
        targets_dl = torch.log(gt_longs / ex_longs.view(1,-1).expand_as(gt_longs))

    elif ex_rois.dim() == 3:
        ex_widths = ex_rois[:, :, 3] - ex_rois[:, :, 0] + 1.0
        ex_heights = ex_rois[:,:, 4] - ex_rois[:,:, 1] + 1.0
        ex_longs = ex_rois[:,:, 5] - ex_rois[:,:, 2] + 1.0
        ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights
        ex_ctr_z = ex_rois[:, :, 2] + 0.5 * ex_longs

        gt_widths = gt_rois[:, :, 3] - gt_rois[:, :, 0] + 1.0
        gt_heights = gt_rois[:, :, 4] - gt_rois[:, :, 1] + 1.0
        gt_longs = gt_rois[:, :, 5] - gt_rois[:, :, 2] + 1.0
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights
        gt_ctr_z = gt_rois[:, :, 2] + 0.5 * gt_longs

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dz = (gt_ctr_z - ex_ctr_z) / ex_longs
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)
        targets_dl = torch.log(gt_longs / ex_longs)
    else:
        raise ValueError('ex_roi input dimension is not correct.')

    targets = torch.stack(
        (targets_dx, targets_dy, targets_dz, targets_dw, targets_dh, targets_dl), 2)

    return targets

def bbox_transform_inv(boxes, deltas, batch_size):
    widths = boxes[:, :, 3] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 4] - boxes[:, :, 1] + 1.0
    longs = boxes[:, :, 5] - boxes[:, :, 2] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights
    ctr_z = boxes[:, :, 2] + 0.5 * longs

    dx = deltas[:, :, 0::6]
    dy = deltas[:, :, 1::6]
    dz = deltas[:, :, 2::6]
    dw = deltas[:, :, 3::6]
    dh = deltas[:, :, 4::6]
    dl = deltas[:, :, 5::6]

    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_ctr_z = dy * longs.unsqueeze(2) + ctr_z.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)
    pred_l = torch.exp(dl) * longs.unsqueeze(2)

    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, :, 0::6] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::6] = pred_ctr_y - 0.5 * pred_h
    # z1
    pred_boxes[:, :, 2::6] = pred_ctr_z - 0.5 * pred_l
    # x2
    pred_boxes[:, :, 3::6] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 4::6] = pred_ctr_y + 0.5 * pred_h
    # z2
    pred_boxes[:, :, 5::6] = pred_ctr_z + 0.5 * pred_l

    return pred_boxes

def clip_boxes_batch(boxes, im_shape, batch_size):
    """
    Clip boxes to image boundaries.
    """
    num_rois = boxes.size(1)

    boxes[boxes < 0] = 0
    # batch_x = (im_shape[:,0]-1).view(batch_size, 1).expand(batch_size, num_rois)
    # batch_y = (im_shape[:,1]-1).view(batch_size, 1).expand(batch_size, num_rois)

    batch_x = im_shape[:, 2] - 1
    batch_y = im_shape[:, 1] - 1
    batch_z = im_shape[:, 0] - 1

    boxes[:,:,0][boxes[:,:,0] > batch_x] = batch_x
    boxes[:,:,1][boxes[:,:,1] > batch_y] = batch_y
    boxes[:,:,2][boxes[:,:,2] > batch_z] = batch_z
    boxes[:,:,3][boxes[:,:,3] > batch_x] = batch_x
    boxes[:,:,4][boxes[:,:,4] > batch_y] = batch_y
    boxes[:,:,5][boxes[:,:,5] > batch_z] = batch_z

    return boxes

def clip_boxes(boxes, im_shape, batch_size):

    batch_x = im_shape[:, 2] - 1
    batch_y = im_shape[:, 1] - 1
    batch_z = im_shape[:, 0] - 1

    for i in range(batch_size):
        boxes[i,:,0::6].clamp_(0, batch_x-1)
        boxes[i,:,1::6].clamp_(0, batch_y-1)
        boxes[i,:,2::6].clamp_(0, batch_z-1)
        boxes[i,:,3::6].clamp_(0, batch_x-1)
        boxes[i,:,4::6].clamp_(0, batch_y-1)
        boxes[i,:,5::6].clamp_(0, batch_z-1)

    return boxes


def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 6) ndarray of float
    gt_boxes: (K, 6) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:,3] - gt_boxes[:,0] + 1) *
                (gt_boxes[:,4] - gt_boxes[:,1] + 1) *
                (gt_boxes[:,5] - gt_boxes[:,2] + 1)).view(1, K)

    anchors_area = ((anchors[:,3] - anchors[:,0] + 1) *
                (anchors[:,4] - anchors[:,1] + 1) *
                (anchors[:,5] - anchors[:,2] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 6).expand(N, K, 6)
    query_boxes = gt_boxes.view(1, K, 6).expand(N, K, 6)

    iw = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
        torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,4], query_boxes[:,:,4]) -
        torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    il = (torch.min(boxes[:,:,5], query_boxes[:,:,5]) -
        torch.max(boxes[:,:,2], query_boxes[:,:,2]) + 1)
    il[il < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih * il)
    overlaps = iw * ih * il / ua

    return overlaps

def bbox_overlaps_batch(anchors, gt_boxes):
    """
    anchors: (N, 6) ndarray of float
    gt_boxes: (b, K, 7) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_boxes.size(0)


    if anchors.dim() == 2:

        N = anchors.size(0)
        K = gt_boxes.size(1)

        anchors = anchors.view(1, N, 6).expand(batch_size, N, 6).contiguous()
        gt_boxes = gt_boxes[:,:,:6].contiguous()


        gt_boxes_x = (gt_boxes[:,:,3] - gt_boxes[:,:,0] + 1)
        gt_boxes_y = (gt_boxes[:,:,4] - gt_boxes[:,:,1] + 1)
        gt_boxes_z = (gt_boxes[:,:,5] - gt_boxes[:,:,2] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y * gt_boxes_z).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:,:,3] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,4] - anchors[:,:,1] + 1)
        anchors_boxes_z = (anchors[:,:,5] - anchors[:,:,2] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y * anchors_boxes_z).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1) & (get_boxes_z)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1) & (anchors_boxes_z == 1)

        boxes = anchors.view(batch_size, N, 1, 6).expand(batch_size, N, K, 6)
        query_boxes = gt_boxes.view(batch_size, 1, K, 6).expand(batch_size, N, K, 6)

        iw = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,4], query_boxes[:,:,:,4]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0

        il = (torch.min(boxes[:,:,:,5], query_boxes[:,:,:,5]) -
            torch.max(boxes[:,:,:,2], query_boxes[:,:,:,2]) + 1)
        il[il < 0] = 0

        ua = anchors_area + gt_boxes_area - (iw * ih * il)
        overlaps = iw * ih * il / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

    elif anchors.dim() == 3:
        N = anchors.size(1)
        K = gt_boxes.size(1)

        if anchors.size(2) == 4:
            anchors = anchors[:,:,:6].contiguous()
        else:
            anchors = anchors[:,:,1:7].contiguous()

        gt_boxes = gt_boxes[:,:,:6].contiguous()

        gt_boxes_x = (gt_boxes[:,:,3] - gt_boxes[:,:,0] + 1)
        gt_boxes_y = (gt_boxes[:,:,4] - gt_boxes[:,:,1] + 1)
        gt_boxes_z = (gt_boxes[:,:,5] - gt_boxes[:,:,2] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y * gt_boxes_z).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:,:,3] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,4] - anchors[:,:,1] + 1)
        anchors_boxes_z = (anchors[:,:,5] - anchors[:,:,2] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y * anchors_boxes_z).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1) & (gt_boxes_z == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1) & (anchors_boxes_z == 1)

        boxes = anchors.view(batch_size, N, 1, 6).expand(batch_size, N, K, 6)
        query_boxes = gt_boxes.view(batch_size, 1, K, 6).expand(batch_size, N, K, 6)

        iw = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,4], query_boxes[:,:,:,4]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0

        il = (torch.min(boxes[:,:,:,5], query_boxes[:,:,:,5]) -
            torch.max(boxes[:,:,:,2], query_boxes[:,:,:,2]) + 1)
        il[il < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih * il)

        overlaps = iw * ih * il / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)
    else:
        raise ValueError('anchors input dimension is not correct.')

    return overlaps