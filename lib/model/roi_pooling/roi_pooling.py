import six
import numpy as np
import torch
from torch.autograd import Function
from torch import nn

def _roi_pooling_slice(size, stride, max_size, roi_offset):
    start = int(np.floor(size * stride))
    end = int(np.ceil((size + 1) * stride))

    start = min(max(start + roi_offset, 0), max_size)
    end = min(max(end + roi_offset, 0), max_size)

    return slice(start, end), end - start

class roi_pooling_3d(Function):
    """

    """
    def __init__(self, outl, outh, outw, spatial_scale):
        self.outl, self.outh, self.outw = outl, outh, outw
        self.spatial_scale = spatial_scale

    def forward(self, features, rois):
        """
        Args:

            rois (array): [[index, zmin,ymin,xmin,zmax,ymax,xmax],]
        """

        self.features_shape = features_shape = features.shape
        channels, long, height, width = features_shape[1:]
        self.n_rois = n_rois = rois.shape[0]
        self.rois = rois

        output = np.zeros((n_rois, channels, self.outl, self.outh, self.outw),
                               dtype=features.cpu().data.numpy().dtype)
        self.argmax_data = np.zeros(output.shape, np.int32)

        for i_roi in six.moves.range(n_rois):
            idx, zmin, ymin, xmin, zmax, ymax, xmax = rois[i_roi]
            zmin = int(torch.round(zmin * self.spatial_scale))
            zmax = int(torch.round(zmax * self.spatial_scale))
            ymin = int(torch.round(ymin * self.spatial_scale))
            ymax = int(torch.round(ymax * self.spatial_scale))
            xmin = int(torch.round(xmin * self.spatial_scale))
            xmax = int(torch.round(xmax * self.spatial_scale))
            roi_long = max(zmax - zmin + 1, 1)
            roi_height = max(ymax - ymin + 1, 1)
            roi_width = max(xmax - xmin + 1, 1)
            stridel = 1. * roi_long / self.outl
            strideh = 1. * roi_height / self.outh
            stridew = 1. * roi_width / self.outw

            for outl in six.moves.range(self.outl):
                slicel, lenl = _roi_pooling_slice(
                    outl, stridel, long, zmin)
                if slicel.stop <= slicel.start:
                    continue
                for outh in six.moves.range(self.outh):
                    sliceh, lenh = _roi_pooling_slice(
                        outh, strideh, height, ymin)
                    if sliceh.stop <= sliceh.start:
                        continue
                    for outw in six.moves.range(self.outw):
                        slicew, lenw = _roi_pooling_slice(
                            outw, stridew, width, xmin)
                        if slicew.stop <= slicew.start:
                            continue
                        roi_data = features[int(idx), :, slicel, sliceh, slicew]\
                            .reshape(channels, -1).cpu().numpy()
                        output[i_roi, :, outl, outh, outw] =\
                            np.max(roi_data, axis=1)

                        # get the max idx respect to feature_maps coordinates
                        max_idx_slice = np.unravel_index(
                            np.argmax(roi_data, axis=1), (lenl, lenh, lenw))
                        max_idx_slice_l = max_idx_slice[0] + slicel.start
                        max_idx_slice_h = max_idx_slice[1] + sliceh.start
                        max_idx_slice_w = max_idx_slice[2] + slicew.start
                        max_idx_slice = (max_idx_slice_l * height +\
                                     max_idx_slice_h) * width + max_idx_slice_w
                        self.argmax_data[i_roi, :, outl, outh, outw] = max_idx_slice

        return torch.from_numpy(output)

    def backward(self, grad_output): 

        features_shape = self.features_shape
        channels, long, height, width = features_shape[1:]
        n_rois = self.n_rois
        rois, device = self.rois, self.rois.device

        grad_features = np.zeros(self.features_shape, self.rois.cpu().data.numpy().dtype)

        for i_roi in six.moves.range(self.n_rois):
            idx, zmin, ymin, xmin, zmax, ymax, xmax = rois[i_roi]
            idx = int(idx)
            zmin = int(torch.round(zmin * self.spatial_scale))
            zmax = int(torch.round(zmax * self.spatial_scale))
            ymin = int(torch.round(ymin * self.spatial_scale))
            ymax = int(torch.round(ymax * self.spatial_scale))
            xmin = int(torch.round(xmin * self.spatial_scale))
            xmax = int(torch.round(xmax * self.spatial_scale))
            roi_long = max(zmax - zmin + 1, 1)
            roi_height = max(ymax - ymin + 1, 1)
            roi_width = max(xmax - xmin + 1, 1)

            stridel = float(roi_long) / float(self.outl)
            strideh = float(roi_height) / float(self.outh)
            stridew = float(roi_width) / float(self.outw)


            
            # iterate all the w, h (from feature map) that fall into this ROIs
            for w in six.moves.range(xmin, xmax + 1):
                for h in six.moves.range(ymin, ymax + 1):
                    for l in six.moves.range(zmin, zmax + 1):
                        plstart = int(np.floor(float(l - zmin) / stridel))
                        plend = int(np.ceil(float(l - zmin + 1) / stridel))
                        phstart = int(np.floor(float(h - ymin) / strideh))
                        phend = int(np.ceil(float(h - ymin + 1) / strideh))
                        pwstart = int(np.floor(float(w - xmin) / stridew))
                        pwend = int(np.ceil(float(w - xmin + 1) / stridew))

                        # print('-'*40)
                        # print(phstart, phend, pwstart, pwend)

                        plstart = min(max(plstart, 0), self.outl)
                        plend = min(max(plend, 0), self.outl)                        
                        phstart = min(max(phstart, 0), self.outh)
                        phend = min(max(phend, 0), self.outh)
                        pwstart = min(max(pwstart, 0), self.outw)
                        pwend = min(max(pwend, 0), self.outw)

                        # print('w,h: ', w, h)
                        # unNOTE: not understand
                        for pl in six.moves.range(plstart, plend):
                            for ph in six.moves.range(phstart, phend):
                                for pw in six.moves.range(pwstart, pwend):
                                    max_idx_tmp = self.argmax_data[i_roi, :, pl, ph, pw]
                                    for c in six.moves.range(channels):
                                        if max_idx_tmp[c] == ((l*height + h) *\
                                                                     width + w):
                                            grad_features[idx, c, l, h, w] += \
                                                grad_output[i_roi, c, pl, ph, pw]

        return torch.from_numpy(grad_features).to(device=device), \
                torch.zeros_like(rois, device=device)