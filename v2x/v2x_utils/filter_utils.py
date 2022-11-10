from .geometry_utils import point_in_box
from config import superclass
from utils.setup_log import trace_logger
import numpy as np
np.seterr(invalid='ignore')


def diff_label_filt(frame1, frame2, i, j):
    trace_logger.warning(f'diff_label_filt(..)')
    size = frame1.size[i]
    '''
    if size.all() <= 0:
        print(f'num_boxes: {frame1.num_boxes}')
        print(f'size: {size}, {type(frame1)} {frame1} {frame2}')
    '''
    diff = np.abs(frame1.center[i] - frame2.center[j]) / size
    return diff[0] <= 1 and diff[1] <= 1 and diff[2] <= 1 and frame1.label[i] == frame2.label[j]


class Filter(object):
    def __init__(self):
        pass

    def __call__(self, **args):
        return True


class RectFilter(Filter):
    def __init__(self, bbox):
        super().__init__()
        self.bbox = bbox

    def __call__(self, box, **args):
        trace_logger.warning(f'RectFilter::__call__(..)')
        for corner in box:
            if point_in_box(corner, self.bbox):
                return True
        return False


class SuperClassFilter(Filter):
    def __init__(self, superclass):
        super().__init__()
        self.superclass = superclass

    def __call__(self, box, pred_class):
        trace_logger.warning(f'SuperClassFilter::__call__(..)')
        return superclass[pred_class] == self.superclass


class AndFilter(Filter):
    def __init__(self, filt1, filt2):
        super().__init__()
        self.filt1 = filt1
        self.filt2 = filt2

    def __call__(self, box, pred_class, **args):
        trace_logger.warning(f'AndFilter::__call__(..)')
        return self.filt1(box, pred_class) or self.filt2(box, pred_class)
