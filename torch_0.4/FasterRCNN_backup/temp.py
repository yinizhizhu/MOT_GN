def IOU(Reframe, GTframe):
    """
    Compute the Intersection of Union
    :param Reframe: x, y, w, h
    :param GTframe: x, y, w, h
    :return: Ratio
    """
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2]
    height1 = Reframe[3]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2]
    height2 = GTframe[3]

    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width * height
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area1 + Area2 - Area)
        return ratio, Area/Area1, Area/Area2
    return ratio

# a = [1288, 464, 82, 183]
# b = [1299.7, 457.3, 73, 205.6]
# print IOU(a, b)
#
# bbx1 = ['533', '690', '89', '232']
# print bbx1,
# bbx1 = [float(p) for p in bbx1]
# print bbx1
# bbx2 = [' 431.766', ' 702.118', ' 71.7708', ' 233.574']
# print bbx2,
# bbx2 = [float(p) for p in bbx2]
# print bbx2
# print IOU(bbx1, bbx2)

import time
from math import *
import numpy as np
import matplotlib.pyplot as plt
from test_dataset_m import MDatasetFromFolder


def statistics(data_set):
    frames = data_set.bbx
    width = int(data_set.width)
    height = int(data_set.height)
    seqL = data_set.seqL
    widths = []
    heights = []
    areas = []
    for frame in frames:
        for bbx in frame:
            acreage = bbx[2]*bbx[3]
            areas.append(acreage)
            widths.append(bbx[0])
            heights.append(bbx[1])

    threshold = 0.001
    areas = sorted(areas)
    widths = sorted(widths)
    heights = sorted(heights)
    for i in xrange(len(areas)):
        if areas[i] >= threshold:
            break

    # print areas
    # print widths
    # print heights

    print '     ', width, height
    print '      %.3f%%'%(i*1.0/len(areas)*100)
    print '     ',
    lam = 0.4
    print areas[int(lam*len(areas))],
    print widths[int(lam*len(widths))],
    print heights[int(lam*len(heights))]

    # plt.hist(np.array(areas),bins = 256, normed = 1, facecolor = 'red', edgecolor = 'red', hold = 1)
    # plt.show()
    # plt.hist(np.array(widths),bins = 256, normed = 1, facecolor = 'green', edgecolor = 'green', hold = 1)
    # plt.show()
    # plt.hist(np.array(heights),bins = 256, normed = 1, facecolor = 'blue', edgecolor = 'blue', hold = 1)
    # plt.show()


# sequence_dir = ''
# year = 17
#
# seqs = [9, 11, 13]
#
# test_seqs = [1, 3, 6, 7, 8, 12, 14]
#
# start_a = time.time()
# if __name__ == '__main__':
#     try:
#         types = [['DPM0', -0.6], ['SDP', 0.5], ['FRCNN', 0.5]]
#         # types = [['DPM0', -0.6]]
#         # types = [['SDP', 0.5]]
#         # types = [['FRCNN', 0.5]]
#         for t in types:
#             type, tau_conf_score = t
#             print type
#             for i in xrange(len(seqs)):
#                 seq_index = seqs[i]
#
#                 # seq_dir = 'MOT%d-%02d-%s' % (year, test_seqs[i], type)
#                 # sequence_dir = '../MOT/MOT%d/test/'%year + seq_dir
#                 # print ' test:', sequence_dir
#                 # m_train_set = MDatasetFromFolder(sequence_dir, '../MOT/MOT16/train/MOT16-%02d' % seq_index, tau_conf_score)
#                 # statistics(m_train_set)
#
#                 seq_dir = 'MOT%d-%02d-%s' % (year, seq_index, type)
#                 sequence_dir = '../MOT/MOT%d/train/'%year + seq_dir
#                 print ' train:', sequence_dir
#                 m_train_set = MDatasetFromFolder(sequence_dir, '../MOT/MOT16/train/MOT16-%02d' % seq_index, tau_conf_score)
#                 statistics(m_train_set)
#
#             for i in xrange(len(test_seqs)):
#                 seq_index = test_seqs[i]
#
#                 seq_dir = 'MOT%d-%02d-%s' % (year, seq_index, type)
#                 sequence_dir = '../MOT/MOT%d/test/'%year + seq_dir
#                 print ' test:', sequence_dir
#                 m_train_set = MDatasetFromFolder(sequence_dir, '../MOT/MOT16/train/MOT16-%02d' % seq_index, tau_conf_score)
#                 statistics(m_train_set)
#
#     except KeyboardInterrupt:
#         print ''
#         print '-'*90
#         print 'Existing from training early.'
#         print 'Time consuming:', (time.time()-start_a)/60.0


import torch
import torch.nn as nn

m = nn.AdaptiveAvgPool1d(2)
a = torch.randn(1, 4, 6)
b = m(a)
print a.size()
print a

print ''
print b.size()
print b

b = [i+i*0.1 for i in xrange(10)]
c = [int(p) for p in b]
print b
print c[0:4]

import random

for i in xrange(0, 600, 600/300):
    print i


from easydict import EasyDict as edict

for x in xrange(1, 10):
    cf = x / 10.0
    for y in xrange(1, 10):
        iou = y / 10.0
        container = dict()
        outName = 'FasterRCNN/info_%.1f_%.1f.txt' % (cf, iou)
        container[outName] = edict()

        self.container[outName].all_det_acc_p = 0.0
        self.container[outName].all_det_acc_n = 0.0