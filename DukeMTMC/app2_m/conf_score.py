import random, torch, h5py
from PIL import Image
import numpy as np
from scipy.io import loadmat

SEQLEN = 177841


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class cf():
    def __init__(self, cuda=True):
        self.width, self.height = 1920.0, 1080.0

        self.device = torch.device("cuda" if cuda else "cpu")

        out = open('Detections/conf_score.txt', 'w')
        self.acc_det = [0.0 for i in xrange(9)]
        self.all_det = [0.0 for i in xrange(9)]
        self.all_gt = 0.0
        for camera in xrange(1, 9):
            self.seq_index = camera
            self.readGT()
            gt = self.accumulateGT()
            self.all_gt += gt
            print 'Camera_%d: '%camera
            print >> out, 'For camrera_%d:'%camera
            for gap in xrange(9):
                self.tau_conf_score = 0.3 + gap * 0.05
                self.readDet()
                det = self.accumulateDet()
                acc = self.accuracy()
                self.acc_det[gap] += acc
                self.all_det[gap] += det
                output = '     tau_conf_score (%.2f): %.6f(acc:%.1f/det:%.1f), %.6f(acc:%.1f/gt:%.1f), %.6f(acc_det:%.1f/all_gt:%.1f)'%(
                    self.tau_conf_score,
                    acc/det, acc, det,
                    acc/gt, acc, gt,
                    self.acc_det[gap]/self.all_gt, self.acc_det[gap], self.all_gt
                )
                print >> out, output
                print output
            print ''
        out.close()

    def accuracy(self):
        ans = 0.0
        for i in xrange(1, SEQLEN + 1):
            for j in xrange(len(self.bbx[i])):
                for k in xrange(len(self.gt[i])):
                    if self.IOU(self.bbx[i][j], self.gt[i][k]) >= 0.5:
                        ans += 1.0
                        break
        return ans

    def accumulateDet(self):
        ans = 0.0
        for i in xrange(1, SEQLEN + 1):
            ans += len(self.bbx[i])
        return ans

    def accumulateGT(self):
        ans = 0.0
        for i in xrange(1, SEQLEN + 1):
            ans += len(self.gt[i])
        return ans

    def IOU(self, Reframe, GTframe):
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
        return ratio

    def fixBBx(self, x1, y1, x2, y2):
        width = x2 - x1
        height = y2 - y1
        x1 -= width/12
        y1 -= height/12
        return x1, y1, width*7/6, height*7/6

    def getBBx(self, pose):
        x1, x2, y1, y2 = 10000.0, 0.0, 10000.0, 0.0
        conf_score = []
        for i in xrange(0, 54, 3):
            x, y, conf = float(pose[2 + i]*self.width), float(pose[3 + i]*self.height), float(pose[4 + i])
            # print x, y, conf
            if conf == 0.0:
                continue
            conf_score.append(conf)
            if x < x1:
                x1 = x
            if x > x2:
                x2 = x
            if y < y1:
                y1 = y
            if y > y2:
                y2 = y

        conf_score = sum(conf_score)/len(conf_score)
        if conf_score >= self.tau_conf_score:
            x1, y1, w, h = self.fixBBx(x1, y1, x2, y2)
            return [x1, y1, w, h]
        return None

    def readDet(self):
        # get the det
        self.bbx = [[] for i in xrange(SEQLEN + 1)]

        heads = [0, 5543, 3607, 27244, 31182, 1, 22402, 18968, 46766]
        test = [49700, 227540]

        detection_dir = '/media/lee/DATA/DukeMTMC/detections/openpose/camera%d.mat'%self.seq_index
        det = h5py.File(detection_dir)
        det = det['detections']
        det = np.transpose(det)

        head = test[0] - heads[self.seq_index] + 1
        tail = head + SEQLEN - 1

        for bbx in det:
            frame = int(bbx[1])
            bbx = self.getBBx(bbx)
            if bbx is not None and frame >= head and frame <= tail:
                x, y, w, h = bbx
                self.bbx[frame-head+1].append([x, y, w, h, frame])

    def readGT(self):
        # get the gt
        self.gt = [[] for i in xrange(SEQLEN + 1)]

        heads = [0, 5543, 3607, 27244, 31182, 1, 22402, 18968, 46766]
        test = [49700, 227540]

        trainval_dir = '/media/lee/DATA/DukeMTMC/ground_truth/trainval.mat'
        val = loadmat(trainval_dir)
        val = val['trainData']

        head = test[0] - heads[self.seq_index] + 1
        tail = head + SEQLEN - 1

        for bbx in val:
            camera = int(bbx[0])
            id = int(bbx[1])
            frame = int(bbx[2])
            x, y = float(bbx[3]), float(bbx[4])
            w, h = float(bbx[5]), float(bbx[6])
            if camera == self.seq_index and (frame >= head and frame <= tail):
                self.gt[frame-head+1].append([x, y, w, h, id, frame])

cf()