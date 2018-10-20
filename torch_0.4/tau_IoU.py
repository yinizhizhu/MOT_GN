import shutil, os, time
from PIL import Image
import matplotlib.pyplot as plt
from m_global_set import tau_conf_score
import scipy.stats as st

#   sum             max        min      mean
#               DPM
# 32354.9314411 0.990939826548 0.5 0.765434857844
# False negative:140.861131 %, (59542/42270)
# Missing: 0.000000 % (0/42270)
# Zero: 0
#               SDP
# 67343.8719538 1.0 0.500022322425 0.821046450389
# False negative:24.127673 %, (19790/82022)
# Missing: 0.000000 % (0/82022)
# Zero: 0
#               FRCNN
# 56449.5548949 0.997486530172 0.5 0.875797919399
# False negative:57.958265 %, (37357/64455)
# Missing: 0.000000 % (0/64455)
# Zero: 0


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class tau_iou():
    def __init__(self):
        types = ['DPM', 'SDP', 'FRCNN']
        seqs = [2, 4, 5, 9, 10, 11, 13]  # the set of sequences
        lengths = [600, 1050, 837, 525, 654, 900, 750]  # the length of the sequence

        for type in types:
            head = time.time()
            self.IOU = []
            self.false_negative = 0
            self.missing = 0
            self.zero = 0
            for i in xrange(7):
                start = time.time()
                part_I = '../MOT/MOT16/train/MOT16-%02d'%seqs[i]
                part = '../MOT/MOT17/train/MOT17-%02d-%s'%(seqs[i], type)
                self.dir = part
                self.cleanPath(part)
                self.img_dir = part_I + '/img1/'
                self.gt_dir = part + '/gt/'
                self.det_dir = part + '/det/'

                self.seqL = lengths[i]
                self.readBBx()
                self.doit()
                print 'Time consuming:', (time.time() - start) / 60.0
            print 'Time consuming:', (time.time() - head) / 60.0
            total = len(self.IOU)*1.0
            self.IOU = sorted(self.IOU)
            s = sum(self.IOU)
            mx = max(self.IOU)
            mn = min(self.IOU)
            mean = s/total
            print s, mx, mn, mean
            gt = self.false_negative + total
            print 'False negative:%f %%, (%d/%d)'%(self.false_negative/gt*100, self.false_negative, gt)
            print 'Missing: %f %% (%d/%d)'%(self.missing/total*100, self.missing, total)
            print 'Zero:', self.zero

            plt.figure()
            plt.plot(self.IOU)
            plt.plot([0, self.getIndex(mean, total)], [mean, mean], 'b--')
            for i in xrange(6, 10, 1):
                plt.plot([0, self.getIndex(i/10.0, total)], [i/10.0, i/10.0], 'm:')
            plt.savefig("Results/%s.png"%type)
            plt.close(0)

    def getIndex(self, v, total):
        total = int(total)
        for i in xrange(total):
            if self.IOU[i] > v:
                return i

    def cleanPath(self, part):
        if os.path.exists(part+'/gts/'):
            shutil.rmtree(part+'/gts/')
        if os.path.exists(part+'/dets/'):
            shutil.rmtree(part+'/dets/')

    def fixBB(self, x, y, w, h, size):
        width, height = size
        w = min(w+x, width)
        h = min(h+y, height)
        x = max(x, 0)
        y = max(y, 0)
        w -= x
        h -= y
        return x, y, w, h

    def readBBx(self):
        # get the gt
        self.gt_bbx = [[] for i in xrange(self.seqL + 1)]
        imgs = [None for i in xrange(self.seqL + 1)]
        for i in xrange(1, self.seqL + 1):
            imgs[i] = load_img(self.img_dir+'%06d.jpg'%i)
        gt = self.gt_dir + 'gt.txt'
        f = open(gt, 'r')
        for line in f.readlines():
            line = line.strip().split(',')
            if line[7] == '1':
                index = int(line[0])
                id = int(line[1])
                x, y = float(line[2]), float(line[3])
                w, h = float(line[4]), float(line[5])
                conf_score, l, vr = float(line[6]), int(line[7]), float(line[8])

                # sweep the invisible head-bbx from the training data
                if vr == 0:
                    continue

                img = imgs[i]
                x, y, w, h = self.fixBB(x, y, w, h, img.size)
                self.gt_bbx[index].append([x, y, w, h, conf_score])
        f.close()

        # get the det
        self.det_bbx = [[] for i in xrange(self.seqL + 1)]
        det = self.det_dir + 'det.txt'
        f = open(det, 'r')
        for line in f.readlines():
            line = line.strip().split(',')
            index = int(line[0])
            id = int(line[1])
            x, y = float(line[2]), float(line[3])
            w, h = float(line[4]), float(line[5])
            conf_score = float(line[6])
            if conf_score >= tau_conf_score:
                img = imgs[i]
                x, y, w, h = self.fixBB(x, y, w, h, img.size)
                self.det_bbx[index].append([x, y, w, h, conf_score])
        f.close()

    def C_IOU(self, Reframe, GTframe):
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

    def doit(self):
        min_conf_score = 1.0
        for i in xrange(1, self.seqL+1):
            gts = self.gt_bbx[i]
            dets = self.det_bbx[i]
            for gt in gts:
                iou = 0.0
                for det in dets:
                    cs = det[4]
                    tmp = self.C_IOU(det, gt)
                    if tmp > iou:
                        iou = tmp
                    if cs >= 0.0 and cs < 1e-7:
                        self.zero += 1
                    if tmp >= 0.5:
                        if cs < 0.0:
                            self.missing += 1
                        else:
                            min_conf_score = min(min_conf_score, cs)
                if iou >= 0.5:
                    self.IOU.append(iou)
                else:
                    self.false_negative += 1
        print '     tau_conf_score:', min_conf_score
        return

a = tau_iou()