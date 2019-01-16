import torch.utils.data as data
import cv2, random, torch, shutil, os
import numpy as np
from math import *
from PIL import Image
import torch.nn.functional as F
from mot_model import appearance
from global_set import edge_initial, test_gt_det, tau_dis, app_fine_tune, fine_tune_dir, tau_threshold#, tau_conf_score
from torchvision.transforms import ToTensor


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class ADatasetFromFolder(data.Dataset):
    def __init__(self, part, part_I, tau, cuda=True, show=0):
        super(ADatasetFromFolder, self).__init__()
        self.dir = part
        self.cleanPath(part)
        self.img_dir = part_I + '/img1/'
        self.gt_dir = part + '/gt/'
        self.det_dir = part + '/det/'
        self.device = torch.device("cuda" if cuda else "cpu")
        self.tau_conf_score = tau
        self.show = show

        self.loadAModel()
        self.getSeqL()
        self.readBBx_det()
        self.initBuffer()

    def cleanPath(self, part):
        if os.path.exists(part+'/gts/'):
            shutil.rmtree(part+'/gts/')
        if os.path.exists(part+'/dets/'):
            shutil.rmtree(part+'/dets/')

    def loadAModel(self):
        if app_fine_tune:
            self.Appearance = torch.load(fine_tune_dir)
        else:
            self.Appearance = appearance()
        self.Appearance.to(self.device)
        self.Appearance.eval()  # fixing the BatchN layer

    def getSeqL(self):
        # get the length of the sequence
        info = self.dir+'/seqinfo.ini'
        f = open(info, 'r')
        f.readline()
        for line in f.readlines():
            line = line.strip().split('=')
            if line[0] == 'seqLength':
                self.seqL = int(line[1])
            elif line[0] == 'imWidth':
                self.width = int(line[1])
            elif line[0] == 'imHeight':
                self.height = int(line[1])
        f.close()
        # print 'The length of the sequence:', self.seqL

    def readBBx_det(self):
        # get the gt
        self.bbx = [[] for i in xrange(self.seqL + 1)]
        det = self.det_dir + 'det.txt'
        f = open(det, 'r')
        for line in f.readlines():
            line = line.strip().split(',')
            frame = int(line[0])
            x, y = int(float(line[2])), int(float(line[3]))
            w, h = int(float(line[4])), int(float(line[5]))
            conf_score = float(line[6])
            if conf_score >= self.tau_conf_score:
                self.bbx[frame].append([x, y, w, h, frame])
        f.close()

    def initBuffer(self):
        if self.show:
            cv2.namedWindow('view', flags=0)
            cv2.namedWindow('crop', flags=0)
        self.f_step = 1  # the index of next frame in the process
        self.cur = 0     # the index of current frame in the detections
        self.nxt = 1     # the index of next frame in the detections
        self.detections = [None, None]   # the buffer to storing images: current & next frame
        self.feature(1)

    def setBuffer(self, f):
        self.m = 0
        counter = -1
        while self.m == 0:
            counter += 1
            self.f_step = f + counter
            self.feature(1)
            self.m = len(self.detections[self.cur])
        if counter > 0:
            print '           Empty in setBuffer:', counter
        return counter

    def fixBB(self, x, y, w, h):
        w = min(w+x, self.width)
        h = min(h+y, self.height)
        x = max(x, 0)
        y = max(y, 0)
        w -= x
        h -= y
        return x, y, w, h

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

    def getMN(self, m, n):
        ans = [[None for i in xrange(n)] for i in xrange(m)]
        for i in xrange(m):
            Reframe = self.bbx[self.f_step-self.gap][i]
            for j in xrange(n):
                GTframe = self.bbx[self.f_step][j]
                p = random.random()
                # 1 - match, 0 - mismatch
                ans[i][j] = torch.FloatTensor([1 - p, p]).to(self.device)
        return ans

    def aggregate(self, set):
        if len(set):
            rho = sum(set)
            return rho/len(set)
        print '     The set is empty!'
        return None

    def distance(self, a_bbx, b_bbx):
        w1 = float(a_bbx[2]) * tau_dis
        w2 = float(b_bbx[2]) * tau_dis
        dx = float(a_bbx[0] + a_bbx[2]/2) - float(b_bbx[0] + b_bbx[2]/2)
        dy = float(a_bbx[1] + a_bbx[3]/2) - float(b_bbx[1] + b_bbx[3]/2)
        d = sqrt(dx*dx+dy*dy)
        if d <= w1 and d <= w2:
            return 0.0
        return tau_threshold

    def getRet(self):
        cur = self.f_step-self.gap
        ret = [[0.0 for i in xrange(self.n)] for j in xrange(self.m)]
        for i in xrange(self.m):
            bbx1 = self.bbx[cur][i]
            for j in xrange(self.n):
                ret[i][j] = self.distance(bbx1, self.bbx[self.f_step][j])
        return ret

    def getApp(self, tag, index):
        cur = self.cur if tag else self.nxt
        if torch.is_tensor(index):
            n = index.numel()
            if n < 0:
                print 'The tensor is empyt!'
                return None
            if n == 1:
                return self.detections[cur][index[0]][0]
            ans = torch.cat((self.detections[cur][index[0]][0], self.detections[cur][index[1]][0]), dim=0)
            for i in xrange(2, n):
                ans = torch.cat((ans, self.detections[cur][index[i]][0]), dim=0)
            return ans
        return self.detections[cur][index][0]

    def moveApp(self, index):
        self.bbx[self.f_step].append(self.bbx[self.f_step-self.gap][index])  # add the bbx
        self.detections[self.nxt].append(self.detections[self.cur][index])   # add the appearance

    def delApp(self, index):
        if index >= 0:
            del self.bbx[self.f_step+1][index]
            del self.detections[self.nxt][index]

    def findDetction(self, bbx):
        step = 0
        index, iou = -1, -1.0
        for BBx in self.bbx[self.f_step+1]:
            tmp = self.IOU(bbx, BBx)
            if tmp > iou:
                index = step
                iou = tmp
            step += 1
        if iou < 0.5:
            return -1
        return index

    def addApp(self, bbx):
        self.bbx[self.f_step+1].append(bbx)

        img = load_img(self.img_dir + '%06d.jpg' % (self.f_step + 1))  # initial with loading the first frame
        x, y, w, h, frame = bbx
        crop = img.crop([x, y, x + w, y + h])
        bbx = crop.resize((224, 224), Image.ANTIALIAS)
        ret = self.resnet34(bbx)
        app = ret.data

        if len(self.detections[self.nxt]):
            self.detections[self.nxt].append([app, frame])
        else:
            self.detections[self.nxt] = [[app, frame]]

    def swapFC(self):
        self.cur = self.cur ^ self.nxt
        self.nxt = self.cur ^ self.nxt
        self.cur = self.cur ^ self.nxt

    def resnet34(self, img):
        bbx = ToTensor()(img)
        bbx = bbx.to(self.device)
        bbx = bbx.view(-1, bbx.size(0), bbx.size(1), bbx.size(2))
        ret = self.Appearance(bbx)
        ret = ret.view(1, -1)
        return ret

    def feature(self, tag=0):
        '''
        Getting the appearance of the detections in current frame
        :param tag: 1 - initiating
        :param show: 1 - show the cropped & src image
        :return: None
        '''
        apps = []
        with torch.no_grad():
            bbx_container = []
            img = load_img(self.img_dir + '%06d.jpg' % self.f_step)  # initial with loading the first frame
            for bbx in self.bbx[self.f_step]:
                """
                Bellow Conditions needed be taken into consideration:
                    x, y < 0 and x+w > W, y+h > H
                """
                x, y, w, h, frame = bbx
                x, y, w, h = self.fixBB(x, y, w, h)
                bbx_container.append([x, y, w, h, frame])
                crop = img.crop([x, y, x + w, y + h])
                bbx = crop.resize((224, 224), Image.ANTIALIAS)
                ret = self.resnet34(bbx)
                app = ret.data
                apps.append([app, frame])

                if self.show:
                    img = np.asarray(img)
                    crop = np.asarray(crop)
                    print '%06d'%self.f_step, frame, '***',
                    print w, h, '-',
                    print len(crop[0]), len(crop)
                    cv2.imshow('crop', crop)
                    cv2.imshow('view', img)
                    cv2.waitKey(34)
                    raw_input('Continue?')
                    # cv2.waitKey(34)
            self.bbx[self.f_step] = bbx_container
        if tag:
            self.detections[self.cur] = apps
        else:
            self.detections[self.nxt] = apps

    def loadNext(self):
        self.m = len(self.detections[self.cur])

        self.gap = 0
        self.n = 0
        while self.n == 0:
            self.f_step += 1
            self.gap += 1

            if self.f_step > self.seqL:
                print '           Empty in loadNext:', self.f_step-self.gap+1, '-', self.gap-1
                return self.gap

            self.feature()
            self.n = len(self.detections[self.nxt])

        if self.gap > 1:
            print '           Empty in loadNext:', self.f_step-self.gap+1, '-', self.gap-1

        self.candidates = []
        self.edges = self.getMN(self.m, self.n)

        for i in xrange(self.m):
            for j in xrange(self.n):
                e = self.edges[i][j]
                self.candidates.append([e, i, j])

        # print '     The index of the next frame', self.f_step, len(self.bbx)
        return self.gap

    def showE(self, outName):
        with torch.no_grad():
            out = open(outName, 'a')
            print >> out, ''
            print >> out, '-'*45, '-'*45
            print >> out, '     edge'

            print >> out, ' ', self.f_step-1
            print >> out, self.outE(self.edges, 1)

            out.close()

    def outE(self, edges, tag):
        m = len(edges)
        n = len(edges[0])
        if tag:
            ret = [[None for i in xrange(n)] for j in xrange(m)]
            con1 = [[None for i in xrange(n)] for j in xrange(m)]
            con2 = [[None for i in xrange(n)] for j in xrange(m)]
        else:
            ans = [[None for i in xrange(n)] for j in xrange(m)]
        for i in xrange(m):
            for j in xrange(n):
                if tag:
                    tmp = F.softmax(edges[i][j]).cpu().numpy()
                    ret[i][j] = float(tmp[1])

                    tmp = edges[i][j].cpu().numpy()
                    con1[i][j] = float(tmp[0])
                    con2[i][j] = float(tmp[1])
                else:
                    tmp = edges[i][j].cpu().numpy()
                    ans[i][j] = float(tmp)
        if tag:
            ret = torch.FloatTensor(ret)
            con1 = torch.FloatTensor(con1)
            con2 = torch.FloatTensor(con2)
            return 'Probability', ret, 'Output', con1, con2
        ans = torch.FloatTensor(ans)
        return ans

    def __getitem__(self, index):
        return self.candidates[index]

    def __len__(self):
        return len(self.candidates)