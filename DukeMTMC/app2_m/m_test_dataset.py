import torch.utils.data as data
import random, torch, h5py
from math import *
from PIL import Image
import torch.nn.functional as F
from global_set import edge_initial, test_gt_det, tau_dis, tau_threshold, SEQLEN#, tau_conf_score
import numpy as np


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class MDatasetFromFolder(data.Dataset):
    def __init__(self, camera, tau, cuda=True):
        super(MDatasetFromFolder, self).__init__()
        self.seq_index = camera

        self.width, self.height = 1920.0, 1080.0

        self.device = torch.device("cuda" if cuda else "cpu")
        self.tau_conf_score = tau

        self.readBBx_det()
        self.initBuffer()

    def fixBB(self, x, y, w, h):
        w = min(w+x, self.width)
        h = min(h+y, self.height)
        x = max(x, 0)
        y = max(y, 0)
        w -= x
        h -= y
        return x, y, w, h

    def fixBBx(self, x1, y1, x2, y2):
        width = x2 - x1
        height = y2 - y1
        x1 -= width/12
        x2 += width/12
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

    def readBBx_det(self):
        # get the det
        self.bbx = [[] for i in xrange(SEQLEN + 1)]

        heads = [0, 5543, 3607, 27244, 31182, 1, 22402, 18968, 46766]
        test = [227541, 356648]

        detection_dir = '/media/lee/DATA/DukeMTMC/detections/openpose/camera%d.mat'%self.seq_index
        det = h5py.File(detection_dir)
        det = det['detections']
        det = np.transpose(det)

        head = test[0] - heads[self.seq_index] + 1
        tail = head + SEQLEN - 1

        out = open('Detections/%d.txt'%self.seq_index, 'w')
        for bbx in det:
            frame = int(bbx[1])
            bbx = self.getBBx(bbx)
            if bbx is not None and frame >= head and frame <= tail:
                x, y, w, h = bbx
                self.bbx[frame-head+1].append([x/self.width, y/self.height, w/self.width, h/self.height, frame])
                print >> out, '%d,%d,%d,%f,%f,%f,%f'%(self.seq_index, -1, frame, x, y, w, h)
        out.close()

    def initBuffer(self):
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

    def IOU(self, Reframe, GTframe):
        """
        Compute the Intersection of Union
        :param Reframe: x, y, w, h
        :param GTframe: x, y, w, h
        :return: Ratio
        """
        if edge_initial == 1:
            return random.random()
        elif edge_initial == 3:
            return 0.5
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

    def getMotion(self, tag, index, pre_index=None):
        cur = self.cur if tag else self.nxt
        if tag == 0:
            self.updateVelocity(pre_index, index)
            return self.detections[cur][index][0][pre_index]
        return self.detections[cur][index][0][0]

    def moveMotion(self, index):
        self.bbx[self.f_step].append(self.bbx[self.f_step-self.gap][index])  # add the bbx: x, y, w, h, id, conf_score
        self.detections[self.nxt].append(self.detections[self.cur][index])   # add the motion: [[x, y, w, h, v_x, v_y], id]

    def cleanEdge(self):
        con = []
        index = 0
        for det in self.detections[self.nxt]:
            motion, id = det
            x = motion[0][0].item() + motion[0][4].item()
            y = motion[0][1].item() + motion[0][5].item()
            if (x < 0.0 or x > 1.0) or (y < 0.0 or y > 1.0):
                con.append(index)
            index += 1

        for i in xrange(len(con)-1, -1, -1):
            index = con[i]
            del self.bbx[self.f_step][index]
            del self.detections[self.nxt][index]
        return con

    def swapFC(self):
        self.cur = self.cur ^ self.nxt
        self.nxt = self.cur ^ self.nxt
        self.cur = self.cur ^ self.nxt

    def updateVelocity(self, i, j, tag=True):
        v_x = 0.0
        v_y = 0.0
        if i != -1:
            x1, y1, w1, h1, frame1 = self.bbx[self.f_step-self.gap][i]
            x2, y2, w2, h2, frame2 = self.bbx[self.f_step][j]
            t = frame2 - frame1
            v_x = (x2+w2/2 - (x1+w1/2))/t
            v_y = (y2+h2/2 - (y1+h1/2))/t
        if tag:
            # print 'm=%d,n=%d; i=%d, j=%d'%(len(self.detections[self.cur]), len(self.detections[self.nxt]), i, j)
            self.detections[self.nxt][j][0][i][0][4] = v_x
            self.detections[self.nxt][j][0][i][0][5] = v_y
        else:
            cur_m = self.detections[self.nxt][j][0][0]
            cur_m[0][4] = v_x
            cur_m[0][5] = v_y
            self.detections[self.nxt][j][0] = [cur_m]

    def getMN(self, m, n):
        cur = self.f_step - self.gap
        ans = [[None for i in xrange(n)] for i in xrange(m)]
        for i in xrange(m):
            Reframe = self.bbx[cur][i]
            for j in xrange(n):
                GTframe = self.bbx[self.f_step][j]
                p = self.IOU(Reframe, GTframe)
                # 1 - match, 0 - mismatch
                ans[i][j] = torch.FloatTensor([1 - p, p])
        return ans

    def feature(self, tag=0):
        '''
        Getting the appearance of the detections in current frame
        :param tag: 1 - initiating
        :param show: 1 - show the cropped & src image
        :return: None
        '''
        motions = []
        with torch.no_grad():
            m = 1 if tag else self.m
            for bbx in self.bbx[self.f_step]:
                """
                Bellow Conditions needed be taken into consideration:
                    x, y < 0 and x+w > W, y+h > H
                """
                x, y, w, h, frame = bbx
                cur_m = []
                for i in xrange(m):
                    cur_m.append(torch.FloatTensor([[x, y, w, h, 0.0, 0.0]]).to(self.device))
                motions.append([cur_m, frame])
        if tag:
            self.detections[self.cur] = motions
        else:
            self.detections[self.nxt] = motions

    def loadNext(self):
        self.m = len(self.detections[self.cur])

        self.gap = 0
        self.n = 0
        while self.n == 0:
            self.f_step += 1
            self.gap += 1

            if self.f_step > SEQLEN:
                print '           Empty in loadNext:', self.f_step - self.gap + 1, '-', self.gap - 1
                return self.gap

            self.feature()
            self.n = len(self.detections[self.nxt])

        if self.gap > 1:
            print '           Empty in loadNext:', self.f_step-self.gap+1, '-', self.gap-1

        self.candidates = []
        self.edges = self.getMN(self.m, self.n)

        es = []
        # vs_index = 0
        for i in xrange(self.m):
            # vr_index = self.m
            for j in xrange(self.n):
                e = self.edges[i][j]
                es.append(e)
                self.candidates.append([e, i, j])
            #     vr_index += 1
            # vs_index += 1

        vs = []
        for i in xrange(2):
            n = len(self.detections[i])
            for j in xrange(n):
                v = self.detections[i][j][0][0]
                vs.append(v)

        self.E = self.aggregate(es).to(self.device).view(1,-1)
        self.V = self.aggregate(vs).to(self.device)

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