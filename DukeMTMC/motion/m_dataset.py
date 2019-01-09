import torch.utils.data as data
import random, torch, shutil, os
from scipy.io import loadmat
from PIL import Image
import torch.nn.functional as F
from m_global_set import edge_initial, overlap, SEQLEN


# test_hard = [227541, 263503]
# test_hard_len = test_hard[1] - test_hard[0] + 2
# container_hard = [[] for i in xrange(test_hard_len)]
#
# test_easy = [263504, 356648]
# test_easy_len = test_easy[1] - test_easy[0] + 2
# container_easy = [[] for i in xrange(test_easy_len)]


class DatasetFromFolder(data.Dataset):
    def __init__(self, camera, outName, cuda=True):
        super(DatasetFromFolder, self).__init__()
        self.seq_index = camera

        self.width, self.height = 1920.0, 1080.0

        self.outName = outName
        out = open(self.outName, 'w')
        out.close()

        self.device = torch.device("cuda" if cuda else "cpu")

        self.seqL = SEQLEN
        self.readBBx()
        self.initBuffer()
        print '     Data loader is already!'

    def fixBB(self, x, y, w, h):
        w = min(w+x, self.width)
        h = min(h+y, self.height)
        x = max(x, 0)
        y = max(y, 0)
        w -= x
        h -= y
        return x, y, w, h

    def generator(self, bbx):
        if random.randint(0, 1):
            x, y, w, h = bbx
            tmp = overlap*2/(1+overlap)
            n_w = random.uniform(tmp*w, w)
            n_h = tmp*w*h/n_w

            direction = random.randint(1, 4)
            if direction == 1:
                x = x + n_w - w
                y = y + n_h - h
            elif direction == 2:
                x = x - n_w + w
                y = y + n_h - h
            elif direction == 3:
                x = x + n_w - w
                y = y - n_h + h
            else:
                x = x - n_w + w
                y = y - n_h + h
            ans = [x, y, w, h]
            return ans
        return bbx

    def readBBx(self):
        # get the gt
        self.bbx = [[] for i in xrange(self.seqL + 1)]

        heads = [0, 5543, 3607, 27244, 31182, 1, 22402, 18968, 46766]
        trainval = [49700, 227540]

        trainval_dir = '/media/lee/DATA/DukeMTMC/ground_truth/trainval.mat'
        val = loadmat(trainval_dir)
        val = val['trainData']

        head = trainval[0] - heads[self.seq_index] + 1
        tail = head + SEQLEN - 1

        for bbx in val:
            camera = int(bbx[0])
            id = int(bbx[1])
            frame = int(bbx[2])
            x, y = float(bbx[3]), float(bbx[4])
            w, h = float(bbx[5]), float(bbx[6])
            if camera == self.seq_index and (frame >= head and frame <= tail):
                self.bbx[frame-head+1].append([x/self.width, y/self.height, w/self.width, h/self.height, id, frame])

    def initBuffer(self):
        self.f_step = 1  # the index of next frame in the process
        self.cur = 0     # the index of current frame in the detections
        self.nxt = 1     # the index of next frame in the detections
        self.detections = [None, None]   # the buffer to storing images: current & next frame
        self.feature(1)

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
        _width = width1 + width2 - (endx - startx)

        endy = max(y1 + height1, y2 + height2)
        starty = min(y1, y2)
        _height = height1 + height2 - (endy - starty)

        if _width <= 0 or _height <= 0:
            ratio = 0.0
        else:
            Area = _width * _height
            Area1 = width1 * height1
            Area2 = width2 * height2
            ratio = Area * 1. / (Area1 + Area2 - Area)
        return ratio

    def swapFC(self):
        self.getVelocity()
        self.cur = self.cur ^ self.nxt
        self.nxt = self.cur ^ self.nxt
        self.cur = self.cur ^ self.nxt

    def getMotion(self, tag, index, pre_index=None):
        cur = self.cur if tag else self.nxt
        if torch.is_tensor(index):
            n = index.numel()
            if n < 0:
                print 'The tensor is empyt!'
                return None
            if tag == 0:
                for k in xrange(n):
                    i, j = pre_index[k].item(), index[k].item()
                    self.updateVelocity(i, j)
                if n == 1:
                    return self.detections[cur][index[0]][0][pre_index[0]]
                ans = torch.cat((self.detections[cur][index[0]][0][pre_index[0]],
                                 self.detections[cur][index[1]][0][pre_index[1]]), dim=0)
                for i in xrange(2, n):
                    ans = torch.cat((ans, self.detections[cur][index[i]][0][pre_index[i]]), dim=0)
                return ans
            if n == 1:
                return self.detections[cur][index[0]][0][0]
            ans = torch.cat((self.detections[cur][index[0]][0][0],
                             self.detections[cur][index[1]][0][0]), dim=0)
            for i in xrange(2, n):
                ans = torch.cat((ans, self.detections[cur][index[i]][0][0]), dim=0)
            return ans
        if tag == 0:
            self.updateVelocity(pre_index, index)
            return self.detections[cur][index][0][pre_index]
        return self.detections[cur][index][0][0]

    def updateVelocity(self, i, j, tag=True):
        '''
        :param i: cur_index, -1 - birth
        :param j: nxt_index
        :param tag: True - update the velocity in the next frame, False - Write down the final velocity
        :return:
        '''
        v_x, v_y = 0.0, 0.0
        if i >= 0:
            x1, y1, w1, h1, id1, frame1 = self.bbx[self.f_step-self.gap][i]
            x2, y2, w2, h2, id2, frame2 = self.bbx[self.f_step][j]
            t = frame2 - frame1
            v_x = (x2+w2/2 - (x1+w1/2))/t
            v_y = (y2+h2/2 - (y1+h1/2))/t
        if tag:
            self.detections[self.nxt][j][0][i][0][4] = v_x
            self.detections[self.nxt][j][0][i][0][5] = v_y
        else:
            cur_m = self.detections[self.nxt][j][0][0]
            cur_m[0][4] = v_x
            cur_m[0][5] = v_y
            self.detections[self.nxt][j][0] = [cur_m]

    def getVelocity(self):
        remaining = set(j for j in xrange(self.n))
        # For the connection between two detections
        for (i, j) in self.matches:
            remaining.remove(j)
            self.updateVelocity(i, j, False)

        # For the birth of objects
        for j in remaining:
            self.updateVelocity(-1, j, False)

    def getMN(self):
        cur = self.f_step - self.gap
        ans = [[None for j in xrange(self.n)] for i in xrange(self.m)]
        for i in xrange(self.m):
            Reframe = self.bbx[cur][i]
            for j in xrange(self.n):
                GTframe = self.bbx[self.f_step][j]
                p = self.IOU(Reframe, GTframe)
                # 1 - match, 0 - mismatch
                ans[i][j] = torch.FloatTensor([1 - p, p])
        return ans

    def aggregate(self, sets):
        n = len(sets)
        if n:
            rho = sum(sets)
            return rho/n
        print '     The set is empty!'
        return None

    def initEC(self):
        self.m = len(self.detections[self.cur])
        self.n = len(self.detections[self.nxt])
        self.edges = self.getMN()
        self.candidates = []
        self.matches = []
        self.gts = [[None for j in xrange(self.n)] for i in xrange(self.m)]
        self.step_gt = 0.0
        for i in xrange(self.m):
            for j in xrange(self.n):
                tag = int(self.detections[self.cur][i][1] == self.detections[self.nxt][j][1])
                if tag:
                    self.matches.append((i, j))
                self.gts[i][j] = torch.LongTensor([tag])
                self.step_gt += tag*1.0

        es = []
        # vs_index = 0
        for i in xrange(self.m):
            # vr_index = self.m
            for j in xrange(self.n):
                e = self.edges[i][j]
                gt = self.gts[i][j]
                es.append(e)
                self.candidates.append([e, gt, i, j])
            #     vr_index += 1
            # vs_index += 1

        vs = []
        for i in xrange(2):
            n = len(self.detections[i])
            for j in xrange(n):
                v = self.detections[i][j][0][0]
                vs.append(v)

        self.E = self.aggregate(es).to(self.device).view(1,-1)
        self.V = self.aggregate(vs)

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

    def feature(self, tag=0):
        '''
        Getting the motion of the detections in current frame
        :param tag: 1 - initiating
        :return: None
        '''
        motions = []
        with torch.no_grad():
            m = 1 if tag else self.m
            for bbx in self.bbx[self.f_step]:
                """
                Condition needed be taken into consideration:
                    x, y < 0 and x+w > W, y+h > H
                """
                x, y, w, h, id, frame = bbx
                x += w/2
                y += h/2
                cur_m = []
                for i in xrange(m):
                    cur_m.append(torch.FloatTensor([[x, y, w, h, 0.0, 0.0]]).to(self.device))
                motions.append([cur_m, id, frame])
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

            if self.f_step > self.seqL:
                print '           Empty in loadNext:', self.f_step-self.gap+1, '-', self.gap-1
                return self.gap

            self.feature()
            self.n = len(self.detections[self.nxt])

        if self.gap > 1:
            print '           Empty in loadNext:', self.f_step-self.gap+1, '-', self.gap-1

        self.initEC()

        return self.gap

    def showE(self):
        with torch.no_grad():
            out = open(self.outName, 'a')
            print >> out, ''
            print >> out, '-'*45, '%d * %d'%(self.m, self.n), '-'*45
            print >> out, '     edge'

            print >> out, ' ', self.f_step-1
            print >> out, self.outE(self.edges, 1)

            print >> out, '     connection'

            print >> out, ' ', self.f_step-1
            print >> out, self.outE(self.gts, 0)

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
                    # ret[i][j] = float(tmp[0])
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