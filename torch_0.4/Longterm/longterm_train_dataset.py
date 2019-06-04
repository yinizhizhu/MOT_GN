import torch.utils.data as data
import random, torch, shutil, os, gc
from math import *
from PIL import Image
import torch.nn.functional as F
from m_global_set import edge_initial, overlap, window_size


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, part, outName, cuda=True):
        super(DatasetFromFolder, self).__init__()
        self.dir = part
        self.cleanPath(part)
        self.img_dir = part + '/img1/'
        self.gt_dir = part + '/gt/'
        self.det_dir = part + '/det/'

        self.outName = outName
        out = open(self.outName, 'w')
        out.close()

        self.device = torch.device("cuda" if cuda else "cpu")

        self.getSeqL()
        self.readBBx()
        self.initBuffer()

    def cleanPath(self, part):
        if os.path.exists(part+'/gts/'):
            shutil.rmtree(part+'/gts/')
        if os.path.exists(part+'/dets/'):
            shutil.rmtree(part+'/dets/')

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
                self.width = float(line[1])
            elif line[0] == 'imHeight':
                self.height = float(line[1])
        f.close()
        # print 'The length of the sequence:', self.seqL

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
        bbxs = [[] for i in xrange(self.seqL + 1)]
        gt = self.gt_dir + 'gt.txt'
        f = open(gt, 'r')
        for line in f.readlines():
            line = line.strip().split(',')
            if line[7] == '1':
                frame = int(line[0])
                id = int(line[1])
                x, y = float(line[2]), float(line[3])
                w, h = float(line[4]), float(line[5])
                conf_score, l, vr = float(line[6]), int(line[7]), float(line[8])

                x, y, w, h = self.generator([x, y, w, h])
                x, y, w, h = self.fixBB(x, y, w, h)
                self.bbx[frame].append([x/self.width, y/self.height, w/self.width, h/self.height, frame, id])
                bbxs[frame].append([x, y, w, h, id, frame, id])
        f.close()

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
            self.m = len(self.objects)
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

    def getRet(self):
        ret = [[0.0 for i in xrange(self.n)] for j in xrange(self.m)]
        for i in xrange(self.m):
            objects, index, num, id = self.objects[i]
            bbx1 = objects[index]
            for j in xrange(self.n):
                ret[i][j] = self.distance(bbx1, self.bbx[self.f_step][j])
        return ret

    def getMotion(self, tag, index, vs_pos, pre_index=None):
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
                    return self.detections[self.nxt][index[0]][0][pre_index[0]][vs_pos[0]]
                ans = torch.cat((self.detections[self.nxt][index[0]][0][pre_index[0]][vs_pos[0]],
                                 self.detections[self.nxt][index[1]][0][pre_index[1]][vs_pos[1]]), dim=0)
                for i in xrange(2, n):
                    ans = torch.cat((ans, self.detections[self.nxt][index[i]][0][pre_index[i]][vs_pos[i]]), dim=0)
                return ans
            if n == 1:
                return self.detections[self.cur][index[0]][0][0]
            ans = torch.cat((self.detections[self.cur][index[0]][0][vs_pos[0]],
                             self.detections[self.cur][index[1]][0][vs_pos[1]]), dim=0)
            for i in xrange(2, n):
                ans = torch.cat((ans, self.detections[self.cur][index[i]][0][vs_pos[i]]), dim=0)
            return ans
        if tag == 0:
            self.updateVelocity(pre_index, index)
            return self.detections[self.nxt][index][0][pre_index][vs_pos]
        return self.detections[self.cur][index][0][vs_pos]


    def deleteMotion(self, index):
        del self.objects[index]     # delete the bbx: x, y, w, h, id, conf_score
        del self.detections[self.cur][index]   # delete the motion: [[x, y, w, h, v_x, v_y], id]

    def moveIndexNum(self, index, num):
        index = (index - 1) if index > 0 else (window_size - 1)
        num -= 1
        return index, num

    def updateVelocity(self, i, j, tag=True):
        index0, num0 = 0, 0
        v_x = [0.0 for k in xrange(window_size)]
        v_y = [0.0 for k in xrange(window_size)]

        # Compute the velocity
        if i != -1:
            x2, y2, w2, h2, frame2, id2 = self.bbx[self.f_step][j]
            objects, index, num, id = self.objects[i]
            index0, num0 = index, num
            while num:
                x1, y1, w1, h1, frame1, id2 = objects[index]
                t = float(frame2-frame1)
                v_x[index] = (x2+w2/2 - (x1+w1/2))/t
                v_y[index] = (y2+h2/2 - (y1+h1/2))/t
                index, num = self.moveIndexNum(index, num)

        if tag:
            # For updating the velocity
            while num0:
                # print self.detections[self.nxt][j][0][i][index0]
                self.detections[self.nxt][j][0][i][index0][0][4] = v_x[index0]
                self.detections[self.nxt][j][0][i][index0][0][5] = v_y[index0]
                index0, num0 = self.moveIndexNum(index0, num0)
        else:
            # For association
            id = self.detections[self.nxt][j][2]
            motion0 = self.detections[self.nxt][j][0][i][0]

            if i == -1:
                cur_m = [motion0.clone() for k in xrange(window_size)]
                while num0:
                    motion = motion0.clone()
                    motion[0][4] = v_x[index0]
                    motion[0][5] = v_y[index0]
                    cur_m[index0] = motion
                    index0, num0 = self.moveIndexNum(index0, num0)
                self.detections[self.cur].append([cur_m, self.f_step, 0, 1, id])
                self.objects.append([[[p for p in self.bbx[self.f_step][j]] for k in xrange(window_size)], 0, 1, id])
            else:
                index, num, id = self.detections[self.cur][i][2:]
                index = (index+1)%window_size
                self.detections[self.cur][i][0][index] = motion0.clone()
                self.detections[self.cur][i][1] = self.f_step
                self.detections[self.cur][i][2] = index
                if num < window_size:
                    num += 1

                self.detections[self.cur][i][3] = num
                self.objects[i][0][index] = [x2, y2, w2, h2, frame2, id2]
                self.objects[i][1] = index
                self.objects[i][2] = num

    def getVelocity(self):
        remaining = set(j for j in xrange(self.n))
        # For the connection between two detections
        for (i, j) in self.matches:
            remaining.remove(j)
            self.updateVelocity(i, j, False)

        # For the birth of objects
        for j in remaining:
            self.updateVelocity(-1, j, False)

    def getMN(self, m, n):
        ans = [[[None for i in xrange(n)] for j in xrange(m)] for k in xrange(window_size)]
        for i in xrange(m):
            objects, index, num, id = self.objects[i]
            while num:
                Reframe = objects[index]
                for j in xrange(n):
                    GTframe = self.bbx[self.f_step][j]
                    p = self.IOU(Reframe, GTframe)
                    # 1 - match, 0 - mismatch
                    ans[index][i][j] = torch.FloatTensor([(1 - p)/100.0, p/100.0])
                index, num = self.moveIndexNum(index, num)
        return ans

    def feature(self, tag=0):
        '''
        Getting the appearance of the detections in current frame
        :param tag: 1 - initiating
        :param show: 1 - show the cropped & src image
        :return: None
        '''
        if tag:
            self.objects = []                # The buffer storing objects' bounding boxes

        motions = []
        with torch.no_grad():
            m = window_size if tag else self.m  # Each object contains 5 detections at most
            for bbx in self.bbx[self.f_step]:
                """
                Bellow Conditions need to be taken into consideration:
                    x, y < 0 and x+w > W, y+h > H
                """
                x, y, w, h, frame, id = bbx

                cur_m = []
                for i in xrange(m):
                    if tag:
                        cur_m.append(torch.FloatTensor([[x, y, w, h, 0.0, 0.0]]).to(self.device))
                    else:
                        cur_m.append([torch.FloatTensor([[x, y, w, h, 0.0, 0.0]]).to(self.device) for j in xrange(window_size)])

                if tag:
                    self.objects.append([[[p for p in bbx] for i in xrange(window_size)], 0, 1, id])
                    motions.append([cur_m, frame, 0, 1, id])  # [Detections, frame, index, sum, id]
                else:
                    motions.append([cur_m, frame, id])

        if tag:
            self.detections[self.cur] = motions
        else:
            self.detections[self.nxt] = motions

    def loadNext(self):
        self.m = len(self.objects)

        self.gap = 0
        self.n = 0
        while self.n == 0:
            self.f_step += 1
            self.feature()
            self.n = len(self.detections[self.nxt])
            self.gap += 1

        if self.gap > 1:
            print '           Empty in loadNext:', self.f_step-self.gap+1, '-', self.gap-1

        vs = []
        self.matches = []
        self.step_gt = 0.0
        self.gts = [[None for j in xrange(self.n)] for i in xrange(self.m)]
        m = len(self.detections[self.cur])
        n = len(self.detections[self.nxt])
        for i in xrange(m):
            con, frame, index, num, id = self.detections[self.cur][i]

            for j in xrange(n):
                detections, frame, id2 = self.detections[self.nxt][j]
                tag = int(id == id2)
                if tag:
                    self.matches.append((i, j))
                self.gts[i][j] = torch.LongTensor([tag])
                self.step_gt += tag*1.0

            while num:
                # print len(con[index]), con[index]
                vs.append(con[index])
                for j in xrange(n):
                    detections, frame, id = self.detections[self.nxt][j]
                    # print len(detections[i][index]), detections[i][index]
                    vs.append(detections[i][index])
                index, num = self.moveIndexNum(index, num)

        self.candidates = []
        self.edges = self.getMN(self.m, self.n)

        es = []
        for i in xrange(self.m):
            for j in xrange(self.n):
                for k in xrange(window_size):
                    e = self.edges[k][i][j]
                    if e is not None:
                        es.append(e)
                        self.candidates.append([e, self.gts[i][j].clone(),
                                                i, k, j])

        self.E = self.aggregate(es).to(self.device).view(1,-1)
        self.V = self.aggregate(vs).to(self.device)

        # print '     The index of the next frame', self.f_step, len(self.bbx)
        return self.gap

    def showE(self):
        with torch.no_grad():
            out = open(self.outName, 'a')
            print >> out, ''
            print >> out, '-'*45, '-'*45
            print >> out, '     edge'

            print >> out, ' ', self.f_step-1
            print >> out, self.outE(self.edges[0], 1)

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
                    if edges[i][j] is not None:
                        tmp = F.softmax(edges[i][j]).cpu().numpy()
                        ret[i][j] = float(tmp[1])

                        tmp = edges[i][j].cpu().numpy()
                        con1[i][j] = float(tmp[0])
                        con2[i][j] = float(tmp[1])
                    else:
                        ret[i][j] = -9
                        con1[i][j] = -9
                        con2[i][j] = -9
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
