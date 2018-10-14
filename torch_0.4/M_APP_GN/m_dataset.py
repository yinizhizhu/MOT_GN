import torch.utils.data as data
import random, torch, shutil, os
from PIL import Image
import torch.nn.functional as F
from m_global_set import edge_initial, overlap
from m_mot_model import appearance
from torchvision.transforms import ToTensor


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
        self.device = torch.device("cuda" if cuda else "cpu")

        self.loadAModel()
        self.getSeqL()
        self.readBBx()
        self.initBuffer()

    def cleanPath(self, part):
        if os.path.exists(part+'/gts/'):
            shutil.rmtree(part+'/gts/')
        if os.path.exists(part+'/dets/'):
            shutil.rmtree(part+'/dets/')

    def loadAModel(self):
        self.Appearance = torch.load('../MOT/Fine-tune_GPU_5_3_60_aug/appearance_19.pth')
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
        f.close()
        print '     The length of the sequence:', self.seqL

    def generator(self, bbx):
        x, y, w, h, id, conf_score, vr, width, height = bbx
        tmp = overlap*2/(1+overlap)
        n_w = random.uniform(tmp*w, w)
        n_h = tmp*w*float(h)/n_w

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
        ans = [x, y, w, h, id, conf_score, vr, width, height]
        return ans

    def readBBx(self):
        # get the gt
        self.bbx = [[] for i in xrange(self.seqL + 1)]
        imgs = [None for i in xrange(self.seqL + 1)]
        for i in xrange(1, self.seqL+1):
            img = load_img(self.img_dir + '%06d.jpg' % i)  # initial with loading the first frame
            imgs[i] = img
        gt = self.gt_dir + 'gt.txt'
        f = open(gt, 'r')
        pre = -1
        for line in f.readlines():
            line = line.strip().split(',')
            if line[7] == '1':
                index = int(line[0])
                id = int(line[1])
                x, y = float(line[2]), float(line[3])
                w, h = float(line[4]), float(line[5])
                conf_score, l, vr = float(line[6]), int(line[7]), float(line[8])

                # sweep the invisible head-bbx from the training data
                if pre != id and vr == 0:
                    continue

                pre = id
                img = imgs[index]
                width, height = float(img.size[0]), float(img.size[1])
                tmp_bbx = self.generator([x, y, w, h, id, conf_score, vr, width, height])
                self.bbx[index].append(tmp_bbx)
        f.close()

    def initBuffer(self):
        self.f_step = 1  # the index of next frame in the process
        self.cur = 0     # the index of current frame in the detections
        self.nxt = 1     # the index of next frame in the detections
        self.detections = [None, None]   # the buffer to storing images: current & next frame
        self.feature(1)

    def setBuffer(self, f):
        self.f_step = f
        self.feature(1)

    def fixBB(self, x, y, w, h, size):
        width, height = size
        w = min(w+x, width)
        h = min(h+y, height)
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

    def getMN(self, m, n):
        ans = [[None for i in xrange(n)] for i in xrange(m)]
        for i in xrange(m):
            Reframe = self.bbx[self.f_step-1][i]
            for j in xrange(n):
                GTframe = self.bbx[self.f_step][j]
                p = self.IOU(Reframe, GTframe)
                # 1 - match, 0 - mismatch
                ans[i][j] = torch.FloatTensor([1 - p, p])
        return ans

    def aggregate(self, set):
        if len(set):
            rho = sum(set)
            return rho/len(set)
        print '     The set is empty!'
        return None

    def getApp(self, tag, index, pre_index=None):
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
                return self.detections[cur][index[0]][0]
            ans = torch.cat((self.detections[cur][index[0]][0], self.detections[cur][index[1]][0]), dim=0)
            for i in xrange(2, n):
                ans = torch.cat((ans, self.detections[cur][index[i]][0]), dim=0)
            return ans
        return self.detections[cur][index][0]

    def swapFC(self):
        self.getVelocity()
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
        :return: None
        '''
        apps = []
        with torch.no_grad():
            bbx_container = []
            img = load_img(self.img_dir + '%06d.jpg' % self.f_step)  # initial with loading the first frame
            for bbx in self.bbx[self.f_step]:
                """
                Condition needed be taken into consideration:
                    x, y < 0 and x+w > W, y+h > H
                """
                x, y, w, h, id, conf_score, vr, width, height = bbx
                x, y, w, h = self.fixBB(x, y, w, h, img.size)
                bbx_container.append([x, y, w, h, id, conf_score, vr, width, height])
                motion = torch.FloatTensor([[(x+w/2)/width, (y+h/2)/height, w/width, h/height, 0.0, 0.0]]).to(self.device)

                crop = img.crop([int(x), int(y), int(x + w), int(y + h)])
                bbx = crop.resize((224, 224), Image.ANTIALIAS)
                ret = self.resnet34(bbx)
                app = ret.data
                apps.append([torch.cat((motion, app), dim=1), conf_score, vr])

            self.bbx[self.f_step] = bbx_container
        if tag:
            self.detections[self.cur] = apps
        else:
            self.detections[self.nxt] = apps

    def updateVelocity(self, i, j):
        x1, y1, w1, h1, id1, conf_score1, vr1, width1, height1 = self.bbx[self.f_step-1][i]
        x2, y2, w2, h2, id2, conf_score2, vr2, width2, height2 = self.bbx[self.f_step][j]
        v_x = x2+w2/2 - (x1+w1/2)
        v_y = y2+h2/2 - (y1+h1/2)
        self.detections[self.nxt][j][0][0][4] = v_x/width1
        self.detections[self.nxt][j][0][0][5] = v_y/height1

    def getVelocity(self):
        for (i, j) in self.matches:
            self.updateVelocity(i, j)

    def initEC(self):
        self.m = len(self.detections[self.cur])
        self.n = len(self.detections[self.nxt])
        self.candidates = []
        self.edges = self.getMN(self.m, self.n)
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
                v = self.detections[i][j][0]
                vs.append(v)

        self.E = self.aggregate(es).to(self.device).view(1,-1)
        self.V = self.aggregate(vs).to(self.device)

    def loadNext(self):
        self.f_step += 1
        self.feature()
        self.initEC()
        # print '     The index of the next frame', self.f_step
        # print self.detections[self.cur]
        # print self.detections[self.nxt]

    def showE(self):
        with torch.no_grad():
            out = open(self.outName, 'a')
            print >> out, ''
            print >> out, '-'*45, '-'*45
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