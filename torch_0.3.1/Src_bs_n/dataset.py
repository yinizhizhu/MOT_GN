import torch.utils.data as data
import torchvision, cv2, random, torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import ToTensor


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class appearance(nn.Module):
    def __init__(self):
        super(appearance, self).__init__()
        features = list(torchvision.models.resnet34(pretrained=True).children())[:-1]
        # print features
        self.features = nn.Sequential(*features)

    def forward(self, x):
        return self.features(x)


class DatasetFromFolder(data.Dataset):
    def __init__(self, part, outName, cuda=False, show=0):
        super(DatasetFromFolder, self).__init__()
        self.dir = part
        self.img_dir = part + '/img1/'
        self.gt_dir = part + '/gt/'
        self.det_dir = part + '/det/'
        self.outName = outName
        self.cuda = cuda
        self.show = show
        self.step_ed = 0.0

        self.loadAModel()
        self.getSeqL()
        self.readBBx()
        self.initBuffer()

    def loadAModel(self):
        self.Appearance = appearance()
        if self.cuda:
            self.Appearance = self.Appearance.cuda()
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
        print 'The length of the sequence:', self.seqL

    def readBBx(self):
        # get the gt
        self.bbx = [[] for i in xrange(self.seqL + 1)]
        gt = self.gt_dir + '/gt.txt'
        f = open(gt, 'r')
        pre = -1
        for line in f.readlines():
            line = line.strip().split(',')
            if line[7] == '1':
                index = int(line[0])
                id = int(line[1])
                x, y = int(line[2]), int(line[3])
                w, h = int(line[4]), int(line[5])
                l, vr = int(line[7]), float(line[8])

                # sweep the invisible head-bbx from the training data
                if pre != id and vr == 0:
                    continue

                pre = id
                self.bbx[index].append([x, y, w, h, id, vr])
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

    def feature(self, tag=0):
        '''
        Getting the appearance of the detections in current frame
        :param tag: 1 - initiating
        :param show: 1 - show the cropped & src image
        :return: None
        '''
        apps = []
        for bbx in self.bbx[self.f_step]:
            """
            Condition needed be taken into consideration:
                x, y < 0 and x+w > W, y+h > H
            """
            img = load_img(self.img_dir+'%06d.jpg'%self.f_step)  # initial with loading the first frame
            x, y, w, h, id, vr = bbx
            x, y, w, h = self.fixBB(x, y, w, h, img.size)
            crop = img.crop([x, y, x + w, y + h])
            bbx = crop.resize((224, 224), Image.ANTIALIAS)
            ret = self.resnet34(bbx)
            app = Variable(ret.data)
            apps.append([app, id])

            if self.show:
                img = np.asarray(img)
                crop = np.asarray(crop)
                print '%06d'%self.f_step, id, vr, '***',
                print w, h, '-',
                print len(crop[0]), len(crop)
                cv2.imshow('crop', crop)
                cv2.imshow('view', img)
                cv2.waitKey(34)
                raw_input('Continue?')
                # cv2.waitKey(34)
        if tag:
            self.detections[self.cur] = apps
        else:
            self.detections[self.nxt] = apps

    def getMN(self, m, n):
        ans = [[None for i in xrange(n)] for i in xrange(m)]
        for i in xrange(m):
            for j in xrange(n):
                p = random.random()
                # 1 - match, 0 - mismatch
                ans[i][j] = torch.FloatTensor([1 - p, p]).view(1, -1)
        return ans

    def aggregate(self, set):
        if len(set):
            rho = sum(set)
            return rho/len(set)
        print '     The set is empty!'
        return None

    def getApp(self, tag, index):
        if tag:
            return self.detections[self.cur][index][0]
        return self.detections[self.nxt][index][0]

    def initEC(self):
        self.m = len(self.detections[self.cur])
        self.n = len(self.detections[self.nxt])
        self.candidates = []
        self.edges = self.getMN(self.m, self.n)
        self.gts = [[None for j in xrange(self.n)] for i in xrange(self.m)]
        self.step_gt = 0.0
        for i in xrange(self.m):
            for j in xrange(self.n):
                tag = int(self.detections[self.cur][i][1] == self.detections[self.nxt][j][1])
                self.gts[i][j] = torch.LongTensor([tag])
                self.step_gt += tag*1.0

        self.es = []
        vs_index = 0
        for i in xrange(self.m):
            vr_index = self.m
            for j in xrange(self.n):
                e = Variable(self.edges[i][j])
                gt = Variable(self.gts[i][j])
                if self.cuda:
                    e, gt = e.cuda(), gt.cuda()
                self.es.append(e)
                self.candidates.append([e, gt, vs_index, vr_index])
                vr_index += 1
            vs_index += 1

        self.vs = []
        for i in xrange(2):
            n = len(self.detections[i])
            for j in xrange(n):
                v = self.detections[i][j][0]
                self.vs.append(v)

        self.E = self.aggregate(self.es)
        self.V = self.aggregate(self.vs)

    def swapFC(self):
        self.cur = self.cur ^ self.nxt
        self.nxt = self.cur ^ self.nxt
        self.cur = self.cur ^ self.nxt

    def loadNext(self):
        self.f_step += 1
        self.feature()
        self.initEC()
        self.swapFC()
        print '     The index of the next frame', self.f_step
        # print self.detections[self.cur]
        # print self.detections[self.nxt]

    def showE(self):
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
                    tmp = F.softmax(Variable(edges[i][j])).data.numpy()[0]
                    ret[i][j] = float(tmp[1])

                    tmp = edges[i][j].numpy()[0]
                    con1[i][j] = float(tmp[0])
                    con2[i][j] = float(tmp[1])
                else:
                    tmp = edges[i][j].numpy()[0]
                    ans[i][j] = float(tmp)
        if tag:
            ret = torch.FloatTensor(ret)
            con1 = torch.FloatTensor(con1)
            con2 = torch.FloatTensor(con2)
            return 'Probability', ret, 'Output', con1, con2
        ans = torch.FloatTensor(ans)
        return ans

    def resnet34(self, img):
        bbx = ToTensor()(img)
        bbx = Variable(bbx, volatile=True)
        if self.cuda:
            bbx = bbx.cuda()
        bbx = bbx.view(-1, bbx.size(0), bbx.size(1), bbx.size(2))
        ret = self.Appearance(bbx)
        ret = ret.view(1, -1)
        return ret

    def __getitem__(self, index):
        ans = self.candidates[index]
        print len(ans)
        print ans[0].cpu().data
        print ans[1].cpu().data
        print ans[2]
        print ans[3]
        return ans

    def __len__(self):
        return len(self.candidates)
