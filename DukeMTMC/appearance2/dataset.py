import torch.utils.data as data
import cv2, random, torch, shutil, os
import numpy as np
from PIL import Image
import torch.nn.functional as F
from mot_model import appearance
from global_set import edge_initial, app_fine_tune, fine_tune_dir, overlap, SEQLEN
from torchvision.transforms import ToTensor
from scipy.io import loadmat


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, camera, outName, cuda=True, show=0):
        super(DatasetFromFolder, self).__init__()
        self.seq_index = camera

        frames_dir = ['',
                      '/media/lee/MOT1/camera1/',
                      '/media/lee/MOT1/camera2/',
                      '/media/lee/MOT1/camera3/',
                      '/media/lee/MOT2/camera4/',
                      '/media/lee/MOT2/camera5/',
                      '/media/lee/DATA/DukeMTMC/frames/camera6/',
                      '/media/lee/DATA/DukeMTMC/frames/camera7/',
                      '/media/lee/DATA/DukeMTMC/frames/camera8/']
        self.img_dir = frames_dir[camera]

        self.width, self.height = 1920, 1080

        # clean the content in the text file
        self.outName = outName
        out = open(self.outName, 'w')
        out.close()

        self.device = torch.device("cuda" if cuda else "cpu")
        self.show = show

        self.loadAModel()
        self.readBBx()
        self.initBuffer()

    def loadAModel(self):
        if app_fine_tune:
            self.Appearance = torch.load(fine_tune_dir)
        else:
            self.Appearance = appearance()
        self.Appearance.to(self.device)
        self.Appearance.eval()  # fixing the BatchN layer

    def generator(self, bbx):
        if random.randint(0, 1):
            x, y, w, h = bbx
            x, y, w = float(x), float(y), float(w),
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
            ans = [int(x), int(y), int(w), h]
            return ans
        return bbx

    def readBBx(self):
        # get the gt
        self.bbx = [[] for i in xrange(SEQLEN + 1)]

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
            x, y = int(bbx[3]), int(bbx[4])
            w, h = int(bbx[5]), int(bbx[6])
            if camera == self.seq_index and (frame >= head and frame <= tail):
                x, y, w, h = self.generator([x, y, w, h])
                self.bbx[frame-head+1].append([x, y, w, h, id, frame])

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
            Reframe = self.bbx[self.f_step-self.gap][i]
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
            if len(self.bbx[self.f_step]):
                frame = self.bbx[self.f_step][0][5]
                img = load_img(self.img_dir+'%06d.jpg'%frame)  # initial with loading the first frame
                for bbx in self.bbx[self.f_step]:
                    """
                    Condition needed be taken into consideration:
                        x, y < 0 and x+w > W, y+h > H
                    """
                    x, y, w, h, id, frame = bbx
                    x, y, w, h = self.fixBB(x, y, w, h)
                    bbx_container.append([x, y, w, h, id, frame])
                    crop = img.crop([x, y, x + w, y + h])
                    bbx = crop.resize((224, 224), Image.ANTIALIAS)
                    ret = self.resnet34(bbx)
                    app = ret.data
                    apps.append([app, id])

                    if self.show:
                        img1 = np.asarray(img)
                        crop1 = np.asarray(crop)
                        print '%06d'%self.f_step, id, frame, '***',
                        print w, h, '-',
                        print len(crop1[0]), len(crop1)
                        cv2.imshow('crop', crop1)
                        cv2.imshow('view', img1)
                        cv2.waitKey(34)
                        # raw_input('Continue?')
                        # cv2.waitKey(34)
            self.bbx[self.f_step] = bbx_container
        if tag:
            self.detections[self.cur] = apps
        else:
            self.detections[self.nxt] = apps

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

        for i in xrange(self.m):
            for j in xrange(self.n):
                e = self.edges[i][j]
                gt = self.gts[i][j]
                self.candidates.append([e, gt, i, j])

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

        self.initEC()
        return self.gap

    def showE(self):
        with torch.no_grad():
            out = open(self.outName, 'a')
            print >> out, ''
            print >> out, '-'*45, '%d * %d'%(self.m, self.n), '-'*45
            print >> out, '     edge'

            print >> out, ' ', self.f_step-self.gap
            print >> out, self.outE(self.edges, 1)

            print >> out, '     connection'

            print >> out, ' ', self.f_step-self.gap
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