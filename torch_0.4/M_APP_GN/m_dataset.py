import torch.utils.data as data
import random, torch, shutil, os
from PIL import Image
import torch.nn.functional as F
from m_mot_model import appearance
from torchvision.transforms import ToTensor
from m_global_set import edge_initial, overlap,app_fine_tune, fine_tune_dir


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

        self.loadAModel()
        self.getSeqL()
        self.readBBx()
        self.initBuffer()
        print '     Data loader is already!'

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
        f.close()
        print '     The length of the sequence:', self.seqL

    def fixBB(self, x, y, w, h, size):
        width, height = size
        w = min(w+x, width)
        h = min(h+y, height)
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
        self.bbxApp = [[] for i in xrange(self.seqL + 1)]
        imgs = [None for i in xrange(self.seqL + 1)]
        for i in xrange(1, self.seqL+1):
            imgs[i] = load_img(self.img_dir+'%06d.jpg'%i)
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
                x, y, w, h = self.generator([x, y, w, h])
                x, y, w, h = self.fixBB(x, y, w, h, img.size)
                width, height = float(img.size[0]), float(img.size[1])
                self.bbx[index].append([x/width, y/height, w/width, h/height, id, vr])
                self.bbxApp[index].append([int(x), int(y), int(w), int(h)])
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

    def swapFC(self):
        self.getVelocity()
        self.cur = self.cur ^ self.nxt
        self.nxt = self.cur ^ self.nxt
        self.cur = self.cur ^ self.nxt

    def getMApp(self, tag, index, pre_index=None):
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
                x, y = index[0], pre_index[0]
                if n == 1:
                    return torch.cat((self.detections[cur][x][0][y], self.detections[cur][x][2]), dim=1)
                x2, y2 = index[1], pre_index[1]
                ans = torch.cat((torch.cat((self.detections[cur][x][0][y], self.detections[cur][x][2]), dim=1),
                                 torch.cat((self.detections[cur][x2][0][y2], self.detections[cur][x2][2]), dim=1)), dim=0)
                for i in xrange(2, n):
                    x, y = index[i], pre_index[i]
                    ans = torch.cat((ans,
                                     torch.cat((self.detections[cur][x][0][y], self.detections[cur][x][2]), dim=1)), dim=0)
                return ans
            x = index[0]
            if n == 1:
                return torch.cat((self.detections[cur][x][0][0], self.detections[cur][x][2]), dim=1)
            x2 = index[1]
            ans = torch.cat((torch.cat((self.detections[cur][x][0][0], self.detections[cur][x][2]), dim=1),
                             torch.cat((self.detections[cur][x2][0][0], self.detections[cur][x2][2]), dim=1)), dim=0)
            for i in xrange(2, n):
                x = index[i]
                ans = torch.cat((ans,
                                 torch.cat((self.detections[cur][x][0][0], self.detections[cur][x][2]), dim=1)), dim=0)
            return ans
        if tag == 0:
            self.updateVelocity(pre_index, index)
            return torch.cat((self.detections[cur][index][0][pre_index], self.detections[cur][index][2]), dim=1)
        return torch.cat((self.detections[cur][index][0][0], self.detections[cur][index][2]), dim=1)

    def updateVelocity(self, i, j, tag=True):
        '''
        :param i: cur_index, -1 - birth
        :param j: nxt_index
        :param tag: True - update the velocity in the next frame, False - Write down the final velocity
        :return:
        '''
        v_x, v_y = 0.0, 0.0
        if i >= 0:
            x1, y1, w1, h1, id1, vr1 = self.bbx[self.f_step-1][i]
            x2, y2, w2, h2, id2, vr2 = self.bbx[self.f_step][j]
            v_x = x2+w2/2 - (x1+w1/2)
            v_y = y2+h2/2 - (y1+h1/2)
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
        cur = self.f_step - 1
        ans = [[None for j in xrange(self.n)] for i in xrange(self.m)]
        for i in xrange(self.m):
            Reframe = self.bbx[cur][i]
            for j in xrange(self.n):
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

    def initEC(self):
        self.m = len(self.detections[self.cur])
        self.n = len(self.detections[self.nxt])
        self.candidates = []
        self.edges = self.getMN()
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
                # print self.detections[i][j][0][0]
                # print self.detections[i][j][1]
                # print self.detections[i][j][2]
                v = torch.cat((self.detections[i][j][0][0], self.detections[i][j][2]), dim=1)
                vs.append(v)

        self.E = self.aggregate(es).to(self.device).view(1,-1)
        self.V = self.aggregate(vs).to(self.device)

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
        m_apps = []
        with torch.no_grad():
            j = 0
            m = 1 if tag else len(self.bbx[self.f_step-1])
            img = load_img(self.img_dir+'%06d.jpg'%self.f_step)
            for bbx in self.bbx[self.f_step]:
                """
                Condition needed be taken into consideration:
                    x, y < 0 and x+w > W, y+h > H
                """
                x, y, w, h, id, vr = bbx
                x += w/2
                y += h/2
                cur_m = []
                for i in xrange(m):
                    cur_m.append(torch.FloatTensor([[x, y, w, h, 0.0, 0.0]]).to(self.device))

                x, y, w, h = self.bbxApp[self.f_step][j]
                crop = img.crop([x, y, x + w, y + h])
                bbx = crop.resize((224, 224), Image.ANTIALIAS)
                ret = self.resnet34(bbx)
                app = ret.data
                m_apps.append([cur_m, id, app])
                j += 1
        if tag:
            self.detections[self.cur] = m_apps
        else:
            self.detections[self.nxt] = m_apps

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
