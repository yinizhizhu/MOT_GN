from torchvision.transforms import ToTensor
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch, cv2, torchvision, os
from torch.autograd import Variable
from PIL import Image
import random
from global_set import criterion_s, edge_initial


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


def IOU(Reframe, GTframe):
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

    endx = max(x1+width1,x2+width2)
    startx = min(x1,x2)
    width = width1+width2-(endx-startx)

    endy = max(y1+height1,y2+height2)
    starty = min(y1,y2)
    height = height1+height2-(endy-starty)

    if width <= 0 or height <= 0:
        ratio = 0
    else:
        Area = width*height
        Area1 = width1*height1
        Area2 = width2*height2
        ratio = Area*1./(Area1+Area2-Area)
    return ratio


def appSimilar(app1, app2):
    return torch.mean(torch.abs(app1-app2))


def getMNForCrossEntropyLoss(app1, app2, bbx1, bbx2):
    m = len(bbx1)
    n = len(bbx2)
    ans = [[None for i in xrange(n)] for i in xrange(m)]
    for i in xrange(m):
        am = app1[i]
        for j in xrange(n):
            an = app2[j]
            if edge_initial == 2:
                iou = 1 - appSimilar(am, an)
            else:
                iou = IOU(bbx1[i], bbx2[j])
            # 1 - match, 0 - mismatch
            ans[i][j] = torch.FloatTensor([1-iou, iou]).view(1, -1)
    return ans


def getMNForMSELoss(app1, app2, bbx1, bbx2):
    m = len(bbx1)
    n = len(bbx2)
    ans = [[None for i in xrange(n)] for i in xrange(m)]
    for i in xrange(m):
        am = app1[i]
        for j in xrange(n):
            an = app2[j]
            if edge_initial == 2:
                iou = appSimilar(am, an)
            else:
                iou = IOU(bbx1[i], bbx2[j])
            # 1 - match, 0 - mismatch
            ans[i][j] = torch.FloatTensor([iou]).view(1, -1)
    return ans


class appearance(nn.Module):
    def __init__(self):
        super(appearance, self).__init__()
        features = list(torchvision.models.resnet34(pretrained=True).children())[:-1]
        # print features
        self.features = nn.Sequential(*features)

    def forward(self, x):
        return self.features(x)


class readBB():
    def __init__(self, cuda=True):
        self.cuda = cuda
        self.Appearance = appearance()
        if self.cuda:
            self.Appearance = self.Appearance.cuda()
        self.Appearance.eval()  # Essential step - to avoid the effect of BatchNormalization

        self.trainDir = 'MOT16/train/'
        self.testDir = 'MOT16/test/'
        self.label = ['', 'Pedestrian', 'Person on vehicle',
                      'Car', 'Bicycle', 'Motorbike', 'Non motorized vehicle',
                      'Static person', 'Distractor', 'Occluder',
                      'Occluder on the ground', 'Occluder full', 'Reflection']

    def reset(self):
        self.dirAll = []
        self.bbAll = [] # all the bbxes of ground-truth
        self.eAll = [] # Edges: all the edges of the detection's graph
        self.cAll = [] # true connection - ground truth of edge
        self.aAll = [] # the appearance of bbxes

    def preprocessForCrossEntropyLoss(self, tag, show, refresh):
        """
        :param tag: 0 - train, 1 - test
        :param show: 1 - show the image and crop in real-time
        :param refresh: 1 - resave the crop from the pristine image
        :return: None
        """
        self.reset()
        basis = self.trainDir if tag == 0 else self.testDir
        # trainList = os.listdir(basis)
        # trainList = ['in_place', 'right_left']
        trainList = ['in_place']
        for part in trainList:
            part = basis+part
            self.dirAll.append(part)

            # get the length of the sequence
            info = part+'/seqinfo.ini'
            f = open(info, 'r')
            f.readline()
            for line in f.readlines():
                line = line.strip().split('=')
                if line[0] == 'seqLength':
                    seqL = int(line[1])
            f.close()

            # read the image
            imgs = [0] # store the sequence
            imgDir = part+'/img1/'
            for i in xrange(1, seqL+1):
                img = load_img(imgDir + '%06d.jpg'%i)
                imgs.append(img)

            # save the bb from the pristine image
            detsDir = part+'/dets/'
            if not os.path.exists(detsDir):
                os.mkdir(detsDir)
            print detsDir, os.path.exists(detsDir), '-',
            gtsDir = part+'/gts/'
            if not os.path.exists(gtsDir):
                os.mkdir(gtsDir)
            print gtsDir, os.path.exists(gtsDir)

            # Store the appearance
            apps = [[] for i in xrange(seqL+1)]

            if tag == 0:
                # get the gt
                gts = [[] for i in xrange(seqL + 1)]
                gt = part + '/gt/gt.txt'
                f = open(gt, 'r')
                if show:
                    cv2.namedWindow('view', flags=0)
                    cv2.namedWindow('crop', flags=0)
                for line in f.readlines():
                    line = line.strip().split(',')
                    index = int(line[0])
                    id = int(line[1])
                    x, y = int(float(line[2])), int(float(line[3]))
                    w, h = int(float(line[4])), int(float(line[5]))

                    img = imgs[index]
                    gts[index].append([x,y,w,h, id])
                    crop = imgs[index].crop([x, y, x+w, y+h])
                    bbx = crop.resize((224, 224), Image.ANTIALIAS)
                    ret = self.resnet34(bbx)
                    if self.cuda:
                        apps[index].append(ret.cpu().data)
                    else:
                        apps[index].append(ret.data)

                    if refresh:
                        crop.save(gtsDir+'{}_{}_{}_{}_{}_{}.bmp'.format(index, id, x, y, w, h))
                    if show:
                        img = np.asarray(img)
                        crop = np.asarray(crop)
                        print line
                        print part, index, seqL, '%06d'%index
                        print w, h
                        print len(crop[0]), len(crop)
                        cv2.imshow('crop', crop)
                        cv2.imshow('view', img)
                        cv2.waitKey(34)
                if show:
                    raw_input('Continue?')
                f.close()
                self.bbAll.append(gts)
            self.aAll.append(apps)

        # get the edges (IoU), and gts (1 - True, 0 - False)
        for s in xrange(len(self.bbAll)):  # sequence
            edges = [[]]
            gts = [[]]
            for f in xrange(1, len(self.bbAll[s])-1):  # frame
                appm = self.aAll[s][f]
                appn = self.aAll[s][f+1]
                m = self.bbAll[s][f]
                n = self.bbAll[s][f+1]
                edges.append(getMNForCrossEntropyLoss(appm, appn, m, n))
                gts.append([[torch.LongTensor([0]) for i in xrange(len(n))] for j in xrange(len(m))])
            self.eAll.append(edges)
            self.cAll.append(gts)

    def preprocessForMSELoss(self, tag, show, refresh):
        """
        :param tag: 0 - train, 1 - test
        :param show: 1 - show the image and crop in real-time
        :param refresh: 1 - resave the crop from the pristine image
        :return: None
        """
        self.reset()
        basis = self.trainDir if tag == 0 else self.testDir
        # trainList = os.listdir(basis)
        # trainList = ['in_place', 'right_left']
        trainList = ['in_place']
        for part in trainList:
            part = basis+part
            self.dirAll.append(part)

            # get the length of the sequence
            info = part+'/seqinfo.ini'
            f = open(info, 'r')
            f.readline()
            for line in f.readlines():
                line = line.strip().split('=')
                if line[0] == 'seqLength':
                    seqL = int(line[1])
            f.close()

            # read the image
            imgs = [0] # store the sequence
            imgDir = part+'/img1/'
            for i in xrange(1, seqL+1):
                img = load_img(imgDir + '%06d.jpg'%i)
                imgs.append(img)

            # save the bb from the pristine image
            detsDir = part+'/dets/'
            if not os.path.exists(detsDir):
                os.mkdir(detsDir)
            print detsDir, os.path.exists(detsDir), '-',
            gtsDir = part+'/gts/'
            if not os.path.exists(gtsDir):
                os.mkdir(gtsDir)
            print gtsDir, os.path.exists(gtsDir)

            # Store the appearance
            apps = [[] for i in xrange(seqL+1)]

            if tag == 0:
                # get the gt
                gts = [[] for i in xrange(seqL + 1)]
                gt = part + '/gt/gt.txt'
                f = open(gt, 'r')
                if show:
                    cv2.namedWindow('view', flags=0)
                    cv2.namedWindow('crop', flags=0)
                for line in f.readlines():
                    line = line.strip().split(',')
                    index = int(line[0])
                    id = int(line[1])
                    x, y = int(float(line[2])), int(float(line[3]))
                    w, h = int(float(line[4])), int(float(line[5]))

                    img = imgs[index]
                    gts[index].append([x,y,w,h, id])
                    crop = imgs[index].crop([x, y, x+w, y+h])
                    bbx = crop.resize((256, 256), Image.ANTIALIAS)
                    ret = self.resnet34(bbx)
                    if self.cuda:
                        apps[index].append(ret.cpu().data)
                    else:
                        apps[index].append(ret.data)

                    if refresh:
                        crop.save(gtsDir+'{}_{}_{}_{}_{}_{}.bmp'.format(index, id, x, y, w, h))
                    if show:
                        img = np.asarray(img)
                        crop = np.asarray(crop)
                        print line
                        print part, index, seqL, '%06d'%index
                        print w, h
                        print len(crop[0]), len(crop)
                        cv2.imshow('crop', crop)
                        cv2.imshow('view', img)
                        cv2.waitKey(34)
                if show:
                    raw_input('Continue?')
                f.close()
                self.bbAll.append(gts)
            self.aAll.append(apps)

        # get the edges (IoU), and gts (1 - True, 0 - False)
        for s in xrange(len(self.bbAll)):  # sequence
            edges = [[]]
            gts = [[]]
            for f in xrange(1, len(self.bbAll[s])-1):  # frame
                appm = self.aAll[s][f]
                appn = self.aAll[s][f+1]
                m = self.bbAll[s][f]
                n = self.bbAll[s][f+1]
                edges.append(getMNForMSELoss(appm, appn, m, n))
                gts.append([[torch.FloatTensor([0.0]).view(1, -1) for i in xrange(len(n))] for j in xrange(len(m))])
            self.eAll.append(edges)
            self.cAll.append(gts)

    def resnet34(self, img):
        bbx = ToTensor()(img)
        bbx = Variable(bbx, volatile=True)
        if self.cuda:
            bbx = bbx.cuda()
        bbx = bbx.view(-1, bbx.size(0), bbx.size(1), bbx.size(2))
        ret = self.Appearance(bbx)
        ret = ret.view(1, -1)
        return ret

    def getC(self, t=0):
        if criterion_s:
            self.preprocessForMSELoss(tag=t, show=0, refresh=0)
        else:
            self.preprocessForCrossEntropyLoss(tag=t, show=0, refresh=0)
        for s in xrange(len(self.bbAll)): # sequence
            for f in xrange(1, len(self.bbAll[s])-1): # frame
                x = 0
                for bb in self.bbAll[s][f]:
                    y = 0
                    for bbn in self.bbAll[s][f+1]:
                        if bb[4] == bbn[4]:
                            if criterion_s:
                                self.cAll[s][f][x][y] = torch.FloatTensor([1.0]).view(1, -1)
                            else:
                                self.cAll[s][f][x][y] = torch.LongTensor([1])
                            break
                        y += 1
                    x += 1
        self.frame_end = len(self.cAll[0])
        self.clipShow()
        return self.bbAll, self.aAll, self.eAll, self.cAll, self.frame_end-1

    def clipShow(self):
        print 'Clipshowing...'
        f = open('tmp.txt', 'w')
        print >> f, '   bbAll'
        print >> f, torch.FloatTensor(self.bbAll[0][1])
        print >> f, torch.FloatTensor(self.bbAll[0][2])
        print >> f, ''
        # print >> f, '   aAll'
        # print >> f, self.aAll[0][1]
        # print >> f, ''
        for i in xrange(1, self.frame_end):
            print >> f, '   eAll'
            print >> f, ' ', i
            if criterion_s:
                print >> f, self.outEForMSELoss(self.eAll[0][i], 1)
            else:
                print >> f, self.outEForCrossEntropyLoss(self.eAll[0][i], 1)
            print >> f, ''
            print >> f, '   cAll'
            print >> f, ' ', i
            if criterion_s:
                print >> f, self.outEForMSELoss(self.cAll[0][i], 0)
            else:
                print >> f, self.outEForCrossEntropyLoss(self.cAll[0][i], 0)
            print >> f, ''

        print >> f, '\n The number of sequence:', len(self.bbAll)
        counter_s = 1
        for s in self.bbAll:
            print >> f, 'The {} sequence:'.format(counter_s)
            print >> f, ' The number of frame:', len(s)
            counter_f = 0
            for frame in s:
                print >> f, '      The {} frame:'.format(counter_f)
                print >> f, '         The number of detection:', len(frame)
                print >> f, ''
                counter_f += 1
            print >> f, ''
            counter_s += 1

        f.close()

    def outEForCrossEntropyLoss(self, edges, tag):
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

    def outEForMSELoss(self, edges, tag):
        m = len(edges)
        n = len(edges[0])
        ans = [[None for i in xrange(n)] for j in xrange(m)]
        for i in xrange(m):
            for j in xrange(n):
                tmp = edges[i][j].numpy()[0][0] if tag else edges[i][j].numpy()[0]
                ans[i][j] = float(tmp)
        ans = torch.FloatTensor(ans)
        return ans

    def show(self):
        """
        Show the clip of the sequence to judge the condition
        :return: None
        """
        basis = self.trainDir
        part = 'in_place'
        part = basis+part

        # get the length of the sequence
        info = part+'/seqinfo.ini'
        f = open(info, 'r')
        f.readline()
        for line in f.readlines():
            line = line.strip().split('=')
            if line[0] == 'seqLength':
                seqL = int(line[1])
        f.close()

        # read the image
        imgs = [0] # store the sequence
        imgDir = part+'/img1/'
        for i in xrange(1, seqL+1):
            img = load_img(imgDir + '%06d.jpg'%i)
            imgs.append(img)

        # get the gt
        gt = part + '/show.txt'
        cv2.namedWindow('view', flags=0)
        cv2.namedWindow('crop1', flags=0)
        cv2.namedWindow('crop2', flags=0)
        cv2.namedWindow('crop3', flags=0)
        cv2.namedWindow('crop4', flags=0)
        while True:
            f = open(gt, 'r')
            bbxes = [[] for i in xrange(4)]
            for line in f.readlines():
                line = line.strip().split(',')
                index = int(line[0])
                id = int(line[1])
                x = int(float(line[2]))
                y = int(float(line[3]))
                w = int(float(line[4]))
                h = int(float(line[5]))
                l = int(float(line[7]))
                bbxes[id-1].append([index, x, y, w, h])
            min_n = 1608
            for bbx in bbxes:
                min_n = min(min_n, len(bbx))

            for i in xrange(min_n):
                index, x, y, w, h = bbxes[0][i]
                img0 = imgs[index]
                # x, y, w, h = self.fixBB(x, y, w, h, img.size)
                crop = img0.crop([x, y, x+w, y+h])
                img = np.asarray(img0)
                crop = np.asarray(crop)
                cv2.imshow('crop1', crop)
                cv2.imshow('view', img)

                for j in xrange(2, 5):
                    index, x, y, w, h = bbxes[j-1][i]
                    crop = img0.crop([x, y, x+w, y+h])
                    crop = np.asarray(crop)
                    cv2.imshow('crop{}'.format(j), crop)

                cv2.waitKey(34)
            raw_input('Continue?')
            f.close()

try:
    # test = readBB()
    # test.show()
    # test.getC(0)
    print ' In the mot_dataset.py...'
except KeyboardInterrupt:
    print ''
    print '-'*90
    print 'Existing from training early.'