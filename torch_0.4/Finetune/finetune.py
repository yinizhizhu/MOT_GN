from torchvision.transforms import ToTensor
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch, cv2, torchvision, os
from torch.autograd import Variable
from PIL import Image


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


class finetuning():
    def __init__(self, cuda=True):
        self.cuda = cuda
        self.Appearance = appearance()
        if self.cuda:
            self.Appearance = self.Appearance.cuda()
        self.Appearance.train()  # Essential step - to avoid the effect of BatchNormalization

        self.optimizer = optim.Adam(self.Appearance.parameters(), lr=1e-3)

        self.trainDir = 'MOT16/train/'
        self.testDir = 'MOT16/test/'
        self.label = ['', 'Pedestrian', 'Person on vehicle',
                      'Car', 'Bicycle', 'Motorbike', 'Non motorized vehicle',
                      'Static person', 'Distractor', 'Occluder',
                      'Occluder on the ground', 'Occluder full', 'Reflection']

    def reset(self):
        self.images = [[] for i in xrange(4)]  # the appearance of bbxes

    def loadData(self, tag, show):
        """
        :param tag: 0 - train, 1 - test
        :param show: 1 - show the image and crop in real-time
        :return: None
        """
        self.reset()
        basis = self.trainDir if tag == 0 else self.testDir
        # trainList = os.listdir(basis)
        # trainList = ['in_place', 'right_left']
        trainList = ['in_place']
        for part in trainList:
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

            if tag == 0:
                gt = part + '/gt/gt.txt'
                f = open(gt, 'r')
                if show:
                    cv2.namedWindow('view', flags=0)
                    cv2.namedWindow('crop', flags=0)
                for line in f.readlines():
                    line = line.strip().split(',')
                    index = int(line[0])
                    id = int(line[1]) - 1
                    x, y = int(float(line[2])), int(float(line[3]))
                    w, h = int(float(line[4])), int(float(line[5]))

                    img = imgs[index]
                    crop = imgs[index].crop([x, y, x+w, y+h])
                    bbx = crop.resize((224, 224), Image.ANTIALIAS)
                    self.images[id].append(bbx)

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

    def resnet34(self, img):
        bbx = ToTensor()(img)
        bbx = Variable(bbx)
        if self.cuda:
            bbx = bbx.cuda()
        bbx = bbx.view(-1, bbx.size(0), bbx.size(1), bbx.size(2))
        return bbx

    def mse(self, a, b):
        c = a - b
        ret = torch.mean(c*c)
        return ret

    def finetune(self):
        step = 0
        f = open('finetune.txt', 'w')
        for anchor in xrange(4):
            anchor_n = len(self.images[anchor])
            for anchor_i in xrange(anchor_n):
                img = self.images[anchor][anchor_i]  # the anchor image
                img_v = self.resnet34(img)
                for anchor_j in xrange(anchor_n):
                    if anchor_j != anchor_i:
                        ps = self.images[anchor][anchor_j]  # the positive sample
                        ps_v = self.resnet34(ps)
                        for neg in xrange(4):
                            if anchor != neg:
                                neg_n = len(self.images[neg])
                                for neg_i in xrange(neg_n):
                                    one = Variable(torch.FloatTensor([1.0])).cuda()
                                    zero = Variable(torch.FloatTensor([0.0])).cuda()
                                    ns = self.images[neg][neg_i]  # the negative sample
                                    ns_v = self.resnet34(ns)
                                    ns_f = self.Appearance(ns_v)    # negative
                                    img_f = self.Appearance(img_v)  # anchor
                                    ps_f = self.Appearance(ps_v)    # positive

                                    self.optimizer.zero_grad()
                                    loss1 = self.mse(img_f, ns_f)/2
                                    epoch_loss = loss1.data[0]
                                    print >> f, step, epoch_loss,

                                    loss2 = self.mse(ps_f, ns_f)/2
                                    epoch_loss = loss2.data[0]
                                    print >> f, epoch_loss,

                                    loss3 = self.mse(img_f, ps_f)
                                    epoch_loss = loss3.data[0]
                                    print >> f, epoch_loss
                                    triplet_loss = one+loss3-loss1-loss2
                                    loss = torch.max(zero, triplet_loss)
                                    loss.backward()
                                    self.optimizer.step()
                                    step += 1
        f.close()
        print 'Finish the finetuning!'
try:
    ft = finetuning()
    print 'Loading data...'
    ft.loadData(0, 0)
    print 'Finetuning...'
    ft.finetune()
except KeyboardInterrupt:
    print ''
    print '-'*90
    print 'Existing from training early.'