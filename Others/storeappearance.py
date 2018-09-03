from torchvision.transforms import ToTensor
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch, cv2, torchvision, os
from torch.autograd import Variable
from PIL import Image
import random, shutil


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


class storeApp():
    def __init__(self, cuda=True):
        self.cuda = cuda
        self.Appearance = appearance()
        if self.cuda:
            self.Appearance = self.Appearance.cuda()
        self.Appearance.eval()

        self.trainDir = 'MOT16/train/'
        self.testDir = 'MOT16/test/'
        self.label = ['', 'Pedestrian', 'Person on vehicle',
                      'Car', 'Bicycle', 'Motorbike', 'Non motorized vehicle',
                      'Static person', 'Distractor', 'Occluder',
                      'Occluder on the ground', 'Occluder full', 'Reflection']

    def reset(self):
        self.dirAll = []

    def fixBB(self, x, y, w, h, size):
        width, height = size
        w = min(w+x, width)
        h = min(h+y, height)
        x = max(x, 0)
        y = max(y, 0)
        w -= x
        h -= y
        return x, y, w, h

    def outputApp(self, tag, show, refresh):
        """
        :param tag: 0 - train, 1 - test
        :param show: 1 - show the image and crop in real-time
        :param refresh: 1 - resave the crop from the pristine image
        :return: None
        """
        self.reset()
        basis = self.trainDir if tag == 0 else self.testDir
        # trainList = os.listdir(basis)
        # trainList = ['MOT16-05', 'MOT16-10', 'MOT16-11', 'MOT16-13']
        # trainList = ['MOT16-05']
        trainList = ['in_place', 'right_left']
        print trainList
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
            print 'The length of the sequence:', seqL

            # read the image
            imgs = [0] # store the sequence
            imgDir = part+'/img1/'
            for i in xrange(1, seqL+1):
                img = load_img(imgDir + '%06d.jpg'%i)
                imgs.append(img)

            # save the bb from the pristine image
            detsDir = part+'/dets/'
            if not os.path.exists(detsDir):
                # os.mkdir(detsDir)
                print '{} does not exist.'.format(detsDir)
            else:
                shutil.rmtree(detsDir)
            print detsDir, os.path.exists(detsDir)

            gtsDir = part+'/gts/'
            if not os.path.exists(gtsDir):
                # os.mkdir(gtsDir)
                print '{} does not exist.'.format(gtsDir)
            else:
                shutil.rmtree(gtsDir)
            print gtsDir, os.path.exists(gtsDir)


            if tag == 0:
                # Store the appearance
                outApp = open(part+'/gt/appearance.txt', 'w')

                # get the gt
                gt = part + '/gt/gt.txt'
                f = open(gt, 'r')
                if show:
                    cv2.namedWindow('view', flags=0)
                    cv2.namedWindow('crop', flags=0)
                pre = -1
                for line in f.readlines():
                    line = line.strip().split(',')
                    if line[7] == '1' or '_' in part:
                        """
                        Condition needed be taken into consideration:
                            x, y < 0 and x+w > W, y+h > H
                        """
                        index = int(line[0])
                        id = int(line[1])
                        x, y = int(float(line[2])), int(float(line[3]))
                        w, h = int(float(line[4])), int(float(line[5]))
                        l, vr = int(line[7]), float(line[8])

                        # sweep the invisible head-bbx from the training data
                        if pre != id and vr == 0:
                            continue

                        img = imgs[index]
                        x, y, w, h = self.fixBB(x, y, w, h, img.size)

                        pre = id
                        crop = imgs[index].crop([x, y, x+w, y+h])
                        bbx = crop.resize((224, 224), Image.ANTIALIAS)
                        ret = self.resnet34(bbx)
                        if self.cuda:
                            app = ret.cpu().data
                        else:
                            app = ret.data
                        app = app.numpy()[0]
                        for a in app:
                            print >> outApp, a,
                        print >> outApp, ''

                        if refresh:
                            crop.save(gtsDir+'{}_{}_{}_{}_{}_{}.bmp'.format(index, id, x, y, w, h))
                        if show:
                            img = np.asarray(img)
                            crop = np.asarray(crop)
                            print line
                            print part, index, seqL, '%06d'%index, self.label[l]
                            print w, h
                            print len(crop[0]), len(crop)
                            cv2.imshow('crop', crop)
                            cv2.imshow('view', img)
                            cv2.waitKey(34)
                if show:
                    raw_input('Continue?')
                f.close()
                outApp.close()

    def resnet34(self, img):
        bbx = ToTensor()(img)
        bbx = Variable(bbx, volatile=True)
        if self.cuda:
            bbx = bbx.cuda()
        bbx = bbx.view(-1, bbx.size(0), bbx.size(1), bbx.size(2))
        ret = self.Appearance(bbx)
        ret = ret.view(1, -1)
        return ret

try:
    test = storeApp()
    test.outputApp(0, 0, 0)
    print ''
except KeyboardInterrupt:
    print ''
    print '-'*90
    print 'Existing from training early.'