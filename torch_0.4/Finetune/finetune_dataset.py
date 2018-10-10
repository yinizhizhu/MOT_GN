import random, time, cv2
from torchvision.transforms import ToTensor
import numpy as np
from math import *
from PIL import Image
import torch.utils.data as data
from global_set import tau_vr, tau_frame


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

overlap = 0.85
p_tag = 0  # 1 - nearby, 0 - no limitation


class DatasetFromFolder(data.Dataset):
    def __init__(self, tag):
        super(DatasetFromFolder, self).__init__()
        basic_dir = 'MOT16/train/MOT16-'
        parts = ['02', '04', '05', '09', '10', '11', '13']
        self.negative_tag = tag  # 1 - hard mining, 0 - random
        self.candidates = []
        self.counter = 0

        start = time.time()
        for part in parts:
            head = time.time()
            print '     ', part,
            part = basic_dir+part
            self.img_dir = part + '/img1/'
            self.gt_dir = part + '/gt/'
            self.det_dir = part + '/det/'

            self.getSeqL(part)
            self.readImg()
            self.readBBx()
            self.sampleCollection()
            print '     Time consuming:', (time.time()-head)/60.0
            # break
        print 'Time consuming:', (time.time()-start)/60.0

    def getSeqL(self, part):
        # get the length of the sequence
        info = part+'/seqinfo.ini'
        f = open(info, 'r')
        f.readline()
        for line in f.readlines():
            line = line.strip().split('=')
            if line[0] == 'seqLength':
                self.seqL = int(line[1])
        f.close()
        print ' *  The length of the sequence:', self.seqL, ' * ',

        # get the number of the tracks
        gt = self.gt_dir + 'gt.txt'
        f = open(gt, 'r')
        self.track_num = 0
        for line in f.readlines():
            line = line.strip().split(',')
            id = int(line[1])
            if id > self.track_num:
                self.track_num = id
        f.close()
        print 'The number of the tracks:', self.track_num

    def readBBx(self):
        # get the gt
        self.bbx = [[] for i in xrange(self.track_num+1)]
        self.look_up = [[] for i in xrange(self.seqL + 1)]
        gt = self.gt_dir + 'gt.txt'
        f = open(gt, 'r')
        pre = -1
        for line in f.readlines():
            line = line.strip().split(',')
            if line[7] == '1':
                index = int(line[0])
                id = int(line[1])
                x, y = int(line[2]), int(line[3])
                w, h = int(line[4]), int(line[5])
                conf_score, l, vr = float(line[6]), int(line[7]), float(line[8])

                # sweep the invisible head-bbx from the training data
                if pre != id and vr == 0:
                    continue

                pre = id
                # x, y, w, h, frame_number
                if vr >= tau_vr:
                    self.bbx[id].append([x, y, w, h, index])
                    self.look_up[index].append([x, y, w, h])
        f.close()

    def distance(self, a_bbx, b_bbx):
        delta_x = a_bbx[0]-b_bbx[0]
        delta_y = a_bbx[1]-b_bbx[1]
        return sqrt(delta_x*delta_x + delta_y*delta_y)

    def positive(self, id, i):
        bbx = self.bbx[id][i]
        x, y, w, h, index = bbx
        n = len(self.bbx[id])
        if p_tag:
            head, tail = i, i

            step = head-1
            while step > 0 and abs(index - self.bbx[id][step][4]) < tau_frame:
                head = step

            step = tail+1
            while step < n and abs(index-self.bbx[id][step][4]) < tau_frame:
                tail = step
        else:
            head, tail = 0, n-1

        while tail - head:
            step = random.randint(head, tail)
            if step != i:
                return self.bbx[id][step]
        return None

    def negative(self, bbx, index):
        n = len(self.look_up[index])
        if n < 2:
            return None
        container = []
        for i in xrange(n):
            container.append([i, self.distance(bbx, self.look_up[index][i])])
        container = sorted(container, key=lambda a: a[1])
        i = random.randint(1, min(n-1, 4))
        # print i, n
        try:
            i = container[i][0]
        except IndexError:
            print i, n
            print container
        return self.look_up[index][i]

    def negative_random(self, id):
        step = random.randint(0, len(self.bbx[id])-1)
        return self.bbx[id][step]

    def readImg(self):
        self.imgs = [None]
        for i in xrange(1, self.seqL+1):
            img = load_img(self.img_dir+'%06d.jpg'%i)
            self.imgs.append(img)

    def fixBB(self, x, y, w, h, size):
        width, height = size
        w = min(w+x, width)
        h = min(h+y, height)
        x = max(x, 0)
        y = max(y, 0)
        w -= x
        h -= y
        return x, y, w, h

    def cropPatch(self, bbx, index=None):
        if index is None:
            x, y, w, h, index = bbx
        else:
            if self.negative_tag:
                x, y, w, h = bbx
            else:
                x, y, w, h, index = bbx
        img = self.imgs[index]
        x, y, w, h = self.fixBB(x, y, w, h, img.size)
        crop = img.crop([x, y, x + w, y + h])
        patch = crop.resize((224, 224), Image.ANTIALIAS)
        bbx = ToTensor()(patch)
        # bbx = bbx.view(-1, bbx.size(0), bbx.size(1), bbx.size(2))
        return bbx

    def generator(self, bbx, index=None):
        n = len(bbx)
        if n == 5:
            x, y, w, h, index = bbx
        else:
            x, y, w, h = bbx
        x, y, w = float(x), float(y), float(w),
        tmp = overlap*2/(2+overlap)
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
        if n == 5:
            ans.append(index)
        return ans

    def sampleCollection(self, show=0):
        if show:
            cv2.namedWindow('anchor', flags=0)
            cv2.namedWindow('positive', flags=0)
            cv2.namedWindow('negative', flags=0)
        f = open('Fine-tune/finetune.txt', 'a')
        print >> f, self.gt_dir
        counter = 0
        container = []
        for i in xrange(1, self.track_num+1):
            num = len(self.bbx[i])
            if num > 1:
                container.append(i)
                print >> f, i, num
                counter += 1
        self.counter += counter
        print >> f, 'Number:', counter, self.counter
        f.close()

        for i in xrange(counter):
            step = 0
            while step < 10:
                index = container[i]
                positive, negative = None, None
                while positive is None or negative is None:
                    a_index = random.randint(0, len(self.bbx[index])-1)
                    anchor = self.bbx[index][a_index]
                    positive = self.positive(index, a_index)
                    if self.negative_tag:
                        negative = self.negative(anchor, index)
                    else:
                        j = i
                        while j == i:
                            j = random.randint(0, counter - 1)
                        id = container[j]
                        negative = self.negative_random(id)

                a = self.cropPatch(anchor)
                p = self.cropPatch(positive)
                n = self.cropPatch(negative, index)
                if show:
                    img = np.asarray(a)
                    cv2.imshow('anchor', img)
                    crop1 = np.asarray(p)
                    cv2.imshow('positive', crop1)
                    crop2 = np.asarray(n)
                    cv2.imshow('negative', crop2)
                    cv2.waitKey(300)
                    # raw_input('Continue?')
                self.candidates.append([a, p, n])
                step += 1
                for j in xrange(9):
                    tmpA = self.generator(anchor)
                    tmpP = self.generator(positive)
                    tmpN = self.generator(negative)
                    a = self.cropPatch(tmpA)
                    p = self.cropPatch(tmpP)
                    n = self.cropPatch(tmpN, index)
                    self.candidates.append([a, p, n])
                    step += 1

        # free up the space
        self.imgs = []
        self.bbx = []
        self.look_up = []

    def __getitem__(self, index):
        return self.candidates[index]

    def __len__(self):
        return len(self.candidates)
