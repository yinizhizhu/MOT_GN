import cv2
import numpy as np
from PIL import Image

label = ['', 'Pedestrian', 'Person on vehicle',
              'Car', 'Bicycle', 'Motorbike', 'Non motorized vehicle',
              'Static person', 'Distractor', 'Occluder',
              'Occluder on the ground', 'Occluder full', 'Reflection']


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


def fixBB(x, y, w, h, size):
    width, height = size
    w = min(w+x, width)
    h = min(h+y, height)
    x = max(x, 0)
    y = max(y, 0)
    w -= x
    h -= y
    return x, y, w, h


def show():
    """
    Show the clip of the sequence to judge the condition
    :return: None
    """
    basis = 'MOT16/train/'
    part = 'MOT16-02'
    part = basis + part

    # get the length of the sequence
    info = part + '/seqinfo.ini'
    f = open(info, 'r')
    f.readline()
    for line in f.readlines():
        line = line.strip().split('=')
        if line[0] == 'seqLength':
            seqL = int(line[1])
    f.close()

    # read the image
    imgs = [0]  # store the sequence
    imgDir = part + '/img1/'
    for i in xrange(1, seqL + 1):
        img = load_img(imgDir + '%06d.jpg' % i)
        imgs.append(img)

    # get the gt
    gt = part + '/show.txt'
    cv2.namedWindow('view', flags=0)
    cv2.namedWindow('crop', flags=0)
    while True:
        f = open(gt, 'r')
        for line in f.readlines():
            line = line.strip().split(',')
            index = int(line[0])
            id = int(line[1])
            x = int(line[2])
            y = int(line[3])
            w = int(line[4])
            h = int(line[5])
            l = int(line[7])
            vr = float(line[8])
            img = imgs[index]
            x, y, w, h = fixBB(x, y, w, h, img.size)
            crop = imgs[index].crop([x, y, x + w, y + h])
            img = np.asarray(img)
            crop = np.asarray(crop)
            # print line
            print part, index, seqL, '%06d' % index, label[l], vr
            # print w, h
            # print len(crop[0]), len(crop)
            cv2.imshow('crop', crop)
            cv2.imshow('view', img)
            cv2.waitKey(34)
        raw_input('Continue?')
        f.close()

try:
    show()
except KeyboardInterrupt:
    print 'Existing from showing.'