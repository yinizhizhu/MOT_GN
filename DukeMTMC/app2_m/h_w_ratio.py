import cv2, torch, h5py, os, shutil
from PIL import Image
from global_set import SEQLEN#, tau_conf_score
import numpy as np


font = cv2.FONT_HERSHEY_SIMPLEX
color = [(255,0,0),(0,255,0),(0,0,255)]


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class hwRatio():
    def __init__(self, camera, tau, out_dir, cuda=True):
        self.seq_index = camera

        self.width, self.height = 1920.0, 1080.0

        self.device = torch.device("cuda" if cuda else "cpu")
        self.tau_conf_score = tau

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

        self.maxRatio = 1.0
        self.out_dir = out_dir
        self.readBBx_det()
        print 'In camera_%d:'%camera, self.maxRatio

    def fixBB(self, x, y, w, h):
        w = min(w+x, self.width)
        h = min(h+y, self.height)
        x = max(x, 0)
        y = max(y, 0)
        w -= x
        h -= y
        return x, y, w, h

    def fixBBx(self, x1, y1, x2, y2):
        width = x2 - x1
        height = y2 - y1
        x1 -= width/12
        y1 -= height/12
        return int(x1), int(y1), int(width*7/6), int(height*7/6)

    def getBBx(self, pose):
        x1, x2, y1, y2 = 10000.0, 0.0, 10000.0, 0.0
        conf_score = []
        for i in xrange(0, 54, 3):
            x, y, conf = float(pose[2 + i]*self.width), float(pose[3 + i]*self.height), float(pose[4 + i])
            # print x, y, conf
            if conf == 0.0:
                continue
            conf_score.append(conf)
            if x < x1:
                x1 = x
            if x > x2:
                x2 = x
            if y < y1:
                y1 = y
            if y > y2:
                y2 = y

        conf_score = sum(conf_score)/len(conf_score)
        if conf_score >= self.tau_conf_score:
            x1, y1, w, h = self.fixBBx(x1, y1, x2, y2)
            if w == 0 or h == 0:
                return None
            ratio = h * 1.0 / w
            if ratio >= 1.0:
                return [int(x1), int(y1), int(w), int(h)]
        return None

    def readBBx_det(self):
        # get the det
        heads = [0, 5543, 3607, 27244, 31182, 1, 22402, 18968, 46766]
        test = [227541, 356648]

        detection_dir = '/media/lee/DATA/DukeMTMC/detections/openpose/camera%d.mat'%self.seq_index
        det = h5py.File(detection_dir)
        det = det['detections']
        det = np.transpose(det)

        head = test[0] - heads[self.seq_index] + 1 + 5000
        tail = head + 1000 - 1

        imgs = [None for frame in xrange(head, tail+1)]

        for bbx in det:
            frame = int(bbx[1])
            bbx = self.getBBx(bbx)
            if bbx is not None and frame >= head and frame <= tail:
                x, y, w, h = bbx
                ratio = h * 1.0 / w

                if ratio >= 4.0:
                    if imgs[frame-head] is None:
                        imgs[frame-head] = cv2.imread(self.img_dir + '%06d.jpg' % frame)
                    cv2.rectangle(imgs[frame-head], (x, y), (x+w, y+h), color[0], 2)
                    cv2.putText(imgs[frame-head], '%.4f'%(ratio), (x, y + 21), font, 0.6, color[2], 2, cv2.LINE_AA)
                    cv2.putText(imgs[frame-head], '%d*%d'%(w, h), (x, y + h - 21), font, 0.6, color[1], 2, cv2.LINE_AA)

                self.maxRatio = max(self.maxRatio, ratio)

        step = 1
        for img in imgs:
            if img is not None:
                cv2.imwrite(self.out_dir + '%06d.jpg'%step, img)
            step += 1


def deleteDir(del_dir):
    shutil.rmtree(del_dir)


basic_dir = 'H_W_Ratio/'
if not os.path.exists(basic_dir):
    os.mkdir(basic_dir)

for i in xrange(1, 9):
    seq_dir = basic_dir + 'camera%d/'%i
    if os.path.exists(seq_dir):
        deleteDir(seq_dir)
    os.mkdir(seq_dir)
    hwRatio(i, 0.55, seq_dir)