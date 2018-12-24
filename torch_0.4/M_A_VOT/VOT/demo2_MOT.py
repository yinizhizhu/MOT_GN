# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import glob, cv2, torch
import numpy as np
from os.path import realpath, dirname, join

from net import SiamRPNvot, SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect

# load VOT net
# net = SiamRPNvot()
# net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNVOT.model')))

#load BIG net
net = SiamRPNBIG()
net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNBIG.model')))

net.eval().cuda()

# image and init box
image_files = sorted(glob.glob('./demo2_MOT/*.jpg'))

x, y, w, h = 1487,71,53,145
# init_rbox = [x, y, x+w, y, x+w, y+h, x, y+h]
# [cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)
[cx, cy, w, h] = [x+w/2, y+h/2, w, h]

# tracker init
target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
im = cv2.imread(image_files[0])  # HxWxC
state = SiamRPN_init(im, target_pos, target_sz, net)

# tracking and visualization
toc = 0
step = 1
for f, image_file in enumerate(image_files):
    im = cv2.imread(image_file)
    tic = cv2.getTickCount()
    state = SiamRPN_track(state, im)  # track
    toc += cv2.getTickCount()-tic
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
    score = state['score']
    res = [int(l) for l in res]
    cv2.rectangle(im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)

    cv2.putText(im, '%.3f'%state['score'],(res[0], res[1]+21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3,(255,255,255), 1, cv2.LINE_AA)

    cv2.imshow('SiamRPN', im)
    cv2.imwrite('demo2_seq/%06d.jpg'%step, im)
    cv2.waitKey(1)
    step += 1

print('Tracking Speed {:.1f}fps'.format((len(image_files)-1)/(toc/cv2.getTickFrequency())))


# 'ffmpeg -y -r 25 -i demo2_seq/%6d.jpg -ar 22050 -b 50000 -r 24 -vtag DIVX -f avi out.mp4'