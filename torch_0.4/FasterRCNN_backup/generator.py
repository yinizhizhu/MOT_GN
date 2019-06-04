from __future__ import absolute_import
from __future__ import division

import _init_paths
import random, os, time, cv2, shutil, torch, pprint
from PIL import Image
import numpy as np

from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

from easydict import EasyDict as edict

output = True


def initPath():
    if os.path.exists('../Datasets/') and output:
        shutil.rmtree('../Datasets/')
    if not os.path.exists('../Datasets/'):
        os.mkdir('../Datasets/')
    if not os.path.exists('../Datasets/gt/'):
        os.mkdir('../Datasets/gt/')
    if not os.path.exists('../Datasets/det/'):
        os.mkdir('../Datasets/det/')

initPath()
gt_dir = '../Datasets/gt/'
det_dir = '../Datasets/det/'


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    # print im_shape, im_size_max, im_size_min

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


class G:
    def __init__(self):
        self.container = dict()
        self.cf_head = 1
        self.cf_tail = 10
        self.iou_head = 1
        self.iou_tail = 10
        for x in xrange(self.cf_head, self.cf_tail):
            cf = x/10.0
            for y in xrange(self.iou_head, self.iou_tail):
                iou = y/10.0
                outName = 'FasterRCNN/info_%.1f_%.1f.txt'%(cf, iou)
                self.container[outName] = edict()

                self.container[outName].all_gt_acc_p = 0.0
                self.container[outName].all_gt_acc_n = 0.0
                self.container[outName].all_det_acc_p = 0.0
                self.container[outName].all_det_acc_n = 0.0

                out = open(outName, 'w')
                out.close()

        seqs = [2, 4, 5, 9, 10, 11, 13]  # the set of sequences
        types = [['DPM0', -0.6], ['SDP', 0.5], ['FRCNN', 0.5]]

        # seqs = [9]
        # types = [['DPM0', -0.6]]

        start_a = time.time()

        self.device = torch.device("cuda")

        self.loadFasterRCNN()

        self.all_gt_p = 0.0
        self.all_gt_n = 0.0

        self.all_det_p = 0.0
        self.all_det_n = 0.0

        for t in types:
            self.type, conf_score = t
            head = time.time()
            for i in xrange(len(seqs)):
                self.count_gt_p = 0
                self.count_gt_n = 0
                for x in xrange(self.cf_head, self.cf_tail):
                    cf = x/10.0
                    for y in xrange(self.iou_head, self.iou_tail):
                        iou = y/10.0
                        outName = 'FasterRCNN/info_%.1f_%.1f.txt'%(cf, iou)
                        self.container[outName].gt_acc_p = 0
                        self.container[outName].gt_acc_n = 0
                        self.container[outName].det_acc_p = 0
                        self.container[outName].det_acc_n = 0

                self.count_gt_p = 0
                self.count_gt_n = 0
                self.count_det_p = 0
                self.count_det_n = 0

                self.seq = seqs[i]

                start = time.time()
                print '     Generating datasets...'

                self.img_dir = '../MOT/MOT16/train/MOT16-%02d/img1/'%self.seq
                print self.img_dir

                seq_dir = 'MOT17-%02d-%s' % (self.seq, self.type)
                part = '../MOT/MOT17/train/' + seq_dir
                self.dir = part
                self.gt_dir = part + '/gt/'
                self.det_dir = part + '/det/'
                self.tau_conf_score = conf_score
                print self.gt_dir, self.det_dir

                self.getSeqL()
                self.readBBx_gt()
                self.readBBx_det()
                self.g()

                print '\n\t\tSeq time consuming:', (time.time()-start)/60.0
            print '\tType time consuming:', (time.time()-head)/60.0
        print 'Total time consuming:', (time.time()-start_a)/60.0

        out = open('FasterRCNN/results.txt', 'w')
        self.results = sorted(self.results, key=lambda a:a[0])
        print >> out, self.all_det_p+self.all_det_n, self.all_det_p, self.all_det_n
        print >> out, self.all_gt_p+self.all_gt_n, self.all_gt_p, self.all_gt_n
        for p in self.results:
            for t in p:
                print >> out, t,
            print >> out, ''
        out.close()

    def getSeqL(self):
        # get the length of the sequence
        info = self.dir+'/seqinfo.ini'
        f = open(info, 'r')
        f.readline()
        for line in f.readlines():
            line = line.strip().split('=')
            if line[0] == 'seqLength':
                self.seqL = int(line[1])
            elif line[0] == 'imWidth':
                self.width = int(line[1])
            elif line[0] == 'imHeight':
                self.height = int(line[1])
        f.close()
        print '     The length of the sequence:', self.seqL

    def readBBx_gt(self):
        # get the gt
        self.gts = [[] for i in xrange(self.seqL + 1)]
        gt = self.gt_dir + 'gt.txt'
        f = open(gt, 'r')
        for line in f.readlines():
            line = line.strip().split(',')
            label = line[7]
            # pedestrian, person on vehicle, static person, distractor, reflection
            if label == '1' or label == '2' or label == '7' or label == '8' or label == '12':
                frame = int(line[0])
                id = int(line[1])
                x, y = int(line[2]), int(line[3])
                w, h = int(line[4]), int(line[5])
                conf_score, l, vr = float(line[6]), int(line[7]), float(line[8])

                # sweep the invisible head-bbx from the training data
                if vr > 0:
                    self.gts[frame].append([x, y, w, h])
        f.close()

    def readBBx_det(self):
        # get the det
        self.dets = [[] for i in xrange(self.seqL + 1)]
        det = self.det_dir + 'det.txt'
        f = open(det, 'r')
        for line in f.readlines():
            line = line.strip().split(',')
            frame = int(line[0])
            x, y = int(float(line[2])), int(float(line[3]))
            w, h = int(float(line[4])), int(float(line[5]))
            conf_score = float(line[6])
            if conf_score >= self.tau_conf_score:
                self.dets[frame].append([x, y, w, h])
        f.close()

    def IOU(self, Reframe, GTframe):
        """
        Compute the Intersection of Union
        :param Reframe: x, y, w, h
        :param GTframe: x, y, w, h
        :return: Ratio
        """
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

    def g(self):
        gap = int(self.seqL/30)
        # pos & type & seq & frame & counter
        # 'pos_%s_%d_%d_%d.jpg' % (self.type, self.seq, frame, self.count_p)
        self.ps = []
        for frame in xrange(1, self.seqL+1, gap):
            print frame,
            self.img = cv2.imread(self.img_dir + '%06d.jpg' % frame)

            # Generating bounding box in Datasets/gt/
            for bbx in self.gts[frame]:
                # Generating the positive samples
                if self.saveCrop(bbx, gt_dir+'pos_%s_%d_%d_%d.jpg'%(self.type,
                                                                 self.seq, frame, self.count_gt_p), 0):
                    self.count_gt_p += 1

                overlap = 0.6 + random.randint(0, 3)/10.0
                x, y, w, h = bbx
                x, y, w = float(x), float(y), float(w)
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
                tmp = [int(x), int(y), int(w), h]
                if self.saveCrop(tmp, gt_dir+'pos_%s_%d_%d_%d.jpg'%(self.type,
                                                                 self.seq, frame, self.count_gt_p), 0):
                    self.count_gt_p += 1

                # Generating the negative samples
                step = 0
                overlap = 0.1 + random.randint(0, 3)/10.0
                while step < 100:
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
                    neg = [int(x), int(y), int(w), h]
                    tag = True
                    for p in self.gts[frame]:
                        if self.IOU(p, neg) > overlap:
                            tag = False
                            break
                    if tag:
                        if self.saveCrop(neg, gt_dir+'neg_%s_%d_%d_%d.jpg'%(self.type,
                                                                         self.seq, frame, self.count_gt_n), 1):
                            self.count_gt_n += 1
                        break
                    step += 1

            # Generating bounding box in Datasets/det
            for bbx in self.dets[frame]:
                tag = False
                for tmp in self.gts[frame]:
                    if self.IOU(bbx, tmp) >= 0.5:
                        tag = True
                        break
                if tag:
                    if self.saveCrop(bbx, det_dir+'pos_%s_%d_%d_%d.jpg'%(self.type,
                                                                      self.seq, frame, self.count_det_p), 2):
                        self.count_det_p += 1
                else:
                    if self.saveCrop(bbx, det_dir+'neg_%s_%d_%d_%d.jpg'%(self.type,
                                                                      self.seq, frame, self.count_det_n), 3):
                        self.count_det_n += 1

        self.all_gt_p += self.count_gt_p
        self.all_gt_n += self.count_gt_n

        self.all_det_p += self.count_det_p
        self.all_det_n += self.count_det_n
        self.results = []
        for x in xrange(self.cf_head, self.cf_tail):
            cf = x/10.0
            for y in xrange(self.iou_head, self.iou_tail):
                iou = y/10.0
                outName = 'FasterRCNN/info_%.1f_%.1f.txt'%(cf, iou)
                self.container[outName].all_gt_acc_p += self.container[outName].gt_acc_p
                self.container[outName].all_gt_acc_n += self.container[outName].gt_acc_n

                self.container[outName].all_det_acc_p += self.container[outName].det_acc_p
                self.container[outName].all_det_acc_n += self.container[outName].det_acc_n

                out = open(outName, 'a')

                print >> out, 'type:\t%s\tseq:\t%d\tdet:\t%d\tpos:\t%d\tneg:\t%d'%(self.type, self.seq,
                                                                                   self.count_det_p+self.count_det_n,
                                                                                   self.count_det_p, self.count_det_n)
                print >> out, 'type:\t%s\tseq:\t%d\tgt:\t%d\tpos:\t%d\tneg:\t%d'%(self.type, self.seq,
                                                                                   self.count_gt_p+self.count_gt_n,
                                                                                   self.count_gt_p, self.count_gt_n)

                all_gt_acc = (self.container[outName].all_gt_acc_n+self.container[outName].all_gt_acc_p)*1.0/(self.all_gt_n+self.all_gt_p)
                cur_gt_acc = (self.container[outName].gt_acc_n+self.container[outName].gt_acc_p)*1.0/(self.count_gt_n+self.count_gt_p)
                print >> out, 'type:\t%s\tseq:\t%d\tall_gt_acc:\t%.3f\tcur_gt_acc:\t%.3f'\
                              %(self.type, self.seq, all_gt_acc, cur_gt_acc)

                all_det_acc = (self.container[outName].all_det_acc_n+self.container[outName].all_det_acc_p)*1.0/(self.all_det_n+self.all_det_p)
                cur_det_acc = (self.container[outName].det_acc_n+self.container[outName].det_acc_p)*1.0/(self.count_det_n+self.count_det_p)
                print >> out, 'type:\t%s\tseq:\t%d\tall_det_acc:\t%.3f\tcur_det_acc:\t%.3f'\
                              %(self.type, self.seq, all_det_acc, cur_det_acc)

                if self.all_gt_n > 0:
                    all_gt_n_acc = self.container[outName].all_gt_acc_n*1.0/self.all_gt_n
                else:
                    all_gt_n_acc = -1
                if self.all_gt_p > 0:
                    all_gt_p_acc = self.container[outName].all_gt_acc_p*1.0/self.all_gt_p
                else:
                    all_gt_p_acc = -1
                print >> out, 'all_gt_n_acc:\t%.3f\tall_gt_p_acc:\t%.3f'%(all_gt_n_acc, all_gt_p_acc)

                if self.all_det_n > 0:
                    all_det_n_acc = self.container[outName].all_det_acc_n*1.0/self.all_det_n
                else:
                    all_det_n_acc = -1
                if self.all_det_p > 0:
                    all_det_p_acc = self.container[outName].all_det_acc_p*1.0/self.all_det_p
                else:
                    all_det_p_acc = -1
                print >> out, 'all_det_n_acc:\t%.3f\tall_det_p_acc:\t%.3f'%(all_det_n_acc, all_det_p_acc)

                if self.count_gt_n:
                    cur_gt_n_acc = self.container[outName].gt_acc_n*1.0/self.count_gt_n
                else:
                    cur_gt_n_acc = -1
                if self.count_gt_p:
                    cur_gt_p_acc = self.container[outName].gt_acc_p*1.0/self.count_gt_p
                else:
                    cur_gt_p_acc = -1
                print >> out, 'cur_gt_n_acc:\t%.3f\tcur_gt_p_acc:\t%.3f'%(cur_gt_n_acc, cur_gt_p_acc)

                if self.count_det_n:
                    cur_det_n_acc = self.container[outName].det_acc_n*1.0/self.count_det_n
                else:
                    cur_det_n_acc = -1
                if self.count_det_p:
                    cur_det_p_acc = self.container[outName].det_acc_p*1.0/self.count_det_p
                else:
                    cur_det_p_acc = -1
                print >> out, 'cur_det_n_acc:\t%.3f\tcur_det_p_acc:\t%.3f'%(cur_det_n_acc, cur_det_p_acc)
                print >> out, ''
                out.close()
                self.results.append([all_det_acc, all_det_n_acc, all_det_p_acc,
                                     outName,
                                     all_gt_acc, all_gt_n_acc, all_gt_p_acc])

    def saveCrop(self, bbx, save_dir, tag):
        """
        :param bbx: bounding box
        :param save_dir:
        :param tag: 0 - gt_pos, 1 - gt_neg, 2 - det_pos, 3 - det_neg
        :return:
        """
        x, y, w, h = bbx
        w = min(w+x, self.width)
        h = min(h+y, self.height)
        x = max(x, 0)
        y = max(y, 0)
        w -= x
        h -= y
        return self.detect([x, y, w, h], save_dir, tag)

    def loadFasterRCNN(self):
        pprint.pprint(cfg)
        np.random.seed(cfg.RNG_SEED)
        load_name = 'data/pretrained_model/faster_rcnn_1_7_10021.pth'
        self.pascal_classes = np.asarray(['__background__',
                                     'aeroplane', 'bicycle', 'bird', 'boat',
                                     'bottle', 'bus', 'car', 'cat', 'chair',
                                     'cow', 'diningtable', 'dog', 'horse',
                                     'motorbike', 'person', 'pottedplant',
                                     'sheep', 'sofa', 'train', 'tvmonitor'])
        self.fasterRCNN = resnet(self.pascal_classes, 101, pretrained=True)
        self.fasterRCNN.create_architecture()
        checkpoint = torch.load(load_name)
        self.fasterRCNN.load_state_dict(checkpoint['model'])
        self.fasterRCNN = self.fasterRCNN.to(self.device)
        self.fasterRCNN.eval()

    def detect(self, bbx, save_dir, tag):
        with torch.no_grad():
            thresh = 0.05

            im_data = torch.FloatTensor(1).to(self.device)
            im_info = torch.FloatTensor(1).to(self.device)
            num_boxes = torch.LongTensor(1).to(self.device)
            gt_boxes = torch.FloatTensor(1).to(self.device)

            x, y, w, h = [int(p) for p in bbx]
            x = max(x, 0)
            y = max(y, 0)
            im = self.img[y:(y+h), x:(x+w)]
            # print ' (x=%d, y=%d), %d * %d, (%d, %d) - cropsize: %d * %d' % (x, y, w, h, x+w, y+h, im.shape[1], im.shape[0])
            w, h = im.shape[1], im.shape[0]
            refine_bbx = [0, 0, w, h]
            if w*h == 0:
                print 'What? %d * %d' %(w, h)
                # raw_input('Continue?')
                return False

            blobs, im_scales = _get_image_blob(im)
            assert len(im_scales) == 1, "Only single-image batch implemented"
            im_blob = blobs
            im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

            im_data_pt = torch.from_numpy(im_blob)
            im_data_pt = im_data_pt.permute(0, 3, 1, 2)
            im_info_pt = torch.from_numpy(im_info_np)

            im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.data.resize_(1, 1, 5).zero_()
            num_boxes.data.resize_(1).zero_()

            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = self.fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and stdev
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).to(self.device)

                    box_deltas = box_deltas.view(1, -1, 4 * len(self.pascal_classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                _ = torch.from_numpy(np.tile(boxes, (1, scores.shape[1])))
                pred_boxes = _.to(self.device)

            pred_boxes /= im_scales[0]

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()

            j = 15
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det

            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
                cls_dets = cls_dets[keep.view(-1).long()]

                dets = cls_dets.cpu().numpy()

                out = output
                for x in xrange(self.cf_head, self.cf_tail):
                    cf = x/10.0
                    for y in xrange(self.iou_head, self.iou_tail):
                        iou = y/10.0
                        outName = 'FasterRCNN/info_%.1f_%.1f.txt'%(cf, iou)

                        step = 0
                        for i in range(dets.shape[0]):
                            if dets[i, -1] > cf:
                                x1, y1, w1, h1 = dets[i][:4]
                                det = [x1, y1, w1-x1, h1-y1]
                                ratio = self.IOU(det, refine_bbx)
                                # print det, refine_bbx, ratio
                                if ratio > iou:  # IOU between prediction and detection should not be limited
                                    step += 1

                        # print cls_dets
                        if out:
                            im = vis_detections(im, self.pascal_classes[j], dets, cf)
                            # print save_dir, save_dir.split('/')[3][:3], step
                            cv2.imwrite(save_dir, im)
                            out = False

                        if tag == 0:
                            if step:
                                self.container[outName].gt_acc_p += 1
                        elif tag == 1:
                            if step == 0:
                                self.container[outName].gt_acc_n += 1
                        elif tag == 2:
                            if step:
                                self.container[outName].det_acc_p += 1
                        else:
                            if step == 0:
                                self.container[outName].det_acc_n += 1
                return True
            return False

G()

# year = 17
#
# seqs = [2, 4, 5, 9, 10, 11, 13]  # the set of sequences
# lengths = [600, 1050, 837, 525, 654, 900, 750]  # the length of the sequence
#
# start_a = time.time()
# if __name__ == '__main__':
#     try:
#         types = [['DPM0', -0.6], ['SDP', 0.5], ['FRCNN', 0.5]]
#         # types = [['DPM0', -0.6]]
#         # types = [['SDP', 0.5]]
#         # types = [['FRCNN', 0.5]]
#         for t in types:
#             type, conf_score = types
#             head = time.time()
#             for i in xrange(len(seqs)):
#                 seq_dir = 'MOT%d-%02d-%s' % (year, seqs[i], type)
#                 sequence_dir = '../MOT/MOT%d/train/'%year + seq_dir
#                 print ' ', sequence_dir
#
#                 start = time.time()
#                 print '     Generating datasets...'
#                 gn = G(sequence_dir, conf_score)
#                 print 'Time consuming:', (time.time()-start)/60.0
#
#             print 'Time consuming:', (time.time()-head)/60.0
#     except KeyboardInterrupt:
#         print ''
#         print '-'*90
#         print 'Existing from training early.'
#         print 'Time consuming:', (time.time()-start_a)/60.0