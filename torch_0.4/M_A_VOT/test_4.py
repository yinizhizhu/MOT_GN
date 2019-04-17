# from __future__ import print_function
import numpy as np
from munkres import Munkres
import torch.nn.functional as F
import time, os, shutil, torch
from global_set import edge_initial, test_gt_det, tau_threshold, decay, \
    gap, f_gap, show_recovering, decay_dir, recover_dir, app_dir, u_update, u_dir#, tau_conf_score, decay
from mot_model import appearance
from test_dataset_a import ADatasetFromFolder
from test_dataset_m import MDatasetFromFolder
import cv2
from VOT.net import SiamRPNvot
from VOT.run_SiamRPN import SiamRPN_init, SiamRPN_track
from VOT.utils import cxy_wh_2_rect

torch.manual_seed(123)
np.random.seed(123)


def deleteDir(del_dir):
    shutil.rmtree(del_dir)


year = 17

type = ''
t_dir = ''  # the dir of the final level
sequence_dir = ''  # the dir of the training dataset

# 7 - training with all the sequences for final model
# 4 - training with four sequences for selecting best parameters
# 0 - training with all the sequences but only first 80% of sequence for training, and the rest for validation
train_set_num = 0

if train_set_num == 0:
    seqs = [2, 4, 5, 9, 10, 11, 13]  # the set of sequences
    lengths = [600, 1050, 837, 525, 654, 900, 750]  # the length of the sequence

    test_seqs = [1, 3, 6, 7, 8, 12, 14]
    test_lengths = [450, 1500, 1194, 500, 625, 900, 750]
else:
    seqs = [9, 11, 13]
    lengths = [525, 900, 750]

    test_seqs = [9, 11, 13]
    test_lengths = [525, 900, 750]

# seqs = [11]
# lengths = [900]

tt_tag = 0  # 1 - test, 0 - train

tau_conf_score = 0.0

vot_conf_score = 0.99

# decay = 0.0

class GN():
    def __init__(self, seq_index, begin, end, a, cuda=True):
        '''
        Evaluating with the MotMetrics
        :param seq_index: the number of the sequence
        :param tt: train_test
        :param length: the number of frames which is used for training
        :param cuda: True - GPU, False - CPU
        '''
        self.bbx_counter = 0
        self.seq_index = seq_index
        self.hungarian = Munkres()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.begin = begin
        self.end = end
        self.alpha = a
        self.missingCounter = 0
        self.sideConnection = 0

        print '     Loading the model...'
        self.loadAModel()
        self.loadMModel()
        self.loadVOT()

        if train_set_num == 4:
            # self.out_dir = t_dir + 'motmetrics_%s_4_%.1f%s_%.2f%s%s_tau_tdir1.0/'%(type,
            #                                                                        self.alpha,
            #                                                                        decay_dir, decay,
            #                                                                        recover_dir, u_dir)

            # 1.9 - decay_tag > 0, 1.1~1.8 - decay_tag > 1

            self.out_dir = t_dir + 'motmetrics_%s_4_%.1f%s_%.2f%s%s_vc_%.2f/'%(type,
                                                                                   self.alpha,
                                                                                   decay_dir, decay,
                                                                                   recover_dir, u_dir, vot_conf_score)

            # self.out_dir = t_dir + 'motmetrics_%s_4_%.1f%s_%.2f%s%s_vc_%.2f_exp/'%(type,
            #                                                                        self.alpha,
            #                                                                        decay_dir, decay,
            #                                                                        recover_dir, u_dir, vot_conf_score)
        else:
            self.out_dir = t_dir + 'motmetrics_%s_4_%.1f%s_%.2f%s%s_vc_%.2f_dseq/'%(type,
                                                                                   self.alpha,
                                                                                   decay_dir, decay,
                                                                                   recover_dir, u_dir, vot_conf_score)

        print '		', self.out_dir
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        else:
            deleteDir(self.out_dir)
            os.mkdir(self.out_dir)
        self.initOut()

    def initOut(self):
        print '     Loading Data...'
        self.a_train_set = ADatasetFromFolder(sequence_dir, '../MOT/MOT16/train/MOT16-%02d'%self.seq_index, tau_conf_score)
        self.m_train_set = MDatasetFromFolder(sequence_dir, '../MOT/MOT16/train/MOT16-%02d'%self.seq_index, tau_conf_score)

        gt_training = self.out_dir + 'gt_training.txt'  # the gt of the training data
        self.copyLines(self.seq_index, self.begin, gt_training, self.end)

        detection_dir = self.out_dir +'res_training_det.txt'
        res_training = self.out_dir + 'res_training.txt'  # the result of the training data
        self.createTxt(detection_dir)
        self.createTxt(res_training)
        self.copyLines(self.seq_index, self.begin, detection_dir, self.end, 1)

        self.out = open(res_training, 'w')
        self.evaluation(self.begin, self.end, detection_dir)
        self.out.close()

    def getSeqL(self, info):
        # get the length of the sequence
        f = open(info, 'r')
        f.readline()
        for line in f.readlines():
            line = line.strip().split('=')
            if line[0] == 'seqLength':
                seqL = int(line[1])
        f.close()
        return seqL

    def copyLines(self, seq, head, gt_seq, tail=-1, tag=0):
        '''
        Copy the groun truth within [head, head+num]
        :param seq: the number of the sequence
        :param head: the head frame number
        :param tail: the number the clipped sequence
        :param gt_seq: the dir of the output file
        :return: None
        '''
        if tt_tag:
            basic_dir = '../MOT/MOT%d/test/MOT%d-%02d-%s/' % (year, year, seq, type)
        else:
            basic_dir = '../MOT/MOT%d/train/MOT%d-%02d-%s/' % (year, year, seq, type)
        print '     Testing on', basic_dir, 'Length:', self.end - self.begin+1
        seqL = tail if tail != -1 else self.getSeqL(basic_dir + 'seqinfo.ini')

        det_dir = 'gt/gt_det.txt' if test_gt_det else 'det/det.txt'
        seq_dir = basic_dir + ('gt/gt.txt' if tag == 0 else det_dir)
        inStream = open(seq_dir, 'r')

        outStream = open(gt_seq, 'w')
        for line in inStream.readlines():
            line = line.strip()
            attrs = line.split(',')
            f_num = int(attrs[0])
            if f_num >= head and f_num <= seqL:
                print >> outStream, line
        outStream.close()

        inStream.close()
        return seqL

    def createTxt(self, out_file):
        f = open(out_file, 'w')
        f.close()

    def swapFC(self):
        self.cur = self.cur ^ self.nxt
        self.nxt = self.cur ^ self.nxt
        self.cur = self.cur ^ self.nxt

    def loadAModel(self):
        from mot_model import uphi, ephi, vphi
        name = '%s_%d'%(app_dir, train_set_num)
        if edge_initial == 0:
            model_dir = 'App2_bb'
            i_name = 'IoU'
        elif edge_initial == 1:
            model_dir = 'App2_bb'
            i_name = 'Random'
        tail = 10 if train_set_num == 4 else 13
        self.AUphi = torch.load('../%s/Results/MOT16/%s/%s/uphi_%02d.pth'%(model_dir, i_name, name, tail)).to(self.device)
        self.AVphi = torch.load('../%s/Results/MOT16/%s/%s/vphi_%02d.pth'%(model_dir,i_name, name, tail)).to(self.device)
        self.AEphi1 = torch.load('../%s/Results/MOT16/%s/%s/ephi1_%02d.pth'%(model_dir,i_name, name, tail)).to(self.device)
        self.AEphi2 = torch.load('../%s/Results/MOT16/%s/%s/ephi2_%02d.pth'%(model_dir,i_name, name, tail)).to(self.device)
        self.Au = torch.load('../%s/Results/MOT16/%s/%s/u_%02d.pth'%(model_dir,i_name, name, tail))
        self.Au = self.Au.to(self.device)

    def loadMModel(self):
        from m_mot_model import uphi, ephi
        name = 'all_%d'%train_set_num
        if edge_initial == 0:
            model_dir = 'Motion1_bb'
            i_name = 'IoU'
        elif edge_initial == 1:
            model_dir = 'Motion1_bb'
            i_name = 'Random'
        tail = 10 if train_set_num == 4 else 13
        self.MUphi = torch.load('../%s/Results/MOT16/%s/%s/uphi_%d.pth'%(model_dir,i_name, name, tail)).to(self.device)
        self.MEphi = torch.load('../%s/Results/MOT16/%s/%s/ephi_%d.pth'%(model_dir,i_name, name, tail)).to(self.device)
        self.Mu = torch.load('../%s/Results/MOT16/%s/%s/u_%d.pth'%(model_dir,i_name, name, tail))
        self.Mu = self.Mu.to(self.device)

    def loadVOT(self):
        self.net = SiamRPNvot()
        self.net.load_state_dict(torch.load('VOT/SiamRPNVOT.model'))
        self.net = self.net.to(self.device)

    def outputLine(self, attr, src):
        """
        Output the tracking result into text file
        :param attr: The tracked detection
        :param src: The source code which call this function
        :return: None
        """
        if attr[1] == '-1':
            print src, '-', attr
        line = ''
        for attr in attr[:-1]:
            line += attr + ','
        if show_recovering:
            if src[0] == 'd':
                line += '1'
            elif src[0] == 'R':
                line += '2'
            else:
                line += '0'
        else:
            line = line[:-1]
        self.bbx_counter += 1
        print >> self.out, line

    def linearModel(self, attr1, attr2, src):
        # print 'I got you! *.*'
        # print attr1, attr2
        frame, frame2 = int(attr1[0]), int(attr2[0])
        t = frame2 - frame
        if t > 1:
            self.sideConnection += 1
        if t > f_gap:
            self.outputLine(attr2, src+'LinearModel1')
            return
        x1, y1, w1, h1 = float(attr1[2]), float(attr1[3]), float(attr1[4]), float(attr1[5])
        x2, y2, w2, h2 = float(attr2[2]), float(attr2[3]), float(attr2[4]), float(attr2[5])

        x_delta = (x2-x1)/t
        y_delta = (y2-y1)/t
        w_delta = (w2-w1)/t
        h_delta = (h2-h1)/t

        for i in xrange(1, t+1):
            frame += 1
            x1 += x_delta
            y1 += y_delta
            w1 += w_delta
            h1 += h_delta
            attr1[0] = str(frame)
            attr1[2] = str(x1)
            attr1[3] = str(y1)
            attr1[4] = str(w1)
            attr1[5] = str(h1)
            if i == 1 or i == t:
                self.outputLine(attr1, src+'LinearModel2')
            else:
                self.outputLine(attr1, 'Recovering')
        self.missingCounter += t - 1

    def checker(self, bbxes, len):
        """
        Finding the detections which do not have large IOU with their neighbor detections
        :param bbxes: Container of boudning boxes
        :param len: The number of bounding boxes
        :return: The results of judging whether the detection can tracked with VOT or not
        """
        ans = [1 for i in xrange(len)]
        for i in xrange(len-1):
            for j in xrange(i+1, len):
                if ans[j]:
                    ratio = self.a_train_set.IOU(bbxes[i], bbxes[j])
                    if ratio[1] >= 0.5 and ratio[2] >= 0.5:
                        ans[i] = 0
                        ans[j] = 0
                    if ratio[1] >= 0.8:
                        ans[i] = 0
                    if ratio[2] >= 0.8:
                        ans[j] = 0
        return ans

    def vot(self, bbx):
        """
        Tracking with VOT (DaSiamRPN)
        :param bbx: The initial bounding box
        :return: The prediction and the confidence score
        """
        x, y, w, h, frame = bbx
        [cx, cy, w, h] = [x+w/2, y+h/2, w, h]

        # tracker init
        target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
        cur_img = cv2.imread(self.a_train_set.img_dir + '%06d.jpg' % frame)
        state = SiamRPN_init(cur_img, target_pos, target_sz, self.net)

        state = SiamRPN_track(state, self.img)  # track
        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        score = state['score']

        bbx = [int(l) for l in res]
        return bbx, score

    def tracking_vot(self, gap, update=True):
        if update == False:
            if self.a_train_set.n != len(self.a_train_set.bbx[self.a_train_set.f_step]):
                print 'The length is wrong!'
            self.a_train_set.tags_index = [1 for i in xrange(self.a_train_set.n)]

        indexes = []
        tracked_pred = []
        frame = self.a_train_set.f_step - self.a_train_set.gap
        step = len(self.a_train_set.bbx[frame])
        ans = self.checker(self.a_train_set.bbx[frame], step)
        self.img = cv2.imread(self.a_train_set.img_dir + '%06d.jpg' % (frame+gap))
        step -= 1
        while step >= 0:
            if ans[step]:
                tmp_bbx = self.m_train_set.bbx[frame][step]
                if tmp_bbx[2]*tmp_bbx[3] >= 0.001:
                    bbx = self.a_train_set.bbx[frame][step]
                    pred, score = self.vot(bbx)

                    # if self.a_train_set.f_step > 198:
                    #     print self.id_con[self.cur][step], pred, score

                    if score >= vot_conf_score:

                        # if self.a_train_set.f_step > 198:
                        #     print 'Success'

                        tracked_pred.append([step, pred])
                        if update:
                            self.a_train_set.updateApp(pred, step, gap)
                            self.m_train_set.updateMotion(bbx, pred, step, gap)
                        else:
                            self.a_train_set.delAppCur(step)
                            self.m_train_set.delMotionCur(step)
                            index = self.a_train_set.findDetection(pred)
                            if index >= 0:
                                indexes.append(index)
            step -= 1

        indexes = sorted(indexes, reverse=True)

        # print 'Tracking_vot', '*'*30
        # print indexes
        # print len(self.line_con[self.nxt]), self.line_con[self.nxt]
        # print len(self.id_con[self.nxt]), self.id_con[self.nxt]

        if update == False:
            # print 'tags_index:', self.a_train_set.tags_index
            # print len(indexes), len(self.line_con[self.nxt]),
            # print indexes
            for index in indexes:
                del self.line_con[self.nxt][index]
                del self.id_con[self.nxt][index]
                self.a_train_set.delAppNxt(index)
                self.m_train_set.delMotionNxt(index)

        return tracked_pred

    def transfer(self, line, pred, frame):
        attr = []
        attr.append(str(frame))
        attr.append(line[1])
        for p in pred:
            attr.append(str(p))
        for p in line[6:]:
            attr.append(p)
        attr[-1] = 1
        return attr

    def doTracking(self, a_t_gap):
        # @We first do tracking with DaSiamRPN.
        frame = self.a_train_set.f_step - a_t_gap
        for gap in xrange(1, a_t_gap):
            # print '\n Container:'
            # print self.line_con[self.cur]
            # print self.line_con[self.nxt]
            tracked_pred = self.tracking_vot(gap, True)
            if len(tracked_pred):
                for index, pred in tracked_pred:
                    attr1 = self.line_con[self.cur][index]
                    # print 'Attr1:', attr1
                    attr2 = self.transfer(attr1, pred, frame + gap)
                    self.linearModel(attr1, attr2, 'doTracking1')
                    # print 'Frame:', frame+gap
                    # print 'Attr2:', attr2
                    # print 'Attr1_after:', attr1
                    self.line_con[self.cur][index] = attr2
                    # print self.line_con[self.cur][index]
                    # raw_input('Continue?')

        tracked_pred = self.tracking_vot(a_t_gap, False)
        if len(tracked_pred):
            # print 'DoTracking', '-'*30
            # print tracked_pred

            for p in tracked_pred:
                index, pred = p

                # print '*'*36
                # print index
                # print len(self.line_con[self.cur]), self.line_con[self.cur]

                attr1 = self.line_con[self.cur][index]
                attr2 = self.transfer(attr1, pred, frame + a_t_gap)
                self.linearModel(attr1, attr2, 'doTracking2')
                self.line_backup.append(attr2)
                self.id_backup.append(self.id_con[self.cur][index])
                del self.line_con[self.cur][index]
                del self.id_con[self.cur][index]

    def addBackup(self):
        # print '*'*30
        # print 'Cur:', self.id_con[self.cur]
        # print 'Nxt:', self.id_con[self.nxt]
        n = len(self.line_backup)
        for i in xrange(n):
            line = self.line_backup[i]
            id = self.id_backup[i]

            bbx = [int(line[2]), int(line[3]), int(line[4]), int(line[5])]

            self.a_train_set.addApp(bbx)
            self.m_train_set.addMotion(bbx)

            self.line_con[self.nxt].append(line)
            self.id_con[self.nxt].append(id)

        # print 'Cur2:', self.id_con[self.cur]
        # print 'Nxt2:', self.id_con[self.nxt]

        self.line_backup = []
        self.id_backup = []

    def evaluation(self, head, tail, gtFile):
        '''
        Evaluation on dets
        :param head: the head frame number
        :param tail: the tail frame number
        :param gtFile: the ground truth file name
        :param outFile: the name of output file
        :return: None
        '''
        gtIn = open(gtFile, 'r')
        self.cur, self.nxt = 0, 1
        self.img = None
        self.line_con = [[], []]
        self.id_con = [[], []]
        self.line_backup = []
        self.id_backup = []
        self.id_step = 1

        a_step = head + self.a_train_set.setBuffer(head)
        m_step = head + self.m_train_set.setBuffer(head)
        print 'Starting from the frame:', a_step
        if a_step != m_step:
            print 'Something is wrong!'
            print 'a_step =', a_step, ', m_step =', m_step
            raw_input('Continue?')

        while a_step <= tail:
            # print '*********************************'

            a_t_gap = self.a_train_set.loadNext()
            m_t_gap = self.m_train_set.loadNext()

            if a_t_gap != m_t_gap:
                print 'Something is wrong!'
                print 'a_t_gap =', a_t_gap, ', m_t_gap =', m_t_gap
                raw_input('Continue?')
            a_step += a_t_gap
            m_step += m_step

            print a_step,
            if a_step % 1000 == 0:
                print

            if a_step > tail:
                break

            self.loadCur(self.a_train_set.m, gtIn)

            # print head+step, 'F',
            a_m, m_m = self.a_train_set.m, self.m_train_set.m
            a_n, m_n = self.a_train_set.n, self.m_train_set.n
            if a_m != m_m or a_n != m_n:
                print 'Something is wrong!'
                print 'a_m = %d, m_m = %d' % (a_m, m_m), ', a_n = %d, m_n = %d' % (a_n, m_n)
                raw_input('Continue?')

            self.loadNxt(gtIn, a_n)

            # show_p = 198
            # if a_step > show_p:
            #     print
            #     print 'Before tracking, loading the bounding boxes in current & next frame:'
            #     print 'Current:'
            #     for i in xrange(a_m):
            #         print i, self.line_con[self.cur][i], self.id_con[self.cur][i], self.a_train_set.bbx[self.a_train_set.f_step-a_t_gap][i]
            #     print 'Next:'
            #     for i in xrange(a_n):
            #         print i, self.line_con[self.nxt][i], self.id_con[self.nxt][i], self.a_train_set.bbx[self.a_train_set.f_step][i]

            self.doTracking(a_t_gap)

            self.a_train_set.counter()
            self.m_train_set.counter()

            a_m, m_m = self.a_train_set.m, self.m_train_set.m
            a_n, m_n = self.a_train_set.n, self.m_train_set.n
            if a_m != m_m or a_n != m_n:
                print 'Something is wrong!'
                print 'a_m = %d, m_m = %d' % (a_m, m_m), ', a_n = %d, m_n = %d' % (a_n, m_n)
                raw_input('Continue?')

            if a_n == 0 or a_m == 0:
                # print 'There is no detection in the next frame!'
                # print '*'*30
                # print a_m, len(self.line_con[self.cur])
                # print self.line_con[self.cur]

                for index in xrange(a_n):
                    attrs = self.line_con[self.nxt][index]
                    attrs[1] = str(self.id_step)
                    self.outputLine(attrs, 'Output')
                    self.id_con[self.nxt][index] = self.id_step
                    self.id_step += 1

                for index in xrange(a_m):
                    attrs = self.line_con[self.cur][index]
                    # print '*', attrs, '*'
                    if attrs[-1] + a_t_gap <= gap:
                        attrs[-1] += a_t_gap
                        self.line_con[self.nxt].append(attrs)
                        self.id_con[self.nxt].append(self.id_con[self.cur][index])
                        self.a_train_set.moveApp(index)
                        self.m_train_set.moveMotion(index)

                self.addBackup()
                self.line_con[self.cur] = []
                self.id_con[self.cur] = []
                # print head+step, results
                self.a_train_set.swapFC()
                self.m_train_set.swapFC()
                self.swapFC()
                continue

            self.a_train_set.loadPre()
            self.m_train_set.loadPre()

            m_u_ = self.MUphi(self.m_train_set.E, self.m_train_set.V, self.Mu)

            # update the edges
            # print 'T',
            candidates = []
            E_CON, V_CON = [], []
            for edge in self.a_train_set.candidates:
                e, vs_index, vr_index = edge
                e = e.view(1, -1).to(self.device)
                vs = self.a_train_set.getApp(1, vs_index)
                vr = self.a_train_set.getApp(0, vr_index)

                e1 = self.AEphi1(e, vs, vr, self.Au)
                vr1 = self.AVphi(e1, vs, vr, self.Au)
                candidates.append((e1, vs, vr1, vs_index, vr_index))
                E_CON.append(e1)
                V_CON.append(vs)
                V_CON.append(vr1)

            E = self.a_train_set.aggregate(E_CON).view(1, -1)
            V = self.a_train_set.aggregate(V_CON).view(1, -1)
            u1 = self.AUphi(E, V, self.Au)

            ret = self.a_train_set.getRet()
            decay_tag = [0 for i in xrange(a_m)]
            for i in xrange(a_m):
                for j in xrange(a_n):
                    if ret[i][j] == 0:
                        decay_tag[i] += 1

            # if a_step > show_p:
            #     print '     Ret:'
            #     tmp = 0
            #     for p in ret:
            #         print tmp, p
            #         tmp += 1

            for i in xrange(len(self.a_train_set.candidates)):
                e1, vs, vr1, a_vs_index, a_vr_index = candidates[i]
                m_e, m_vs_index, m_vr_index = self.m_train_set.candidates[i]
                if a_vs_index != m_vs_index or a_vr_index != m_vr_index:
                    print 'Something is wrong!'
                    print 'a_vs_index = %d, m_vs_index = %d'%(a_vs_index, m_vs_index)
                    print 'a_vr_index = %d, m_vr_index = %d'%(a_vr_index, m_vr_index)
                    raw_input('Continue?')
                if ret[a_vs_index][a_vr_index] == tau_threshold:
                    continue

                e2 = self.AEphi2(e1, vs, vr1, u1)
                self.a_train_set.edges[a_vs_index][a_vr_index] = e1.data.view(-1)

                a_tmp = F.softmax(e2)
                a_tmp = a_tmp.cpu().data.numpy()[0]

                m_e = m_e.to(self.device).view(1,-1)
                m_v1 = self.m_train_set.getMotion(1, m_vs_index)
                m_v2 = self.m_train_set.getMotion(0, m_vr_index, m_vs_index)
                m_e_ = self.MEphi(m_e, m_v1, m_v2, m_u_)
                self.m_train_set.edges[m_vs_index][m_vr_index] = m_e_.data.view(-1)
                m_tmp = F.softmax(m_e_)
                m_tmp = m_tmp.cpu().data.numpy()[0]

                t = self.line_con[self.cur][a_vs_index][-1]
                # A = float(a_tmp[0]) * pow(decay, t-1)
                # M = float(m_tmp[0]) * pow(decay, t-1)
                if decay_tag[a_vs_index] > 0:
                    try:
                        A = min(float(a_tmp[0]) * pow(decay, t + a_t_gap -2), 0.999999999999)
                        M = min(float(m_tmp[0]) * pow(decay, t + a_t_gap -2), 0.999999999999)
                        # A = min(float(a_tmp[0]) * pow(decay, t + a_t_gap -2), 1.0)
                        # M = min(float(m_tmp[0]) * pow(decay, t + a_t_gap -2), 1.0)
                    except OverflowError:
                        print 'OverflowError!'
                        A = float(a_tmp[0])
                        M = float(m_tmp[0])
                else:
                    A = float(a_tmp[0])
                    M = float(m_tmp[0])
                ret[a_vs_index][a_vr_index] = A*self.alpha + M*(1-self.alpha)

            # self.a_train_set.showE(outFile)
            # self.m_train_set.showE(outFile)

            # if a_step > show_p:
            #     print
            #     print 'Current:'
            #     for i in xrange(a_m):
            #         print i, self.line_con[self.cur][i], self.id_con[self.cur][i]
            #     print 'Next:'
            #     for i in xrange(a_n):
            #         print i, self.line_con[self.nxt][i], self.id_con[self.nxt][i]
            #     print '     Ret:'
            #     tmp = 0
            #     for p in ret:
            #         print tmp, p
            #         tmp += 1

            results = self.tracking(ret, a_n, m_u_, u1)

            # if a_step > show_p:
            #     for (i, j) in results:
            #         print (i, j)
            #         print self.line_con[self.nxt][j], self.id_con[self.nxt][j]
            #     raw_input('Continue?')

            self.output(a_n)

            self.miss_occlu(results, a_t_gap, ret, a_m)

            self.addBackup()

            self.line_con[self.cur] = []
            self.id_con[self.cur] = []
            # print head+step, results
            self.a_train_set.swapFC()
            self.m_train_set.swapFC()
            self.swapFC()

        gtIn.close()
        print '     The results:', self.id_step, self.bbx_counter

        # tra_tst = 'training sets' if head == 1 else 'validation sets'
        # out = open(outFile, 'a')
        # print >> out, tra_tst
        # out.close()

    def loadCur(self, a_m, gtIn):
        if self.id_step == 1:
            i = 0
            while i < a_m:
                attrs = gtIn.readline().strip().split(',')
                if float(attrs[6]) >= tau_conf_score:
                    attrs.append(1)
                    attrs[1] = str(self.id_step)
                    self.outputLine(attrs, 'LoadCur')
                    self.line_con[self.cur].append(attrs)
                    self.id_con[self.cur].append(self.id_step)
                    self.id_step += 1
                    i += 1

    def loadNxt(self, gtIn, a_n):
        i = 0
        while i < a_n:
            attrs = gtIn.readline().strip().split(',')
            if float(attrs[6]) >= tau_conf_score:
                attrs.append(1)
                self.line_con[self.nxt].append(attrs)
                self.id_con[self.nxt].append(-1)
                i += 1

    def tracking(self, ret, a_n, m_u_, u1):
        # for j in ret:
        #     print j
        results = self.hungarian.compute(ret)

        look_up = set(j for j in xrange(a_n))
        nxt = self.a_train_set.nxt
        for (i, j) in results:
            # print (i,j)
            if ret[i][j] >= tau_threshold:
                continue
            e1 = self.a_train_set.edges[i][j].view(1, -1).to(self.device)
            vs = self.a_train_set.getApp(1, i)
            vr = self.a_train_set.getApp(0, j)

            vr1 = self.AVphi(e1, vs, vr, self.Au)
            self.a_train_set.detections[nxt][j][0] = vr1.data

            look_up.remove(j)
            self.m_train_set.updateVelocity(i, j, False)

            id = self.id_con[self.cur][i]
            self.id_con[self.nxt][j] = id
            attr1 = self.line_con[self.cur][i]
            attr2 = self.line_con[self.nxt][j]
            # print attrs
            attr2[1] = str(id)
            if attr2[1] =='-1':
                print 'In main process:', attr2[1], self.id_con[self.cur][i]
            self.linearModel(attr1, attr2, 'Main')

        if u_update:
            m_u_ = torch.clamp(m_u_, max=1.0, min=-1.0)  # make sure that the global variable not that big
            u1 = torch.clamp(u1, max=1.0, min=-1.0)
            self.Mu = m_u_.data
            self.Au = u1.data

        for j in look_up:
            self.m_train_set.updateVelocity(-1, j, tag=False)
        return results

    def output(self, a_n):
        for i in xrange(a_n):
            if self.id_con[self.nxt][i] == -1:
                self.id_con[self.nxt][i] = self.id_step
                attrs = self.line_con[self.nxt][i]
                attrs[1] = str(self.id_step)
                self.outputLine(attrs, 'Output')
                self.id_step += 1

    def miss_occlu(self, results, a_t_gap, ret, a_m):
        # For missing & Occlusion
        index = 0
        for (i, j) in results:
            while i != index:
                attrs = self.line_con[self.cur][index]
                # print '*', attrs, '*'
                if attrs[-1] + a_t_gap <= gap:
                    attrs[-1] += a_t_gap
                    self.line_con[self.nxt].append(attrs)
                    self.id_con[self.nxt].append(self.id_con[self.cur][index])
                    self.a_train_set.moveApp(index)
                    self.m_train_set.moveMotion(index)
                index += 1
            if ret[i][j] >= tau_threshold:
                attrs = self.line_con[self.cur][index]
                # print '*', attrs, '*'
                if attrs[-1] + a_t_gap <= gap:
                    attrs[-1] += a_t_gap
                    self.line_con[self.nxt].append(attrs)
                    self.id_con[self.nxt].append(self.id_con[self.cur][index])
                    self.a_train_set.moveApp(index)
                    self.m_train_set.moveMotion(index)
            index += 1
        while index < a_m:
            attrs = self.line_con[self.cur][index]
            # print '*', attrs, '*'
            if attrs[-1] + a_t_gap <= gap:
                attrs[-1] += a_t_gap
                self.line_con[self.nxt].append(attrs)
                self.id_con[self.nxt].append(self.id_con[self.cur][index])
                self.a_train_set.moveApp(index)
                self.m_train_set.moveMotion(index)
            index += 1

start_a = time.time()
if __name__ == '__main__':
    try:
        for a in xrange(91, 100):
            vot_conf_score = a/100.0
        # for i in xrange(1):
            if not os.path.exists('Results/'):
                os.mkdir('Results/')

            types = [['DPM0', -0.6], ['SDP', 0.5], ['FRCNN', 0.5]]
            # types = [['DPM0', -0.6]]
            # types = [['SDP', 0.5]]
            # types = [['FRCNN', 0.5]]
            for t in types:
                type, tau_conf_score = t
                head = time.time()
                f_dir = 'Results/MOT%s/' % year
                if not os.path.exists(f_dir):
                    os.mkdir(f_dir)

                if edge_initial == 1:
                    f_dir += 'Random/'
                elif edge_initial == 0:
                    f_dir += 'IoU/'

                if not os.path.exists(f_dir):
                    os.mkdir(f_dir)
                    print f_dir, 'does not exist!'

                for i in xrange(len(seqs)):
                    seq_index = seqs[i]
                    begin = 1
                    end = lengths[i]
                    if train_set_num == 0:
                        begin = int(end*0.8)

                    print 'The sequence:', seq_index, '- The length of the training data:', end - begin+1

                    s_dir = f_dir + '%02d/' % seq_index
                    if not os.path.exists(s_dir):
                        os.mkdir(s_dir)
                        print s_dir, 'does not exist!'

                    t_dir = s_dir + '%d/' % end
                    if not os.path.exists(t_dir):
                        os.mkdir(t_dir)
                        print t_dir, 'does not exist!'

                    if tt_tag:
                        seq_dir = 'MOT%d-%02d-%s' % (year, test_seqs[i], type)
                        sequence_dir = '../MOT/MOT%d/test/'%year + seq_dir
                        print ' ', sequence_dir

                        start = time.time()
                        print '     Evaluating Graph Network...'
                        gn = GN(test_seqs[i], begin, test_lengths[i], 0.3)
                    else:
                        seq_dir = 'MOT%d-%02d-%s' % (year, seqs[i], type)
                        sequence_dir = '../MOT/MOT%d/train/'%year + seq_dir
                        print ' ', sequence_dir

                        start = time.time()
                        print '     Evaluating Graph Network...'
                        gn = GN(seqs[i], begin, end, 0.3)
                        print '     Recover the number missing detections:', gn.missingCounter
                        print '     The number of sideConnections:', gn.sideConnection
                    print 'Time consuming:', (time.time()-start)/60.0
                print 'Time consuming:', (time.time()-head)/60.0
            print 'Total time consuming:', (time.time()-start_a)/60.0
    except KeyboardInterrupt:
        print ''
        print '-'*90
        print 'Existing from training early.'
        print 'Time consuming:', (time.time()-start_a)/60.0