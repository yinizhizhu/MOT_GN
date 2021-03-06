# from __future__ import print_function
import numpy as np
from munkres import Munkres
import torch.nn.functional as F
import time, os, shutil
from m_global_set import edge_initial, test_gt_det, tau_conf_score, tau_threshold, gap, f_gap, show_recovering
from m_test_dataset import DatasetFromFolder
from m_mot_model import *

torch.manual_seed(123)
np.random.seed(123)


def deleteDir(del_dir):
    shutil.rmtree(del_dir)

year = 17

type = ''
t_dir = ''  # the dir of the final level
sequence_dir = ''  # the dir of the training dataset

# seqs = [2, 4, 5, 9, 10, 11, 13]  # the set of sequences
# lengths = [600, 1050, 837, 525, 654, 900, 750]  # the length of the sequence
#
# test_seqs = [1, 3, 6, 7, 8, 12, 14]
# test_lengths = [450, 1500, 1194, 500, 625, 900, 750]

seqs = [9, 11, 13]
lengths = [525, 900, 750]

test_seqs = [9, 11, 13]
test_lengths = [525, 900, 750]

tt_tag = 0  # 1 - test, 0 - train


class GN():
    def __init__(self, seq_index, tt, cuda=True):
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
        self.tt = tt
        self.missingCounter = 0
        self.sideConnection = 0

        print '     Loading the model...'
        self.loadModel()

        self.out_dir = t_dir + 'motmetrics_%s_4/'%type

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        else:
            deleteDir(self.out_dir)
            os.mkdir(self.out_dir)
        self.initOut()

    def initOut(self):
        print '     Loading Data...'
        self.train_set = DatasetFromFolder(sequence_dir, '../MOT/MOT16/train/MOT16-%02d'%self.seq_index)

        gt_training = self.out_dir + 'gt_training.txt'  # the gt of the training data
        self.copyLines(self.seq_index, 1, gt_training, self.tt)

        detection_dir = self.out_dir +'res_training_det.txt'
        res_training = self.out_dir + 'res_training.txt'  # the result of the training data
        self.createTxt(detection_dir)
        self.createTxt(res_training)
        self.copyLines(self.seq_index, 1, detection_dir, self.tt, 1)

        self.evaluation(1, self.tt, detection_dir, res_training)

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
        print '     Testing on', basic_dir, 'Length:', self.tt
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

    def loadModel(self):
        name = 'all_4'

        tail = 10
        if edge_initial == 1:
            i_name = 'Random/'
        elif edge_initial == 0:
            i_name = 'IoU/'
        self.Uphi = torch.load('Results/MOT16/%s/%s/uphi_%d.pth'%(i_name, name, tail)).to(self.device)
        self.Ephi = torch.load('Results/MOT16/%s/%s/ephi_%d.pth'%(i_name, name, tail)).to(self.device)
        self.u = torch.load('Results/MOT16/%s/%s/u_%d.pth'%(i_name, name, tail))
        self.u = self.u.to(self.device)

    def swapFC(self):
        self.cur = self.cur ^ self.nxt
        self.nxt = self.cur ^ self.nxt
        self.cur = self.cur ^ self.nxt

    def linearModel(self, out, attr1, attr2):
        # print 'I got you! *.*'
        t = attr1[-1]
        self.sideConnection += 1
        if t > f_gap:
            return
        frame = int(attr1[0])
        x1, y1, w1, h1 = float(attr1[2]), float(attr1[3]), float(attr1[4]), float(attr1[5])
        x2, y2, w2, h2 = float(attr2[2]), float(attr2[3]), float(attr2[4]), float(attr2[5])

        x_delta = (x2-x1)/t
        y_delta = (y2-y1)/t
        w_delta = (w2-w1)/t
        h_delta = (h2-h1)/2

        for i in xrange(1, t):
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
            line = ''
            for attr in attr1[:-1]:
                line += attr + ','
            if show_recovering:
                line += '1'
            else:
                line = line[:-1]
            print >> out, line
            self.bbx_counter += 1
        self.missingCounter += t-1

    def evaluation(self, head, tail, gtFile, outFile):
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
        line_con = [[], []]
        id_con = [[], []]
        id_step = 1

        step = head + self.train_set.setBuffer(head)
        while step < tail:
            # print '*********************************'
            t_gap = self.train_set.loadNext()
            step += t_gap
            # print head+step, 'F',

            u_ = self.Uphi(self.train_set.E, self.train_set.V, self.u)

            # print 'Fo'
            m = self.train_set.m
            n = self.train_set.n
            # print 'm = %d, n = %d'%(m, n)
            if n==0:
                print 'There is no detection in the rest of sequence!'
                break

            if id_step == 1:
                out = open(outFile, 'a')
                i = 0
                while i < m:
                    attrs = gtIn.readline().strip().split(',')
                    if float(attrs[6]) >= tau_conf_score:
                        attrs.append(1)
                        attrs[1] = str(id_step)
                        line = ''
                        for attr in attrs[:-1]:
                            line += attr + ','
                        if show_recovering:
                            line += '0'
                        else:
                            line = line[:-1]
                        print >> out, line
                        self.bbx_counter += 1
                        line_con[self.cur].append(attrs)
                        id_con[self.cur].append(id_step)
                        id_step += 1
                        i += 1
                out.close()

            i = 0
            while i < n:
                attrs = gtIn.readline().strip().split(',')
                if float(attrs[6]) >= tau_conf_score:
                    attrs.append(1)
                    line_con[self.nxt].append(attrs)
                    id_con[self.nxt].append(-1)
                    i += 1

            # update the edges
            # print 'T',
            ret = self.train_set.getRet()
            for edge in self.train_set.candidates:
                e, vs_index, vr_index = edge
                if ret[vs_index][vr_index] == 1.0:
                    continue
                e = e.to(self.device).view(1,-1)
                v1 = self.train_set.getMApp(1, vs_index).to(self.device)
                v2 = self.train_set.getMApp(0, vr_index, vs_index, line_con[self.cur][vs_index][-1]).to(self.device)
                e_ = self.Ephi(e, v1, v2, u_)
                self.train_set.edges[vs_index][vr_index] = e_.data.view(-1)
                tmp = F.softmax(e_)
                tmp = tmp.cpu().data.numpy()[0]
                ret[vs_index][vr_index] = float(tmp[0])

            # self.train_set.showE(outFile)

            # for j in ret:
            #     print j
            results = self.hungarian.compute(ret)

            out = open(outFile, 'a')
            look_up = set(j for j in xrange(n))
            for (i, j) in results:
                # print (i,j)
                if ret[i][j] >= tau_threshold:
                    continue
                look_up.remove(j)
                self.train_set.updateVelocity(i, j, line_con[self.cur][i][-1], False)

                id = id_con[self.cur][i]
                id_con[self.nxt][j] = id
                attr1 = line_con[self.cur][i]
                attr2 = line_con[self.nxt][j]
                # print attrs
                attr2[1] = str(id)
                if attr1[-1] > 1:
                    # for the missing detections
                    self.linearModel(out, attr1, attr2)
                line = ''
                for attr in attr2[:-1]:
                    line += attr + ','
                if show_recovering:
                    line += '0'
                else:
                    line = line[:-1]
                print >> out, line
                self.bbx_counter += 1

            for j in look_up:
                self.train_set.updateVelocity(-1, j, tag=False)

            for i in xrange(n):
                if id_con[self.nxt][i] == -1:
                    id_con[self.nxt][i] = id_step
                    attrs = line_con[self.nxt][i]
                    attrs[1] = str(id_step)
                    line = ''
                    for attr in attrs[:-1]:
                        line += attr + ','
                    if show_recovering:
                        line += '0'
                    else:
                        line = line[:-1]
                    print >> out, line
                    self.bbx_counter += 1
                    id_step += 1
            out.close()

            # For missing & Occlusion
            index = 0
            for (i, j) in results:
                while i != index:
                    attrs = line_con[self.cur][index]
                    # print '*', attrs, '*'
                    if attrs[-1] + t_gap <= gap:
                        attrs[-1] += t_gap
                        line_con[self.nxt].append(attrs)
                        id_con[self.nxt].append(id_con[self.cur][index])
                        self.train_set.moveMApp(index)
                    index += 1

                if ret[i][j] >= tau_threshold:
                    attrs = line_con[self.cur][index]
                    # print '*', attrs, '*'
                    if attrs[-1] + t_gap <= gap:
                        attrs[-1] += t_gap
                        line_con[self.nxt].append(attrs)
                        id_con[self.nxt].append(id_con[self.cur][index])
                        self.train_set.moveMApp(index)
                index += 1
            while index < m:
                attrs = line_con[self.cur][index]
                # print '*', attrs, '*'
                if attrs[-1] + t_gap <= gap:
                    attrs[-1] += t_gap
                    line_con[self.nxt].append(attrs)
                    id_con[self.nxt].append(id_con[self.cur][index])
                    self.train_set.moveMApp(index)
                index += 1

            # con = self.train_set.cleanEdge()
            # for i in xrange(len(con)-1, -1, -1):
            #     index = con[i]
            #     del line_con[self.nxt][index]
            #     del id_con[self.nxt][index]

            line_con[self.cur] = []
            id_con[self.cur] = []
            # print head+step, results
            self.train_set.swapFC()
            self.swapFC()
        gtIn.close()
        print '     The results:', id_step, self.bbx_counter

        # tra_tst = 'training sets' if head == 1 else 'validation sets'
        # out = open(outFile, 'a')
        # print >> out, tra_tst
        # out.close()

if __name__ == '__main__':
    try:
        types = ['DPM', 'SDP', 'FRCNN']
        for t in types:
            type = t
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
                tt = lengths[i]
                print 'The sequence:', seq_index, '- The length of the training data:', tt

                s_dir = f_dir + '%02d/' % seq_index
                if not os.path.exists(s_dir):
                    os.mkdir(s_dir)
                    print s_dir, 'does not exist!'

                t_dir = s_dir + '%d/' % tt
                if not os.path.exists(t_dir):
                    os.mkdir(t_dir)
                    print t_dir, 'does not exist!'

                if tt_tag:
                    seq_dir = 'MOT%d-%02d-%s' % (year, test_seqs[i], type)
                    sequence_dir = '../MOT/MOT%d/test/'%year + seq_dir
                    print ' ', sequence_dir

                    start = time.time()
                    print '     Evaluating Graph Network...'
                    gn = GN(test_seqs[i], test_lengths[i])
                else:
                    seq_dir = 'MOT%d-%02d-%s' % (year, seqs[i], type)
                    sequence_dir = '../MOT/MOT%d/train/'%year + seq_dir
                    print ' ', sequence_dir

                    start = time.time()
                    print '     Evaluating Graph Network...'
                    gn = GN(seqs[i], lengths[i])
                print '     Recover the number missing detections:', gn.missingCounter
                print '     The number of sideConnections:', gn.sideConnection
                print 'Time consuming:', (time.time()-start)/60.0
            print 'Time consuming:', (time.time()-head)/60.0
    except KeyboardInterrupt:
        print 'Time consuming:', (time.time()-start)/60.0
        print ''
        print '-'*90
        print 'Existing from training early.'
