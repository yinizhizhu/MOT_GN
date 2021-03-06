# from __future__ import print_function
import numpy as np
from mot_model import *
from munkres import Munkres
import torch.nn.functional as F
import time, os, shutil, commands
from global_set import edge_initial, test_gt_det, tau_conf_score
from test_dataset import DatasetFromFolder

torch.manual_seed(123)
np.random.seed(123)


def deleteDir(del_dir):
    shutil.rmtree(del_dir)

gap = 25
year = 16
t_dir = ''  # the dir of the final level
# metrics_dir = ''  # the dir of the motmetrics
sequence_dir = ''  # the dir of the training dataset
seqs = [2, 4, 5, 9, 10, 11, 13]  # the set of sequences
lengths = [600, 1050, 837, 525, 654, 900, 750]  # the length of the sequence


# def cleanText():
#     out = open('res_dir_list.txt', 'w')
#     out.close()
# cleanText()


class GN():
    def __init__(self, seq_index, tt, length, cuda=True):
        '''
        Evaluating with the MotMetrics
        :param seq_index: the number of the sequence
        :param tt: train_test
        :param length: the number of frames which is used for training
        :param cuda: True - GPU, False - CPU
        '''

        # top index: 0 - correct matching, 1 - false matching, second index: 0 - min, 1 - max, 2 - total, 3 - counter
        self.ctau = [[1.0, 0.0, 0.0, 0] for i in xrange(2)]  # get the threshold for matching cost

        self.seq_index = seq_index
        self.hungarian = Munkres()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.tt = tt
        self.length = length
        self.missingCounter = 0

        print '     Loading the model...'
        self.loadModel()

        self.out_dir = t_dir + 'motmetrics/'
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        else:
            deleteDir(self.out_dir)
            os.mkdir(self.out_dir)
        self.initOut()

    def initOut(self):
        start = time.time()
        print '     Loading Data...'
        print '     Training'
        self.train_set = DatasetFromFolder(sequence_dir)

        gt_training = self.out_dir + 'gt_training.txt'  # the gt of the training data
        self.copyLines(self.seq_index, 1, gt_training, self.tt)

        detection_dir = self.out_dir +'res_training_det.txt'
        res_training = self.out_dir + 'res_training.txt'  # the result of the training data
        self.createTxt(detection_dir)
        self.createTxt(res_training)
        self.copyLines(self.seq_index, 1, detection_dir, self.tt, 1)

        # Evaluating on the training data
        # motmetrics = open(metrics_dir, 'a')
        # print >> motmetrics, '*'*30, self.tt, '*'*30
        # print >> motmetrics, 'Training'
        self.evaluation(1, self.tt, detection_dir, res_training)
        print '     Time consuming:', (time.time()-start)/60.0
        # cmd = 'python3 evaluation.py %s %s'%(gt_training, res_training)
        # (status, output) = commands.getstatusoutput(cmd)
        # print >> motmetrics, output
        # print >> motmetrics, 'The time consuming:{}\n\n'.format((time.time()-start)/60)
        # motmetrics.close()

        if self.tt < self.length:
            # Evaluating on the validation data
            start = time.time()
            print '     Validation'

            # The distant sequence
            head = self.length - self.tt + 1
            tail = self.length

            # The sequence nearby
            # head = self.tt
            # tail = 2*self.tt-1

            gt_valiadation = self.out_dir + 'gt_validation.txt'  # the gt of the validation data
            self.copyLines(self.seq_index, head, gt_valiadation, tail)

            detection_dir = self.out_dir + 'res_validation_det.txt'
            res_validation = self.out_dir + 'res_validation.txt'  # the result of the validation data
            self.createTxt(detection_dir)
            self.createTxt(res_validation)
            self.copyLines(self.seq_index, head, detection_dir, tail, 1)

            # motmetrics = open(metrics_dir, 'a')
            # print >> motmetrics, 'Validation'
            self.evaluation(head, tail, detection_dir, res_validation)
            print '     Time consuming:', (time.time()-start)/60.0
            # cmd = 'python3 evaluation.py %s %s'%(gt_valiadation, res_validation)
            # (status, output) = commands.getstatusoutput(cmd)
            # print >> motmetrics, output
            # print >> motmetrics, 'The time consuming:{}\n\n'.format((time.time()-start)/60)
            # motmetrics.close()
        else:
            # Evaluating on the validation data
            for seq in seqs:
                if seq == self.seq_index:
                    continue
                print '     %02d_Validation'%seq
                start = time.time()
                seq_dir = 'MOT16/train/MOT%d-%02d' % (year, seq)
                self.train_set = DatasetFromFolder(seq_dir)
                gt_seq = self.out_dir + 'gt_%02d.txt' % seq
                seqL = self.copyLines(seq, 1, gt_seq)

                detection_dir = self.out_dir + 'res_%02d_det.txt' % seq
                c_validation = self.out_dir + 'res_%02d.txt' % seq
                self.createTxt(detection_dir)
                self.createTxt(c_validation)
                self.copyLines(seq, 1, detection_dir, tag=1)

                # motmetrics = open(metrics_dir, 'a')
                # print >> motmetrics, '%02d_validation'%seq
                self.evaluation(1, seqL, detection_dir, c_validation)
                print '     Time consuming:', (time.time()-start)/60.0
                # cmd = 'python3 evaluation.py %s %s'%(gt_seq, c_validation)
                # (status, output) = commands.getstatusoutput(cmd)
                # print >> motmetrics, output
                # print >> motmetrics, 'The time consuming:{}\n\n'.format((time.time()-start)/60)
                # motmetrics.close()

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
        basic_dir = 'MOT16/train/MOT%d-%02d/' % (year, seq)
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
        self.Uphi = torch.load(t_dir+'uphi.pth').to(self.device)
        self.Ephi = torch.load(t_dir+'ephi.pth').to(self.device)
        self.u = torch.load(t_dir+'u.pth')
        self.u = self.u.to(self.device)

    def swapFC(self):
        self.cur = self.cur ^ self.nxt
        self.nxt = self.cur ^ self.nxt
        self.cur = self.cur ^ self.nxt

    def showCTau(self):
        for i in xrange(2):
            print '     Min:', self.ctau[i][0], 'Max:', self.ctau[i][1],
            if self.ctau[i][3]:
                print 'Mean:', self.ctau[i][2]/self.ctau[i][3]
            else:
                print 'Total:', self.ctau[i][2], 'Counter:', self.ctau[i][3]

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
        id_step = 1

        step = head + self.train_set.setBuffer(head)
        while step < tail:
            step += self.train_set.loadNext()
            # print head+step, 'F',

            u_ = self.Uphi(self.train_set.E, self.train_set.V, self.u)

            # print 'Fo'
            m = self.train_set.m
            n = self.train_set.n
            if n==0:
                print 'There is no detection in the rest of sequence!'
                break

            if id_step == 1:
                i = 0
                while i < m:
                    attrs = gtIn.readline().strip().split(',')
                    if float(attrs[6]) >= tau_conf_score:
                        line_con[self.cur].append(attrs)
                        id_step += 1
                        i += 1

            i = 0
            while i < n:
                attrs = gtIn.readline().strip().split(',')
                if float(attrs[6]) >= tau_conf_score:
                    attrs.append(1)
                    line_con[self.nxt].append(attrs)
                    i += 1

            # update the edges
            # print 'T',
            ret = [[0.0 for i in xrange(n)] for j in xrange(m)]
            for edge in self.train_set.candidates:
                e, vs_index, vr_index = edge
                e = e.to(self.device).view(1,-1)
                v1 = self.train_set.getApp(1, vs_index)
                v2 = self.train_set.getApp(0, vr_index)
                e_ = self.Ephi(e, v1, v2, u_)
                self.train_set.edges[vs_index][vr_index] = e_.data.view(-1)
                tmp = F.softmax(e_)
                tmp = tmp.cpu().data.numpy()[0]
                ret[vs_index][vr_index] = float(tmp[0])

            # self.train_set.showE(outFile)

            # for j in ret:
            #     print j
            for i in xrange(m):
                a_attrs = line_con[self.cur][i]
                for j in xrange(n):
                    index = 1
                    cost = ret[i][j]
                    if a_attrs[1] == line_con[self.nxt][j][1]:
                        index = 0
                    self.ctau[index][0] = min(self.ctau[index][0], cost)
                    self.ctau[index][1] = max(self.ctau[index][1], cost)
                    self.ctau[index][2] += cost
                    self.ctau[index][3] += 1

            line_con[self.cur] = []
            # print head+step, results
            self.train_set.swapFC()
            self.swapFC()
        gtIn.close()

        # tra_tst = 'training sets' if head == 1 else 'validation sets'
        # out = open(outFile, 'a')
        # print >> out, tra_tst
        # out.close()
        self.showCTau()

if __name__ == '__main__':
    try:
        f_dir = 'Results/MOT%s/' % year
        if not os.path.exists(f_dir):
            os.mkdir(f_dir)

        if edge_initial == 1:
            f_dir += 'Random/'
        elif edge_initial == 0:
            f_dir += 'IoU/'

        if not os.path.exists(f_dir):
            print f_dir, 'does not exist!'

        type_dir = 'IoU' if edge_initial == 0 else 'Random'
        metric_dir = 'Results/MOT%d/MotMetrics_%s/' % (year, type_dir)
        if not os.path.exists(metric_dir):
            os.mkdir(metric_dir)

        for i in xrange(7):
            seq_index = seqs[i]
            tts = []
            # tts = [tt for tt in xrange(100, 600, 100)]
            length = lengths[i]
            tts.append(length)

            # metrics_dir = metric_dir+'%02d.txt'%seq_index
            # motmetrics = open(metrics_dir, 'w')
            # motmetrics.close()
            for tt in tts:
                tag = 1
                if tt*2 > length:
                    if tt == length:
                        tag = 0
                    else:
                        continue
                print 'The sequence:', seq_index, '- The length of the training data:', tt

                s_dir = f_dir + '%02d/' % seq_index
                if not os.path.exists(s_dir):
                    print s_dir, 'does not exist!'

                t_dir = s_dir + '%d/' % tt
                if not os.path.exists(t_dir):
                    print t_dir, 'does not exist!'

                seq_dir = 'MOT%d-%02d' % (year, seq_index)
                sequence_dir = 'MOT16/train/' + seq_dir
                print ' ', sequence_dir

                start = time.time()
                print '     Evaluating Graph Network...'
                gn = GN(seq_index, tt, length)
                print '     Recover the number missing detections:', gn.missingCounter
                print 'Time consuming:', (time.time()-start)/60.0
    except KeyboardInterrupt:
        print 'Time consuming:', (time.time()-start)/60
        print ''
        print '-'*90
        print 'Existing from training early.'
