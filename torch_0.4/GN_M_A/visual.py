# from __future__ import print_function
import numpy as np
from munkres import Munkres
import torch.nn.functional as F
import time, os, shutil, torch, cv2
from global_set import edge_initial, test_gt_det, tau_conf_score, tau_threshold, gap, f_gap, show_recovering
from mot_model import appearance
from test_dataset import ADatasetFromFolder
from m_test_dataset import MDatasetFromFolder

torch.manual_seed(123)
np.random.seed(123)
font = cv2.FONT_HERSHEY_SIMPLEX


def readImg(filename):
    """"
    Color image loaded by OpenCV is in BGR mode, but Matplotlib displays in RGB mode.
    cv2.imread(path, style)
        1 - cv2.IMREAD_COLOR
        0 - cv2.IMREAD_GRAYSCALE
        -1 - cv2.IMREAD_UNCHANGED
    """
    # print filename
    img = cv2.imread(filename, 1)
    return img


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

seqs = [4]
lengths = [1050]

test_seqs = [3]
test_lengths = [1500]

tt_tag = 1  # 1 - test, 0 - train

ALPHA_TAG = None  # 1 - a*A+(1-a)*M, 2 - a*A+M, 3 - A+a*M

class GN():
    def __init__(self, seq_index, tt, a, cuda=True):
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
        self.alpha = a
        self.missingCounter = 0
        self.sideConnection = 0

        print '     Loading the model...'
        self.loadAModel()
        self.loadMModel()

        self.out_dir = t_dir + 'motmetrics_%s_show/'%(type)
        print '		', self.out_dir
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        else:
            deleteDir(self.out_dir)
            os.mkdir(self.out_dir)
        self.initWin()
        self.initOut()

    def initWin(self):
        self.color = [(255,0,0),(0,255,0),(0,0,255)]
        self.img_dir = '../MOT/MOT16/test/MOT16-%02d/img1/'%self.seq_index
        self.pre_win = 'Show/Previous'

        self.cur_win = 'Show/Current'

    def initOut(self):
        print '     Loading Data...'
        self.a_train_set = ADatasetFromFolder(sequence_dir, '../MOT/MOT16/test/MOT16-%02d'%self.seq_index)
        self.m_train_set = MDatasetFromFolder(sequence_dir, '../MOT/MOT16/test/MOT16-%02d'%self.seq_index)

        # gt_training = self.out_dir + 'gt_training.txt'  # the gt of the training data
        # self.copyLines(self.seq_index, 1, gt_training, self.tt)

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

    def loadAModel(self):
        from mot_model import uphi, ephi
        if edge_initial == 0:
            model_dir = 'MOT'
            name = 'all_det_ft'
            i_name = 'IoU'
        elif edge_initial == 1:
            model_dir = 'Appearance'
            name = 'all_7_CE'
            i_name = 'Random'
        tail = 10
        self.AUphi = torch.load('../%s/Results/MOT16/%s/%s/uphi_%02d.pth'%(model_dir, i_name, name, tail)).to(self.device)
        self.AEphi = torch.load('../%s/Results/MOT16/%s/%s/ephi_%02d.pth'%(model_dir,i_name, name, tail)).to(self.device)
        self.Au = torch.load('../%s/Results/MOT16/%s/%s/u_%02d.pth'%(model_dir,i_name, name, tail))
        self.Au = self.Au.to(self.device)

    def loadMModel(self):
        from m_mot_model import uphi, ephi
        if edge_initial == 0:
            model_dir = 'MOT_Motion'
            name = 'all_v2_4'
            i_name = 'IoU'
        elif edge_initial == 1:
            model_dir = 'Motion'
            name = 'all_7_CE'
            i_name = 'Random'
        tail = 10
        self.MUphi = torch.load('../%s/Results/MOT16/%s/%s/uphi_%d.pth'%(model_dir,i_name, name, tail)).to(self.device)
        self.MEphi = torch.load('../%s/Results/MOT16/%s/%s/ephi_%d.pth'%(model_dir,i_name, name, tail)).to(self.device)
        self.Mu = torch.load('../%s/Results/MOT16/%s/%s/u_%d.pth'%(model_dir,i_name, name, tail))
        self.Mu = self.Mu.to(self.device)

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

        imgs = [None, None]     # 0 - previous img, 1 - current img
        going_tag = 0           # 0 - frame by frame, 1 - goto going_f

        line_con = [[], []]
        id_con = [[], []]
        id_step = 1

        a_step = head + self.a_train_set.setBuffer(head)
        m_step = head + self.m_train_set.setBuffer(head)
        if a_step != m_step:
            print 'Something is wrong!'
            print 'a_step =', a_step, ', m_step =', m_step
            raw_input('Continue?')

        imgs[self.cur] = readImg(self.img_dir + '%06d.jpg'%a_step)
        going_f = a_step
        while a_step < tail:
            # print '*********************************'
            if going_f <= a_step:
                going_tag = 0

            a_t_gap = self.a_train_set.loadNext()
            m_t_gap = self.m_train_set.loadNext()
            if a_t_gap != m_t_gap:
                print 'Something is wrong!'
                print 'a_t_gap =', a_t_gap, ', m_t_gap =', m_t_gap
                raw_input('Continue?')
            a_step += a_t_gap
            m_step += m_step
            # print head+step, 'F',

            a_u_ = self.AUphi(self.a_train_set.E, self.a_train_set.V, self.Au)
            m_u_ = self.MUphi(self.m_train_set.E, self.m_train_set.V, self.Mu)

            # print 'Fo'
            a_m = self.a_train_set.m
            a_n = self.a_train_set.n
            m_m = self.m_train_set.m
            m_n = self.m_train_set.n

            if a_m != m_m or a_n != m_n:
                print 'Something is wrong!'
                print 'a_m = %d, m_m = %d'%(a_m, m_m), ', a_n = %d, m_n = %d'%(a_n, m_n)
                raw_input('Continue?')
            # print 'm = %d, n = %d'%(m, n)
            if a_n==0:
                print 'There is no detection in the rest of sequence!'
                break

            if id_step == 1:
                out = open(outFile, 'a')
                i = 0
                while i < a_m:
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

                        # draw the rectangle
                        x, y = int(float(attrs[2])), int(float(attrs[3]))
                        w, h = int(float(attrs[4])), int(float(attrs[5]))
                        cv2.rectangle(imgs[self.cur], (x, y), (x+w, y+h), self.color[0], 2)
                        cv2.putText(imgs[self.cur], attrs[1]+'_B', (x+3, y+15), font, 0.6, self.color[0], 2, cv2.LINE_AA)

                        line_con[self.cur].append(attrs)
                        id_con[self.cur].append(id_step)
                        id_step += 1
                        i += 1
                out.close()

            print '     Frame:', a_step
            print id_con[self.cur]
            imgs[self.nxt] = readImg(self.img_dir + '%06d.jpg' % a_step)
            i = 0
            while i < a_n:
                attrs = gtIn.readline().strip().split(',')
                if float(attrs[6]) >= tau_conf_score:
                    # if int(attrs[0]) != a_step:
                    #     print attrs
                    #     print 'Something is Wrong! %d != %d'%(int(attrs[0]), a_step)
                    attrs.append(1)
                    line_con[self.nxt].append(attrs)
                    id_con[self.nxt].append(-1)
                    i += 1

            # update the edges
            # print 'T',
            ret = self.a_train_set.getRet()
            for i in xrange(len(self.a_train_set.candidates)):
                a_e, a_vs_index, a_vr_index = self.a_train_set.candidates[i]
                m_e, m_vs_index, m_vr_index = self.m_train_set.candidates[i]
                if a_vs_index != m_vs_index or a_vr_index != m_vr_index:
                    print 'Something is wrong!'
                    print 'a_vs_index = %d, m_vs_index = %d'%(a_vs_index, m_vs_index)
                    print 'a_vr_index = %d, m_vr_index = %d'%(a_vr_index, m_vr_index)
                    raw_input('Continue?')
                if ret[a_vs_index][a_vr_index] == 1.0:
                    continue
                a_e = a_e.to(self.device).view(1,-1)
                a_v1 = self.a_train_set.getApp(1, a_vs_index)
                a_v2 = self.a_train_set.getApp(0, a_vr_index)
                a_e_ = self.AEphi(a_e, a_v1, a_v2, a_u_)
                self.a_train_set.edges[a_vs_index][a_vr_index] = a_e_.data.view(-1)
                a_tmp = F.softmax(a_e_)
                a_tmp = a_tmp.cpu().data.numpy()[0]

                m_e = m_e.to(self.device).view(1,-1)
                m_v1 = self.m_train_set.getMotion(1, m_vs_index)
                m_v2 = self.m_train_set.getMotion(0, m_vr_index, m_vs_index, line_con[self.cur][m_vs_index][-1])
                m_e_ = self.MEphi(m_e, m_v1, m_v2, m_u_)
                self.m_train_set.edges[m_vs_index][m_vr_index] = m_e_.data.view(-1)
                m_tmp = F.softmax(m_e_)
                m_tmp = m_tmp.cpu().data.numpy()[0]

                ret[a_vs_index][a_vr_index] = float(a_tmp[0])*self.alpha + float(m_tmp[0])*(1-self.alpha)

            # self.a_train_set.showE(outFile)
            # self.m_train_set.showE(outFile)

            # for j in ret:
            #     print j
            results = self.hungarian.compute(ret)

            out = open(outFile, 'a')
            look_up = set(j for j in xrange(a_n))
            for (i, j) in results:
                # print (i,j)
                if ret[i][j] >= tau_threshold:
                    continue
                look_up.remove(j)
                self.m_train_set.updateVelocity(i, j, line_con[self.cur][i][-1], False)

                id = id_con[self.cur][i]
                id_con[self.nxt][j] = id
                attr1 = line_con[self.cur][i]
                attr2 = line_con[self.nxt][j]
                attr2[1] = str(id)
                if attr1[-1] > 1:
                    # for the missing detections & side connection
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
                if id==23:
                    print id_con[self.nxt][j], line_con[self.nxt][j]

            for j in look_up:
                self.m_train_set.updateVelocity(-1, j, tag=False)

            for i in xrange(a_n):
                attrs = line_con[self.nxt][i]
                color = self.color[1]
                state = '_C'
                if id_con[self.nxt][i] == -1:
                    color = self.color[0]
                    state = '_B'
                    id_con[self.nxt][i] = id_step
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

                # if i not in look_up:
                #     color = self.color[2]
                #     state = '_M'

                # draw the rectrangle
                x, y = int(float(attrs[2])), int(float(attrs[3]))
                w, h = int(float(attrs[4])), int(float(attrs[5]))
                cv2.rectangle(imgs[self.nxt], (x, y), (x+w, y+h), color, 2)
                cv2.putText(imgs[self.nxt], attrs[1]+state, (x+3, y+15), font, 0.6, color, 2, cv2.LINE_AA)

            out.close()

            # visualization
            cv2.imwrite(self.pre_win + '.png', imgs[self.cur])
            cv2.imwrite(self.cur_win + '.png', imgs[self.nxt])
            for i in xrange(a_m):
                if id_con[self.cur][i] == 23:
                    print line_con[self.cur][i]
                    break
            if going_tag == 0:
                id1, id2 = 1, 1
                while id1 != -1:
                    inp = raw_input('Input:')
                    if ',' in inp:
                        nums = inp.split(',')
                        id1, id2 = int(nums[0]), int(nums[1])
                        if id1 != -1:
                            id_tag = int(nums[2])
                            if id_tag:
                                # t -> t-1
                                for i in xrange(a_n):
                                    if id_con[self.nxt][i] == id1:
                                        id1 = i
                                        break
                                for i in xrange(a_m):
                                    if id_con[self.cur][i] == id2:
                                        id2 = i
                                        break
                                print ret[id2][id1]
                            else:
                                # t-1 -> t
                                for i in xrange(a_n):
                                    if id_con[self.cur][i] == id1:
                                        id1 = i
                                        break
                                for i in xrange(a_m):
                                    if id_con[self.nxt][i] == id2:
                                        id2 = i
                                        break
                                print ret[id1][id2]
                        else:
                            going_tag = 1
                            going_f = id2
                    else:
                        id1 = int(inp)
                        if id1 != -1:
                            for i in xrange(a_n):
                                if id_con[self.nxt][i] == id1:
                                    id1 = i
                                    break
                            for i in xrange(a_n):
                                if ret[i][id1] != 1:
                                    print id_con[self.cur][i], ret[i][id1]

            # For missing & Occlusion
            index = 0
            for (i, j) in results:
                while i != index:
                    # if a_step > 80:
                    #     print id_con[self.cur][index], line_con[self.cur][index]
                    attrs = line_con[self.cur][index]
                    # print '*', attrs, '*'
                    if attrs[-1] + a_t_gap <= gap:
                        attrs[-1] += a_t_gap
                        line_con[self.nxt].append(attrs)
                        id_con[self.nxt].append(id_con[self.cur][index])
                        self.a_train_set.moveApp(index)
                        self.m_train_set.moveMotion(index)
                    index += 1

                if ret[i][j] >= tau_threshold:
                    # if a_step > 80:
                    #     print id_con[self.cur][index], line_con[self.cur][index]
                    attrs = line_con[self.cur][index]
                    # print '*', attrs, '*'
                    if attrs[-1] + a_t_gap <= gap:
                        attrs[-1] += a_t_gap
                        line_con[self.nxt].append(attrs)
                        id_con[self.nxt].append(id_con[self.cur][index])
                        self.a_train_set.moveApp(index)
                        self.m_train_set.moveMotion(index)

                index += 1
            while index < a_m:
                if a_step > 80:
                    print id_con[self.cur][index], line_con[self.cur][index]
                attrs = line_con[self.cur][index]
                # print '*', attrs, '*'
                if attrs[-1] + a_t_gap <= gap:
                    attrs[-1] += a_t_gap
                    line_con[self.nxt].append(attrs)
                    id_con[self.nxt].append(id_con[self.cur][index])
                    self.a_train_set.moveApp(index)
                    self.m_train_set.moveMotion(index)
                index += 1

            # con = self.m_train_set.cleanEdge()
            # for i in xrange(len(con)-1, -1, -1):
            #     index = con[i]
            #     del line_con[self.nxt][index]
            #     del id_con[self.nxt][index]

            line_con[self.cur] = []
            id_con[self.cur] = []
            cv2.imwrite('Show/%06d.png'%(a_step-1), imgs[self.cur])
            imgs[self.cur] = []
            # print head+step, results
            self.a_train_set.swapFC()
            self.m_train_set.swapFC()
            self.swapFC()
        gtIn.close()
        print '     The results:', id_step, self.bbx_counter

        # tra_tst = 'training sets' if head == 1 else 'validation sets'
        # out = open(outFile, 'a')
        # print >> out, tra_tst
        # out.close()

if __name__ == '__main__':
    try:
        start_x = time.time()
        for x in xrange(1, 2):
            ALPHA_TAG = x
            start_a = time.time()
            for a in xrange(7, 8):
                if not os.path.exists('Results/'):
                    os.mkdir('Results/')

                # types = ['DPM', 'SDP', 'FRCNN']
                types = ['FRCNN']
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
                            gn = GN(test_seqs[i], test_lengths[i], a/10.0)
                        else:
                            seq_dir = 'MOT%d-%02d-%s' % (year, seqs[i], type)
                            sequence_dir = '../MOT/MOT%d/train/'%year + seq_dir
                            print ' ', sequence_dir

                            start = time.time()
                            print '     Evaluating Graph Network...'
                            gn = GN(seqs[i], lengths[i], a/10.0)
                            print '     Recover the number missing detections:', gn.missingCounter
                            print '     The number of sideConnections:', gn.sideConnection
                        print 'Time consuming:', (time.time()-start)/60.0
                    print 'Time consuming:', (time.time()-head)/60.0
                print 'Total time consuming:', (time.time()-start_a)/60.0
            print 'Final time consuming:', (time.time()-start_x)/60.0
    except KeyboardInterrupt:
        print 'Time consuming:', (time.time()-start)/60.0
        print ''
        print '-'*90
        print 'Existing from training early.'
