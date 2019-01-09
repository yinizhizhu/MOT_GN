# from __future__ import print_function
import numpy as np
from m_mot_model import *
from munkres import Munkres
import torch.nn.functional as F
import time, os, shutil, random
from global_set import edge_initial, SEQLEN,\
    tau_threshold, gap, f_gap, show_recovering, decay, u_update#, tau_conf_score
from mot_model import appearance
from test_dataset import ADatasetFromFolder
from m_test_dataset import MDatasetFromFolder

torch.manual_seed(123)
np.random.seed(123)


def deleteDir(del_dir):
    shutil.rmtree(del_dir)


tt_tag = 1  # 1 - test, 0 - train


class GN():
    def __init__(self, cuda=True):
        '''
        Evaluating with the MotMetrics
        :param seq_index: the number of the sequence
        :param tt: train_test
        :param length: the number of frames which is used for training
        :param cuda: True - GPU, False - CPU
        '''
        self.hungarian = Munkres()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.alpha = 0.7
        self.missingCounter = 0
        self.sideConnection = 0

        self.outName = 'Duke.txt'
        out = open(self.outName, 'w')
        out.close()

        self.tau_conf_score = 0.5
        self.initOut()

    def initOut(self):
        for i in xrange(1, 9):
            start_t = time.time()
            self.camera = i
            self.bbx_counter = 0

            print '     Loading the model...'
            self.loadAModel()
            self.loadMModel()

            print '     Loading Data...'
            self.a_train_set = ADatasetFromFolder(i, self.tau_conf_score)
            self.m_train_set = MDatasetFromFolder(i, self.tau_conf_score)

            self.evaluation()
            print '     This camera:', self.camera, '- Time:', (time.time() - start_t)/60

    def loadAModel(self):
        from mot_model import uphi, ephi, vphi
        model_dir = 'Duke_app2'
        name = 'all_8'
        if edge_initial == 0:
            i_name = 'IoU'
        elif edge_initial == 1:
            i_name = 'Random'
        tail = 2
        self.AUphi = torch.load('../%s/Results/MOT16/%s/%s/uphi_%d.pth'%(model_dir, i_name, name, tail)).to(self.device)
        self.AVphi = torch.load('../%s/Results/MOT16/%s/%s/vphi_%d.pth'%(model_dir,i_name, name, tail)).to(self.device)
        self.AEphi1 = torch.load('../%s/Results/MOT16/%s/%s/ephi1_%d.pth'%(model_dir,i_name, name, tail)).to(self.device)
        self.AEphi2 = torch.load('../%s/Results/MOT16/%s/%s/ephi2_%d.pth'%(model_dir,i_name, name, tail)).to(self.device)
        self.Au = torch.load('../%s/Results/MOT16/%s/%s/u_%d.pth'%(model_dir,i_name, name, tail))
        self.Au = self.Au.to(self.device)

    def loadMModel(self):
        from m_mot_model import uphi, ephi
        model_dir = 'Duke_m'
        name = 'all_8'
        if edge_initial == 0:
            i_name = 'IoU'
        elif edge_initial == 1:
            i_name = 'Random'
        tail = 2
        self.MUphi = torch.load('../%s/Results/DukeMTMC/%s/%s/uphi_%d.pth'%(model_dir,i_name, name, tail)).to(self.device)
        self.MEphi = torch.load('../%s/Results/DukeMTMC/%s/%s/ephi_%d.pth'%(model_dir,i_name, name, tail)).to(self.device)
        self.Mu = torch.load('../%s/Results/DukeMTMC/%s/%s/u_%d.pth'%(model_dir,i_name, name, tail))
        self.Mu = self.Mu.to(self.device)

    def swapFC(self):
        self.cur = self.cur ^ self.nxt
        self.nxt = self.cur ^ self.nxt
        self.cur = self.cur ^ self.nxt

    def linearModel(self, out, attr1, attr2):
        # print 'I got you! *.*'
        self.sideConnection += 1
        frame, frame2 = int(attr1[2]), int(attr2[2])
        t = frame2 - frame
        if t > f_gap:
            return
        x1, y1, w1, h1 = float(attr1[3]), float(attr1[4]), float(attr1[5]), float(attr1[6])
        x2, y2, w2, h2 = float(attr2[3]), float(attr2[4]), float(attr2[5]), float(attr2[6])

        x_delta = (x2-x1)/t
        y_delta = (y2-y1)/t
        w_delta = (w2-w1)/t
        h_delta = (h2-h1)/t

        for i in xrange(1, t):
            frame += 1
            x1 += x_delta
            y1 += y_delta
            w1 += w_delta
            h1 += h_delta
            attr1[2] = str(frame)
            attr1[3] = str(x1)
            attr1[4] = str(y1)
            attr1[5] = str(w1)
            attr1[6] = str(h1)
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

    def evaluation(self):
        gtIn = open('Detections/%d.txt'%self.camera, 'r')
        self.cur, self.nxt = 0, 1
        line_con = [[], []]
        id_con = [[], []]
        id_step = 1

        head = 1
        a_step = head + self.a_train_set.setBuffer(head)
        m_step = head + self.m_train_set.setBuffer(head)
        if a_step != m_step:
            print 'Something is wrong!'
            print 'a_step =', a_step, ', m_step =', m_step
            raw_input('Continue?')

        while a_step <= SEQLEN:
            # print '*********************************'
            a_t_gap = self.a_train_set.loadNext()
            m_t_gap = self.m_train_set.loadNext()
            if a_t_gap != m_t_gap:
                print 'Something is wrong!'
                print 'a_t_gap =', a_t_gap, ', m_t_gap =', m_t_gap
                raw_input('Continue?')
            a_step += a_t_gap
            m_step += m_step
            if a_step > SEQLEN:
                break

            print a_step,
            if a_step % 1000 == 0:
                print ''
            # print head+step, 'F',

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
                out = open(self.outName, 'a')
                i = 0
                while i < a_m:
                    attrs = gtIn.readline().strip().split(',')
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
            while i < a_n:
                attrs = gtIn.readline().strip().split(',')
                attrs.append(1)
                line_con[self.nxt].append(attrs)
                id_con[self.nxt].append(-1)
                i += 1

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

                t = line_con[self.cur][a_vs_index][-1]
                if decay_tag[a_vs_index] > 0:
                    A = min(float(a_tmp[0]) * pow(decay, t-1), 1.0)
                    M = min(float(m_tmp[0]) * pow(decay, t-1), 1.0)
                else:
                    A = float(a_tmp[0])
                    M = float(m_tmp[0])
                ret[a_vs_index][a_vr_index] = A*self.alpha + M*(1-self.alpha)

            # self.a_train_set.showE(outFile)
            # self.m_train_set.showE(outFile)

            # for j in ret:
            #     print j
            results = self.hungarian.compute(ret)

            out = open(self.outName, 'a')
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

            if u_update:
                self.Mu = m_u_.data
                self.Au = u1.data

            for j in look_up:
                self.m_train_set.updateVelocity(-1, j, False)

            for i in xrange(a_n):
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
                    if attrs[-1] + a_t_gap <= gap:
                        attrs[-1] += a_t_gap
                        line_con[self.nxt].append(attrs)
                        id_con[self.nxt].append(id_con[self.cur][index])
                        self.a_train_set.moveApp(index)
                        self.m_train_set.moveMotion(index)
                    index += 1
                if ret[i][j] >= tau_threshold:
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
        start = time.time()
        print '     Evaluating Graph Network...'
        gn = GN()
        print 'Time consuming:', (time.time()-start)/60.0
    except KeyboardInterrupt:
        print 'Time consuming:', (time.time()-start)/60.0
        print ''
        print '-'*90
        print 'Existing from training early.'
