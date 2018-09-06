# from __future__ import print_function
import numpy as np
import time, torch
import torch.nn.functional as F
from munkres import Munkres
from mot_model import u_num
from pycrayon import CrayonClient
from torch.autograd import Variable
from jingyuan_dataset import readBB
from global_set import u_initial, u_evaluation, train_test, criterion_s

curveName = list()
curveName.append('regular_loss')
curveName.append('penalized_loss')
curveName.append('epoch_num')

cc = CrayonClient(hostname="localhost", port=8889)

exper = list()
for name in curveName:
    cc.remove_experiment(name)
    exper.append(cc.create_experiment(name))

torch.manual_seed(123)
np.random.seed(123)

learn_rate = [1e-3, 1e-4]


class cross_sequence():
    def __init__(self, cuda=True):
        # all the tensor should set the 'volatile' as True, and False when update the network
        self.hungarian = Munkres()
        self.cuda = cuda
        self.nEpochs = 999
        self.tau = 3
        # self.frame_end = len(self.edges[0])-1
        self.modelDir = 'Model/'
        self.outName = 'result.txt'

        self.show_process = 0   # interaction
        self.step_input = 1

        print '     Loading Data...'
        start = time.time()
        bbAll, aAll, eAll, cAll, self.tail = readBB(self.cuda).getC(0)
        t_data = time.time() - start
        # print self.tail
        self.tail = 30
        self.tau = 1
        self.frame_head = 1
        self.frame_end = self.frame_head + self.tau
        self.loss_threhold = 0.03

        print '     Preparing the model...'
        self.reset(E=eAll, B=bbAll, A=aAll, C=cAll)

        self.Uphi = torch.load(self.modelDir+'in_place_uphi.pth')
        self.Ephi = torch.load(self.modelDir+'in_place_ephi.pth')

        print '     Logging...'
        self.log(t_data)

        if self.cuda:
            print '     >>>>>> CUDA <<<<<<'
            self.Uphi = self.Uphi.cuda()
            self.Ephi = self.Ephi.cuda()

    def log(self, t_data):
        out = open(self.outName, 'w')
        print >> out, self.Uphi
        print >> out, self.Ephi
        print >> out, 'Time consuming for loading datasets:', t_data
        out.close()

    def resetU(self):
        if u_initial:
            self.u = torch.load(self.modelDir+'u.pth')
            # self.u = torch.FloatTensor([random.random() for i in xrange(u_num)]).view(1, -1)
        else:
            self.u = torch.FloatTensor([0.0 for i in xrange(u_num)]).view(1, -1)

    def reset(self, E, B, A, C):
        self.edges = E
        self.bbx = B
        self.appearance = A
        self.connection = C
        self.resetU()

    def aggregate(self, set):
        if len(set):
            num = len(set)
            rho = sum(set)
            return rho/num
        print '     The set is empty!'
        return None

    def eval(self):
        self.preEvaluate()
        self.evaluation()

    def preEvaluate(self):
        total_gt = 0.0
        total_ed = 0.0
        for s in xrange(len(self.edges)):  # the sequence
            self.frame_head = self.tail
            while self.frame_head < train_test:
                print self.frame_head, 'F',
                self.frame_end = self.frame_head + self.tau
                es = []  # set of edges
                v_index = 0
                candidates = []
                for f in xrange(self.frame_head, self.frame_end):  # the frame
                    m = len(self.edges[s][f])
                    n = len(self.edges[s][f][0])
                    vs_index = v_index
                    for j in xrange(m):
                        vr_index = v_index + m
                        for k in xrange(n):
                            e = Variable(self.edges[s][f][j][k], volatile=True)
                            if self.cuda:
                                e = e.cuda()
                            es.append(e)
                            candidates.append([e, vs_index, vr_index, f, j, k])
                            vr_index += 1
                        vs_index += 1
                    v_index += m

                print 'S',
                vs = []  # set of vertices
                for f in xrange(self.frame_head, self.frame_end+1):  # the frame
                    m = len(self.appearance[s][f])
                    for i in xrange(m):
                        v = Variable(self.appearance[s][f][i], volatile=True)
                        if self.cuda:
                            v = v.cuda()
                        vs.append(v)

                # update the global variable u
                e_ = self.aggregate(es)
                v_ = self.aggregate(vs)
                u_ = self.Uphi(e_, v_, Variable(self.u, volatile=True).cuda())
                # self.u = u_.cpu().data if self.cuda else u_.data

                print 'Fo'
                step_ed = 0.0
                rets = []
                step_gt = 0.0
                for f in xrange(self.frame_head, self.frame_end):  # the frame
                    m = len(self.edges[s][f])
                    n = len(self.edges[s][f][0])
                    ret = [[0.0 for i in xrange(n)] for j in xrange(m)]
                    rets.append(ret)
                    for i in xrange(m):
                        for j in xrange(n):
                            step_gt += self.connection[s][f][i][j].numpy()[0]
                total_gt += step_gt

                # update the edges
                print 'T',
                for edge in candidates:
                    e, vs_index, vr_index, f, j, k = edge
                    v1 = vs[vs_index]
                    v2 = vs[vr_index]
                    e_ = self.Ephi(e, v1, v2, u_)
                    tmp = F.softmax(e_)
                    tmp = tmp.cpu().data.numpy()[0]
                    rets[f - self.frame_head][j][k] = float(tmp[0])

                for ret in rets:
                    for j in ret:
                        print j
                    results = self.hungarian.compute(ret)
                    print self.frame_head, results,
                    for (j, k) in results:
                        step_ed += self.connection[s][f][j][k].numpy()[0]
                total_ed += step_ed

                print 'Fi'
                print 'Step ACC:{}/{}({}%)'.format(int(step_ed), int(step_gt), step_ed/step_gt*100)
                self.frame_head += self.tau
        print 'Final ACC:{}/{}({}%)'.format(int(total_ed), int(total_gt), total_ed/total_gt*100)
        out = open(self.outName, 'a')
        print >> out, 'Final ACC:', total_ed/total_gt
        out.close()

    def evaluation(self):
        if u_evaluation:  # Initiate the u with random at evaluation
            self.resetU()
            #  Otherwise, we use the learned u at evaluation

        total_gt = 0.0
        total_ed = 0.0
        for s in xrange(len(self.edges)):  # the sequence
            self.frame_head = self.tail
            while self.frame_head < train_test:
                print self.frame_head, 'F',
                self.frame_end = self.frame_head + self.tau
                es = []  # set of edges
                v_index = 0
                candidates = []
                for f in xrange(self.frame_head, self.frame_end):  # the frame
                    m = len(self.edges[s][f])
                    n = len(self.edges[s][f][0])
                    vs_index = v_index
                    for j in xrange(m):
                        vr_index = v_index + m
                        for k in xrange(n):
                            e = Variable(self.edges[s][f][j][k], volatile=True)
                            if self.cuda:
                                e = e.cuda()
                            es.append(e)
                            candidates.append([e, vs_index, vr_index, f, j, k])
                            vr_index += 1
                        vs_index += 1
                    v_index += m

                print 'S',
                vs = []  # set of vertices
                for f in xrange(self.frame_head, self.frame_end+1):  # the frame
                    m = len(self.appearance[s][f])
                    for i in xrange(m):
                        v = Variable(self.appearance[s][f][i], volatile=True)
                        if self.cuda:
                            v = v.cuda()
                        vs.append(v)

                # update the global variable u
                e_ = self.aggregate(es)
                v_ = self.aggregate(vs)
                u_ = self.Uphi(e_, v_, Variable(self.u, volatile=True).cuda())
                # self.u = u_.cpu().data if self.cuda else u_.data

                # update the edges
                print 'T',
                for edge in candidates:
                    e, vs_index, vr_index, f, j, k = edge
                    v1 = vs[vs_index]
                    v2 = vs[vr_index]
                    e_ = self.Ephi(e, v1, v2, u_)
                    self.edges[s][f][j][k] = e_.cpu().data if self.cuda else e_.data

                self.showE()

                print 'Fo'
                step_ed = 0.0
                for f in xrange(self.frame_head, self.frame_end):  # the frame
                    m = len(self.edges[s][f])
                    n = len(self.edges[s][f][0])
                    ret = [[0.0 for i in xrange(n)] for j in xrange(m)]
                    step_gt = 0.0
                    for j in xrange(m):
                        for k in xrange(n):
                            step_gt += self.connection[s][f][j][k].numpy()[0]
                            e = F.softmax(Variable(self.edges[s][f][j][k]))
                            tmp = e.data.numpy()[0]
                            ret[j][k] = float(tmp[0])
                    total_gt += step_gt
                    for j in ret:
                        print j
                    results = self.hungarian.compute(ret)
                    print self.frame_head, results,
                    for (j, k) in results:
                        step_ed += self.connection[s][f][j][k].numpy()[0]
                total_ed += step_ed
                print 'Fi'
                print 'Step ACC:{}/{}({}%)'.format(int(step_ed), int(step_gt), step_ed/step_gt*100)
                self.frame_head += self.tau
        print 'Final ACC:{}/{}({}%)'.format(int(total_ed), int(total_gt), total_ed/total_gt*100)
        out = open(self.outName, 'a')
        print >> out, 'Final ACC:', total_ed/total_gt
        out.close()

    def showE(self):
        out = open(self.outName, 'a')
        print >> out, ''
        print >> out, '-'*45, '-'*45
        print >> out, '     edge'
        if criterion_s:
            for i in xrange(self.frame_head, self.frame_end):
                print >> out, ' ', i
                print >> out, self.outEForMSELoss(self.edges[0][i], 1)
        else:
            for i in xrange(self.frame_head, self.frame_end):
                print >> out, ' ', i
                print >> out, self.outEForCrossEntropyLoss(self.edges[0][i], 1)
        print >> out, '     connection'
        if criterion_s:
            for i in xrange(self.frame_head, self.frame_end):
                print >> out, ' ', i
                print >> out, self.outEForMSELoss(self.connection[0][i], 0)
        else:
            for i in xrange(self.frame_head, self.frame_end):
                print >> out, ' ', i
                print >> out, self.outEForCrossEntropyLoss(self.connection[0][i], 0)
        print >> out, '     u'
        print >> out, self.u.view(10, -1) # reshape the size of z with aspect of 10 * 10
        out.close()

    def outEForCrossEntropyLoss(self, edges, tag):
        m = len(edges)
        n = len(edges[0])
        if tag:
            ret = [[None for i in xrange(n)] for j in xrange(m)]
            con1 = [[None for i in xrange(n)] for j in xrange(m)]
            con2 = [[None for i in xrange(n)] for j in xrange(m)]
        else:
            ans = [[None for i in xrange(n)] for j in xrange(m)]
        for i in xrange(m):
            for j in xrange(n):
                if tag:
                    tmp = F.softmax(Variable(edges[i][j])).data.numpy()[0]
                    ret[i][j] = float(tmp[1])

                    tmp = edges[i][j].numpy()[0]
                    con1[i][j] = float(tmp[0])
                    con2[i][j] = float(tmp[1])
                else:
                    tmp = edges[i][j].numpy()[0]
                    ans[i][j] = float(tmp)
        if tag:
            ret = torch.FloatTensor(ret)
            con1 = torch.FloatTensor(con1)
            con2 = torch.FloatTensor(con2)
            return 'Probability', ret, 'Output', con1, con2
        ans = torch.FloatTensor(ans)
        return ans

    def outEForMSELoss(self, edges, tag):
        m = len(edges)
        n = len(edges[0])
        ans = [[None for i in xrange(n)] for j in xrange(m)]
        for i in xrange(m):
            for j in xrange(n):
                tmp = edges[i][j].numpy()[0][0] if tag else edges[i][j].numpy()[0]
                ans[i][j] = float(tmp)
        ans = torch.FloatTensor(ans)
        return ans

start = time.time()
gn = cross_sequence()
try:
    print '     Starting Graph Network...'
    gn.eval()
except KeyboardInterrupt:
    a = raw_input('Evaluating?')
    if a == 'yes':
        gn.evaluation()
    print 'Time consuming:', time.time()-start
    print ''
    print '-'*90
    print 'Existing from training early.'
