# from __future__ import print_function
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from mot_model import *
# from mot_dataset import readBB
from jingyuan_dataset import readBB
import time, random
from munkres import Munkres
from pycrayon import CrayonClient
from global_set import edge_initial, u_initial, u_evaluation, train_test

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


class GN():
    def __init__(self, lr=5e-3, cuda=True):
        # all the tensor should set the 'volatile' as True, and False when update the network
        self.hungarian = Munkres()
        self.cuda = cuda
        self.nEpochs = 999
        self.tau = 3
        # self.frame_end = len(self.edges[0])-1
        self.lr = lr
        self.modelDir = 'Model/'
        self.outName = 'result.txt'

        self.show_process = 0   # interaction
        self.step_input = 1

        self.helpEpoch = None
        self.helpName = None
        self.helpCounter = 0
        self.helpLoss1 = None
        self.helpLoss2 = None

        print '     Loading Data...'
        start = time.time()
        bbAll, aAll, eAll, cAll, self.tail = readBB(self.cuda).getC(0)
        t_data = time.time() - start
        # print self.tail
        self.tail = train_test
        self.tau = 1
        self.frame_head = 1
        self.frame_end = self.frame_head + self.tau
        self.loss_threhold = 0.03

        print '     Preparing the model...'
        self.reset(E=eAll, B=bbAll, A=aAll, C=cAll)

        self.Uphi = uphi()
        self.Ephi = ephi()

        self.criterion = nn.MSELoss() if criterion_s else nn.CrossEntropyLoss()

        self.optimizer = optim.Adam([
            {'params': self.Uphi.parameters()},
            {'params': self.Ephi.parameters()}],
            lr=lr)

        print '     Logging...'
        self.log(t_data)

        if self.cuda:
            print '     >>>>>> CUDA <<<<<<'
            self.Uphi = self.Uphi.cuda()
            self.Ephi = self.Ephi.cuda()
            self.criterion = self.criterion.cuda()

    def print_grad(self, g):
        out = open(self.outName, 'a')
        if self.helpCounter == 0:
            print >> out, '-'*36, self.frame_head, self.helpEpoch[0], self.helpEpoch[1],
            print >> out, '-'*36
        tmp = torch.mean(torch.abs(g)).cpu()
        tmp = tmp.data.numpy()[0]
        print >> out, tmp,
        if self.helpCounter:
            print >> out, self.helpLoss2
        else:
            print >> out, self.helpLoss1, '| ',
        out.close()
        self.helpCounter += 1
        self.helpCounter %= 2

    def log(self, t_data):
        out = open(self.outName, 'w')
        print >> out, self.criterion
        print >> out, 'lr:{}'.format(self.lr)
        print >> out, self.optimizer.state_dict()
        print >> out, self.Uphi
        print >> out, self.Ephi
        print >> out, 'Time consuming for loading datasets:', t_data
        out.close()

    def resetU(self):
        if u_initial:
            self.u = torch.FloatTensor([random.random() for i in xrange(u_num)]).view(1, -1)
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

    def updateNetwork(self):
        step = 1
        average_epoch = 0
        self.frame_head = 1
        # self.frame_end = self.frame_head+self.tau
        # self.showE()
        edge_counter = 0.0
        counter_tag = 1
        while self.frame_head < self.tail:
            start = time.time()
            show_name = 'LOSS_{}'.format(step)
            print '         Step -', step
            self.frame_end = self.frame_head+self.tau
            counter_tag = 1
            for epoch in xrange(1, self.nEpochs):
                # compute the aggregation of e, v
                es = []  # set of edges
                v_index = 0
                candidates = []
                for s in xrange(len(self.edges)):  # the sequence
                    for f in xrange(self.frame_head, self.frame_end):  # the frame
                        m = len(self.edges[s][f])
                        n = len(self.edges[s][f][0])
                        if counter_tag:
                            edge_counter += m*n
                        vs_index = v_index
                        for j in xrange(m):
                            vr_index = v_index + m
                            for k in xrange(n):
                                e = Variable(self.edges[s][f][j][k], requires_grad=True)
                                gt = Variable(self.connection[s][f][j][k])
                                if self.cuda:
                                    e, gt = e.cuda(), gt.cuda()
                                es.append(e)
                                candidates.append([e, gt, vs_index, vr_index])
                                vr_index += 1
                            vs_index += 1
                        v_index += m

                    vs = []  # set of vertices
                    for f in xrange(self.frame_head, self.frame_end+1):  # the frame
                        m = len(self.appearance[s][f])
                        for i in xrange(m):
                            v = Variable(self.appearance[s][f][i], requires_grad=True)
                            vs.append(v.cuda() if self.cuda else v)

                    num = 0
                    epoch_loss = 0.0
                    arpha_loss = 0.0
                    E = self.aggregate(es)
                    V = self.aggregate(vs)
                    for edge in candidates:
                        self.helpEpoch = (epoch, num)
                        self.optimizer.zero_grad()

                        u_ = self.Uphi(E, V, Variable(self.u, requires_grad=True).cuda())
                        e, gt, vs_index, vr_index = edge
                        v1 = vs[vs_index]
                        v2 = vs[vr_index]
                        e_ = self.Ephi(e, v1, v2, u_)

                        # u_.register_hook(self.print_grad)
                        # e.register_hook(self.print_grad)

                        if self.show_process:
                            print '-'*66
                            print vs_index, vr_index
                            print 'e:', e.cpu().data.numpy()[0][0],
                            print 'e_:', e_.cpu().data.numpy()[0][0],
                            if criterion_s:
                                print 'GT:', gt.cpu().data.numpy()[0][0]
                            else:
                                print 'GT:', gt.cpu().data.numpy()[0]

                        # Penalize the u to let its value not too big
                        arpha = torch.mean(torch.abs(u_))
                        arpha_loss += arpha.data[0]
                        self.helpLoss1 = arpha.data[0]
                        arpha.backward(retain_graph=True)

                        # Penalize the e to let its value not too big
                        # arphaE = torch.mean(torch.abs(e_))
                        # arphaE.backward(retain_graph=True)

                        #  The regular loss
                        loss = self.criterion(e_, gt)
                        epoch_loss += loss.data[0]
                        self.helpLoss2 = loss.data[0]
                        loss.backward()

                        # update the network: Uphi and Ephi
                        self.optimizer.step()

                        #  Show the parameters of the Uphi and Ephi to check the process of optimiser
                        # print self.Uphi.features[0].weight.data
                        # print self.Ephi.features[0].weight.data
                        # raw_input('continue?')

                        num += 1

                    if self.show_process and self.step_input:
                        a = raw_input('Continue(0-step, 1-run, 2-run with showing)?')
                        if a == '1':
                            self.show_process = 0
                        elif a == '2':
                            self.step_input = 0

                    epoch_loss /= num
                    print '         Loss of epoch {}: {}.'.format(epoch, epoch_loss)
                    exper[0].add_scalar_value(show_name, epoch_loss, epoch)
                    exper[1].add_scalar_value(show_name, arpha_loss, epoch)
                if epoch_loss < self.loss_threhold:
                    break
                counter_tag = 0

            print '         Time consuming:{}\n\n'.format(time.time()-start)
            self.updateUE()
            self.showE()
            self.frame_head += self.tau
            average_epoch += epoch
            exper[2].add_scalar_value('epoch', epoch, step)
            step += 1
        print 'Average edge:', edge_counter*1.0/step, '.',
        print 'Average epoch:', average_epoch*1.0/step, 'for',
        print 'Random' if edge_initial else 'IoU'

    def updateUE(self):
        # compute the aggregation of e, v
        es = []  # set of edges
        v_index = 0
        candidates = []
        for s in xrange(len(self.edges)):  # the sequence
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
                        candidates.append([e, vs_index, vr_index, s, f, j, k])
                        vr_index += 1
                    vs_index += 1
                v_index += m

        vs = []  # set of vertices
        for s in xrange(len(self.edges)):  # the sequence
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
        self.u = u_.cpu().data if self.cuda else u_.data

        # update the edges
        for edge in candidates:
            e, vs_index, vr_index, s, f, j, k = edge
            v1 = vs[vs_index]
            v2 = vs[vr_index]
            e_ = self.Ephi(e, v1, v2, u_)
            self.edges[s][f][j][k] = e_.cpu().data if self.cuda else e_.data

    def saveModel(self):
        print 'Saving the Uphi model...'
        torch.save(self.Uphi, self.modelDir+'in_place_uphi.pth')
        print 'Saving the Ephi model...'
        torch.save(self.Ephi, self.modelDir+'in_place_ephi.pth')
        print 'Saving the global variable u...'
        torch.save(self.u, self.modelDir+'u.pth')
        print 'Done!'

    def update(self):
        start = time.time()
        self.preEvaluate()
        self.updateNetwork()
        self.saveModel()
        self.evaluation()
        print 'The final time consuming:{}\n\n'.format(time.time()-start)

    def preEvaluate(self):
        total_gt = 0.0
        total_ed = 0.0
        for s in xrange(len(self.edges)):  # the sequence
            self.frame_head = self.tail
            while self.frame_head < 2*train_test:
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
            while self.frame_head < 2*train_test:
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

if __name__ == '__main__':
    start = time.time()
    gn = GN()
    try:
        print '     Starting Graph Network...'
        gn.update()
    except KeyboardInterrupt:
        a = raw_input('Evaluating?')
        if a == 'yes':
            gn.evaluation()
        print 'Time consuming:', time.time()-start
        print ''
        print '-'*90
        print 'Existing from training early.'
