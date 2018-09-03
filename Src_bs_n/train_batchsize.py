# from __future__ import print_function
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from mot_model import *
from torch.utils.data import DataLoader
from dataset import DatasetFromFolder
import time, random
from munkres import Munkres
from pycrayon import CrayonClient
from global_set import edge_initial, u_initial, train_test

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
        self.outName = 'result.txt'

        self.show_process = 0   # interaction
        self.step_input = 1

        print '     Loading Data...'
        start = time.time()
        train_set = DatasetFromFolder('MOT16/train/MOT16-05', self.outName)
        self.data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=4, shuffle=True)
        t_data = time.time() - start

        # print self.tail
        self.tail = train_test
        self.tau = 1
        self.frame_head = 1
        self.frame_end = self.frame_head + self.tau
        self.loss_threhold = 0.03

        print '     Preparing the model...'
        self.resetU()

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
        while self.frame_head < self.tail:
            self.data_loader.dataset.loadNext()  # Get the next frame
            start = time.time()
            show_name = 'LOSS_{}'.format(step)
            print '         Step -', step
            for epoch in xrange(1, self.nEpochs):
                num = 0
                epoch_loss = 0.0
                arpha_loss = 0.0
                for iteration in enumerate(self.data_loader, 1):
                    e, gt, vs_index, vr_index = iteration

                    self.optimizer.zero_grad()

                    u_ = self.Uphi(self.data_loader.dataset.E, self.data_loader.dataset.V,
                                   Variable(self.u, requires_grad=True).cuda())
                    v1 = self.data_loader.dataset.getApp(1)
                    v2 = self.data_loader.dataset.getApp(0)
                    e_ = self.Ephi(e, v1, v2, u_)

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

            print '         Time consuming:{}\n\n'.format(time.time()-start)
            self.updateUE()
            self.data_loader.dataset.showE()
            self.showU()
            self.frame_head += self.tau
            average_epoch += epoch
            exper[2].add_scalar_value('epoch', epoch, step)
            step += 1
        print 'Average edge:', edge_counter*1.0/step, '.',
        print 'Average epoch:', average_epoch*1.0/step, 'for',
        print 'Random' if edge_initial else 'IoU'

    def updateUE(self):
        u_ = self.Uphi(self.data_loader.dataset.E, self.data_loader.dataset.V,
                       Variable(self.u, volatile=True).cuda())

        self.u = u_.cpu().data if self.cuda else u_.data

        # update the edges
        for edge in self.data_loader.dataset:
            e, gt, vs_index, vr_index = edge
            v1 = self.data_loader.dataset.vs[vs_index]
            v2 = self.data_loader.dataset.vs[vr_index]
            e_ = self.Ephi(e, v1, v2, u_)
            self.data_loader.dataset.edges[vs_index][vr_index] = e_.cpu().data if self.cuda else e_.data

    def update(self):
        start = time.time()
        self.evaluation()
        # self.updateNetwork()
        self.evaluation()
        print 'The final time consuming:{}\n\n'.format(time.time()-start)

    def evaluation(self):
        self.data_loader.dataset.setBuffer(self.tail)
        total_gt = 0.0
        total_ed = 0.0
        self.frame_head = self.tail
        while self.frame_head < 2*train_test:
            self.data_loader.dataset.loadNext()
            print self.frame_head, 'F',
            self.frame_end = self.frame_head + self.tau

            u_ = self.Uphi(self.data_loader.dataset.E, self.data_loader.dataset.V,
                           Variable(self.u, volatile=True).cuda())

            print 'Fo'
            m = self.data_loader.dataset.m
            n = self.data_loader.dataset.n
            ret = [[0.0 for i in xrange(n)] for j in xrange(m)]
            step_gt = self.data_loader.dataset.step_gt
            total_gt += step_gt

            # update the edges
            print 'T',
            for edge in self.data_loader.dataset.candidates:
                e, gt, vs_index, vr_index = edge
                v1 = self.data_loader.dataset.vs[vs_index]
                v2 = self.data_loader.dataset.vs[vr_index]
                e_ = self.Ephi(e, v1, v2, u_)
                self.data_loader.dataset.edges[vs_index][vr_index-m] = e_.cpu().data if self.cuda else e_.data
                tmp = F.softmax(e_)
                tmp = tmp.cpu().data.numpy()[0]
                ret[vs_index][vr_index-m] = float(tmp[0])

            self.data_loader.dataset.showE()
            self.showU()

            for j in ret:
                print j
            results = self.hungarian.compute(ret)
            print self.frame_head, results,
            step_ed = 0.0
            for (j, k) in results:
                step_ed += self.data_loader.dataset.gts[j][k].numpy()[0]
            total_ed += step_ed

            print 'Fi'
            print 'Step ACC:{}/{}({}%)'.format(int(step_ed), int(step_gt), step_ed/step_gt*100)
            self.frame_head += self.tau
        print 'Final ACC:{}/{}({}%)'.format(int(total_ed), int(total_gt), total_ed/total_gt*100)
        out = open(self.outName, 'a')
        print >> out, 'Final ACC:', total_ed/total_gt
        out.close()

    def showU(self):
        out = open(self.outName, 'a')
        print >> out, '     u'
        print >> out, self.u.view(10, -1) # reshape the size of z with aspect of 10 * 10
        out.close()


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
