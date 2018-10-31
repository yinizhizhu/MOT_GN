# from __future__ import print_function
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from dataset import DatasetFromFolder
import time, random, os, shutil
from munkres import Munkres
from global_set import edge_initial, u_initial
from mot_model import *

from tensorboardX import SummaryWriter

torch.manual_seed(123)
np.random.seed(123)

t_dir = ''  # the dir of the final level


def deleteDir(del_dir):
    shutil.rmtree(del_dir)


class GN():
    def __init__(self, lr=5e-4, batchs=8, cuda=True):
        '''
        :param tt: train_test
        :param tag: 1 - evaluation on testing data, 0 - without evaluation on testing data
        :param lr:
        :param batchs:
        :param cuda:
        '''
        # all the tensor should set the 'volatile' as True, and False when update the network
        self.hungarian = Munkres()
        self.device = torch.device("cuda" if cuda else "cpu")
        self.nEpochs = 999
        self.lr = lr
        self.batchsize = batchs
        self.numWorker = 4

        self.show_process = 0   # interaction
        self.step_input = 1

        print '     Preparing the model...'
        self.resetU()

        self.Uphi = uphi().to(self.device)
        self.Vphi = vphi().to(self.device)
        self.Ephi = ephi().to(self.device)

        self.criterion = nn.MSELoss() if criterion_s else nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(self.device)

        self.criterion_v = nn.MSELoss().to(self.device)

        self.optimizer = optim.Adam([
            {'params': self.Uphi.parameters()},
            {'params': self.Vphi.parameters()},
            {'params': self.Ephi.parameters()}],
            lr=lr)

        # seqs = [2, 4, 5, 9, 10, 11, 13]
        # lengths = [600, 1050, 837, 525, 654, 900, 750]
        seqs = [2, 4, 5, 10]
        lengths = [600, 1050, 837, 654]

        for i in xrange(len(seqs)):
            self.writer = SummaryWriter()
            # print '     Loading Data...'
            seq = seqs[i]
            self.seq_index = seq
            start = time.time()
            sequence_dir = '../MOT/MOT16/train/MOT16-%02d'%seq

            self.outName = t_dir+'result_%02d.txt'%seq
            out = open(self.outName, 'w')
            out.close()

            self.train_set = DatasetFromFolder(sequence_dir, self.outName)

            self.train_test = lengths[i]
            self.tag = 0
            self.loss_threhold = 0.03
            self.update()

            print '     Logging...'
            t_data = time.time() - start
            self.log(t_data)

    def getEdges(self):  # the statistic data of the graph among two frames' detections
        self.train_set.setBuffer(1)
        step = 1
        edge_counter = 0.0
        for head in xrange(1, self.train_test):
            self.train_set.loadNext()  # Get the next frame
            edge_counter += self.train_set.m * self.train_set.n
            step += 1
            self.train_set.swapFC()
        out = open(self.outName, 'a')
        print >> out, 'Average edge:', edge_counter*1.0/step
        out.close()

    def showNetwork(self):
        # add the graph into tensorboard
        E = torch.rand(1, 2).to(self.device)
        V = torch.rand(1, 512).to(self.device)
        u = torch.rand(1, 100).to(self.device)
        self.writer.add_graph(self.Uphi, (E, V, u))

        E = torch.rand(1, 2).to(self.device)
        V1 = torch.rand(1, 512).to(self.device)
        V2 = torch.rand(1, 512).to(self.device)
        u = torch.rand(1, 100).to(self.device)
        self.writer.add_graph(self.Ephi, (E, V1, V2, u))

    def log(self, t_data):
        out = open(self.outName, 'a')
        print >> out, self.criterion
        print >> out, 'lr:{}'.format(self.lr)
        print >> out, self.optimizer.state_dict()
        print >> out, self.Uphi
        print >> out, self.Vphi
        print >> out, self.Ephi
        print >> out, self.u
        print >> out, 'Time consuming for loading datasets:', t_data
        out.close()
        # self.showNetwork()

    def resetU(self):
        if u_initial:
            self.u = torch.FloatTensor([random.random() for i in xrange(u_num)]).view(1, -1)
        else:
            self.u = torch.FloatTensor([0.0 for i in xrange(u_num)]).view(1, -1)
        self.u = self.u.to(self.device)

    def updateNetwork(self):
        self.train_set.setBuffer(1)
        step = 1
        average_epoch = 0
        edge_counter = 0.0
        for head in xrange(1, self.train_test):
            self.train_set.loadNext()  # Get the next frame
            edge_counter += self.train_set.m * self.train_set.n
            show_name = 'LOSS_{}'.format(step)
            print '         Step -', step
            start = time.time()
            data_loader = DataLoader(dataset=self.train_set, num_workers=self.numWorker, batch_size=self.batchsize, shuffle=True)
            for epoch in xrange(1, self.nEpochs):
                num = 0
                epoch_loss = 0.0
                v_loss = 0.0
                arpha_loss = 0.0
                beta_loss = 0.0
                for iteration in enumerate(data_loader, 1):
                    index, (e, gt, vs_index, vr_index) = iteration
                    # print '*'*36
                    # print e.size()
                    # print gt.size()
                    e = e.to(self.device)
                    gt = gt.to(self.device)

                    self.optimizer.zero_grad()

                    u_ = self.Uphi(self.train_set.E, self.train_set.V, self.u)
                    v1 = self.train_set.getApp(1, vs_index)
                    v2 = self.train_set.getApp(0, vr_index)
                    e_ = self.Ephi(e, v1, v2, u_)
                    v2_ = self.Vphi(e_, v1, v2, u_)

                    arpha = torch.mean(torch.abs(u_))
                    arpha_loss += arpha.item()
                    arpha.backward(retain_graph=True)

                    loss = self.criterion(e_, gt.squeeze(1))
                    epoch_loss += loss.item()
                    loss.backward(retain_graph=True)

                    tmp_gt = 1 - torch.FloatTensor(gt.cpu().numpy()).to(self.device)
                    v_l = self.criterion_v(tmp_gt * v2, tmp_gt * v2_)
                    v_loss += v_l.item()
                    v_l.backward()

                    self.optimizer.step()

                    #  Show the parameters of the Uphi and Ephi to check the process of optimiser
                    # print self.Uphi.features[0].weight.data
                    # print self.Ephi.features[0].weight.data
                    # raw_input('continue?')

                    num += self.batchsize

                if self.show_process and self.step_input:
                    a = raw_input('Continue(0-step, 1-run, 2-run with showing)?')
                    if a == '1':
                        self.show_process = 0
                    elif a == '2':
                        self.step_input = 0

                epoch_loss /= num
                print '         Loss of epoch {}: {}.'.format(epoch, epoch_loss)
                self.writer.add_scalars(show_name, {'regular': epoch_loss, 'v': v_loss/num,
                                                    'u': arpha_loss/num}, epoch)
                if epoch_loss < self.loss_threhold:
                    break

            print '         Time consuming:{}\n\n'.format(time.time()-start)
            self.updateUVE()
            self.train_set.showE()
            self.showU()
            average_epoch += epoch
            self.writer.add_scalar('epoch', epoch, step)
            step += 1
            self.train_set.swapFC()
        out = open(self.outName, 'a')
        print >> out, 'Average edge:', edge_counter*1.0/step, '.',
        print >> out, 'Average epoch:', average_epoch*1.0/step, 'for',
        print >> out, 'Random' if edge_initial else 'IoU'
        out.close()

    def saveModel(self):
        print 'Saving the Uphi model...'
        torch.save(self.Uphi, t_dir+'uphi_%02d.pth'%self.seq_index)
        print 'Saving the Vphi model...'
        torch.save(self.Vphi, t_dir+'vphi_%02d.pth'%self.seq_index)
        print 'Saving the Ephi model...'
        torch.save(self.Ephi, t_dir+'ephi_%02d.pth'%self.seq_index)
        print 'Saving the global variable u...'
        torch.save(self.u, t_dir+'u_%02d.pth'%self.seq_index)
        print 'Done!'

    def updateUVE(self):
        u_ = self.Uphi(self.train_set.E, self.train_set.V, self.u)

        self.u = u_.data

        # update the edges
        nxt = self.train_set.nxt
        for edge in self.train_set:
            e, gt, vs_index, vr_index = edge
            e = e.view(1,-1).to(self.device)
            v1 = self.train_set.getApp(1, vs_index)
            v2 = self.train_set.getApp(0, vr_index)
            e_ = self.Ephi(e, v1, v2, u_)
            v2_ = self.Vphi(e_, v1, v2, u_)
            if gt.item():
                self.train_set.detections[nxt][vr_index][0] = v2_.data
            self.train_set.edges[vs_index][vr_index] = e_.data.view(-1)

    def update(self):
        start = time.time()
        # self.evaluation(1)
        if self.tag:
            self.evaluation(self.train_test)
        self.updateNetwork()
        self.saveModel()
        # self.evaluation(1)
        if self.tag:
            self.evaluation(self.train_test)
        out = open(self.outName, 'a')
        print >> out, 'The final time consuming:{}\n\n'.format((time.time()-start)/60)
        out.close()
        self.outputScalars()

    def outputScalars(self):
        self.writer.export_scalars_to_json(t_dir + 'scalars_%02d.json'%self.seq_index)
        self.writer.close()

    def evaluation(self, head):
        self.train_set.setBuffer(head)
        total_gt = 0.0
        total_ed = 0.0
        for step in xrange(1, self.train_test):
            self.train_set.loadNext()
            # print head+step, 'F',

            u_ = self.Uphi(self.train_set.E, self.train_set.V, self.u)

            # print 'Fo'
            m = self.train_set.m
            n = self.train_set.n
            ret = [[0.0 for i in xrange(n)] for j in xrange(m)]
            step_gt = self.train_set.step_gt
            total_gt += step_gt

            # update the edges
            # print 'T',
            nxt = self.train_set.nxt
            for edge in self.train_set.candidates:
                e, gt, vs_index, vr_index = edge
                e = e.view(1,-1)
                v1 = self.train_set.getApp(1, vs_index)
                v2 = self.train_set.getApp(0, vr_index)
                e_ = self.Ephi(e, v1, v2, u_)
                v2_ = self.Vphi(e_, v1, v2, u_)
                if gt.item():
                    self.train_set.detections[nxt][vr_index][0] = v2_.data
                self.train_set.edges[vs_index][vr_index] = e_.data.view(-1)
                tmp = F.softmax(e_)
                tmp = tmp.cpu().data.numpy()[0]
                ret[vs_index][vr_index] = float(tmp[0])

            self.train_set.showE()
            self.showU()

            # for j in ret:
            #     print j
            results = self.hungarian.compute(ret)
            # print head+step, results,
            step_ed = 0.0
            for (j, k) in results:
                step_ed += self.train_set.gts[j][k].numpy()[0]
            total_ed += step_ed

            # print 'Fi'
            print '     {} Step ACC:{}/{}({}%)'.format(head+step, int(step_ed), int(step_gt), step_ed/step_gt*100)
            self.train_set.swapFC()

        tra_tst = 'training sets' if head == 1 else 'testing sets'
        # print 'Final {} ACC:{}/{}({}%)'.format(tra_tst, int(total_ed), int(total_gt), total_ed/total_gt*100)
        out = open(self.outName, 'a')
        print >> out, 'Final {} ACC:{}/{}({}%)'.format(tra_tst, int(total_ed), int(total_gt), total_ed/total_gt*100)
        out.close()

    def showU(self):
        out = open(self.outName, 'a')
        print >> out, '     u'
        print >> out, self.u.view(10, -1) # reshape the size of z with aspect of 10 * 10
        out.close()


if __name__ == '__main__':
    try:
        if not os.path.exists('Results/'):
            os.mkdir('Results/')

        year = 16
        f_dir = 'Results/MOT%s/' % year
        if not os.path.exists(f_dir):
            os.mkdir(f_dir)

        if edge_initial == 1:
            f_dir += 'Random/'
        elif edge_initial == 0:
            f_dir += 'IoU/'

        if not os.path.exists(f_dir):
            os.mkdir(f_dir)

        name_dir = 'all/'
        t_dir = f_dir + name_dir
        if not os.path.exists(t_dir):
            os.mkdir(t_dir)
        else:
            deleteDir(t_dir)
            os.mkdir(t_dir)

        start = time.time()
        print '     Starting Graph Network...'
        gn = GN()
        print 'Time consuming:', time.time()-start
    except KeyboardInterrupt:
        print 'Time consuming:', time.time()-start
        print ''
        print '-'*90
        print 'Existing from training early.'
# ImportError:/libgomp.so.1: version 'GOMP_4.0' not found (required by torch/lib/libcaffe2.so)
#/usr/lib/x86_64-linux-gnu/libgomp.so.1
