# from __future__ import print_function
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from dataset import DatasetFromFolder
import time, random, os, shutil
from munkres import Munkres
from global_set import edge_initial, u_initial, app_dir, SEQLEN
from mot_model import *

torch.manual_seed(123)
np.random.seed(123)

t_dir = ''  # the dir of the final level


def deleteDir(del_dir):
    shutil.rmtree(del_dir)


class GN():
    def __init__(self, lr=2e-6, cuda=True):
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
        self.nEpochs = 666

        self.lr = lr

        self.show_process = 0   # interaction
        self.step_input = 1

        self.loss_threhold = 0.21

        print '     Preparing the model...'
        # self.resetU()
        # self.Uphi = uphi().to(self.device)
        # self.Vphi = vphi().to(self.device)
        # self.Ephi1 = ephi().to(self.device)
        # self.Ephi2 = ephi().to(self.device)
        self.Uphi = torch.load(t_dir+'uphi_%d.pth'%3).to(self.device)
        self.Vphi = torch.load(t_dir+'vphi_%d.pth'%3).to(self.device)
        self.Ephi1 = torch.load(t_dir+'ephi1_%d.pth'%3).to(self.device)
        self.Ephi2 = torch.load(t_dir+'ephi2_%d.pth'%3).to(self.device)
        self.u = torch.load(t_dir+'u_%d.pth'%3).to(self.device)

        self.criterion = nn.MSELoss() if criterion_s else nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(self.device)

        self.criterion_v = nn.MSELoss().to(self.device)

        self.optimizer1 = optim.Adam([
            {'params': self.Ephi1.parameters()}],
            lr=lr)
        self.optimizer2 = optim.Adam([
            {'params': self.Uphi.parameters()},
            {'params': self.Vphi.parameters()},
            {'params': self.Ephi2.parameters()}],
            lr=lr)

        self.loadData()

    def loadData(self):
        self.train_test = SEQLEN
        for camera in xrange(4, 9):
            # print '     Loading Data...'
            self.seq_index = camera
            start = time.time()

            self.outName = t_dir+'result_%d.txt'%camera
            out = open(self.outName, 'w')
            out.close()

            self.train_set = DatasetFromFolder(camera, self.outName, show=0)

            self.update()

            print '     Logging...'
            t_data = (time.time() - start)/60
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

    def log(self, t_data):
        out = open(self.outName, 'a')
        print >> out, self.criterion
        print >> out, 'lr:{}'.format(self.lr)
        print >> out, self.optimizer1.state_dict()
        print >> out, self.optimizer2.state_dict()
        print >> out, self.Uphi
        print >> out, self.Vphi
        print >> out, self.Ephi1
        print >> out, self.Ephi2
        print >> out, self.u
        print >> out, 'Time consuming for loading datasets:', t_data
        out.close()

    def resetU(self):
        if u_initial:
            self.u = torch.FloatTensor([random.random() for i in xrange(u_num)]).view(1, -1)
        else:
            self.u = torch.FloatTensor([0.0 for i in xrange(u_num)]).view(1, -1)
        self.u = self.u.to(self.device)

    def updateNetwork(self):
        head = 1
        step = head + self.train_set.setBuffer(head)
        average_epoch = 0
        edge_counter = 0.0
        while step <= self.train_test:
            gap = self.train_set.loadNext()  # Get the next frame
            step += gap
            if step > self.train_test:
                break

            edges = self.train_set.m * self.train_set.n
            edge_counter += edges
            print '         Step -', step, '%d * %d'%(self.train_set.m, self.train_set.n)

            if edges >= 32:
                numWorker = 4
                batchsize = 8
            elif edges >= 18:
                numWorker = 3
                batchsize = 6
            else:
                numWorker = 2
                batchsize = 4

            data_loader = DataLoader(dataset=self.train_set, num_workers=numWorker, batch_size=batchsize, shuffle=True)

            for epoch_i in xrange(1, self.nEpochs):
                num = 0
                epoch_loss_i = 0.0
                for iteration in enumerate(data_loader, 1):
                    index, (e, gt, vs_index, vr_index) = iteration
                    e = e.to(self.device)
                    gt = gt.to(self.device)
                    vs = self.train_set.getApp(1, vs_index)
                    vr = self.train_set.getApp(0, vr_index)

                    self.optimizer1.zero_grad()
                    e1 = self.Ephi1(e, vs, vr, self.u)
                    # update the Ephi1
                    loss = self.criterion(e1, gt.squeeze(1))
                    loss.backward()
                    self.optimizer1.step()
                    num += e.size(0)
                if epoch_loss_i / num < self.loss_threhold:
                    break
            print '         Updating the Ephi1: %d times.'%epoch_i

            for epoch in xrange(1, self.nEpochs):
                num = 0
                epoch_loss = 0.0
                v_loss = 0.0
                arpha_loss = 0.0

                candidates = []
                E_CON, V_CON = [], []
                for iteration in enumerate(data_loader, 1):
                    index, (e, gt, vs_index, vr_index) = iteration
                    e = e.to(self.device)
                    gt = gt.to(self.device)
                    vs = self.train_set.getApp(1, vs_index)
                    vr = self.train_set.getApp(0, vr_index)

                    e1 = self.Ephi1(e, vs, vr, self.u)

                    e1 = e1.data
                    vr1 = self.Vphi(e1, vs, vr, self.u)
                    candidates.append((e1, gt, vs, vr, vr1))
                    E_CON.append(torch.mean(e1, 0))
                    V_CON.append(torch.mean(vs, 0))
                    V_CON.append(torch.mean(vr1.data, 0))

                E = self.train_set.aggregate(E_CON).view(1, -1)        # This part is the aggregation for Edge
                V = self.train_set.aggregate(V_CON).view(1, -1)        # This part is the aggregation for vertex
                # print E.view(1, -1)

                n = len(candidates)
                for i in xrange(n):
                    e1, gt, vs, vr, vr1 = candidates[i]
                    tmp_gt = 1 - torch.FloatTensor(gt.cpu().numpy()).to(self.device)

                    self.optimizer2.zero_grad()

                    u1 = self.Uphi(E, V, self.u)
                    e2 = self.Ephi2(e1, vs, vr1, u1)

                    # Penalize the u to let its value not too big
                    arpha = torch.mean(torch.abs(u1))
                    arpha_loss += arpha.item()
                    arpha.backward(retain_graph=True)

                    v_l = self.criterion_v(tmp_gt * vr, tmp_gt * vr1)
                    v_loss += v_l.item()
                    v_l.backward(retain_graph=True)

                    #  The regular loss
                    loss = self.criterion(e2, gt.squeeze(1))
                    epoch_loss += loss.item()
                    loss.backward()

                    # update the network: Uphi and Ephi
                    self.optimizer2.step()

                    num += e1.size(0)

                if self.show_process and self.step_input:
                    a = raw_input('Continue(0-step, 1-run, 2-run with showing)?')
                    if a == '1':
                        self.show_process = 0
                    elif a == '2':
                        self.step_input = 0

                epoch_loss /= num
                print '         Loss of epoch {}: {}.'.format(epoch, epoch_loss)
                if epoch_loss < self.loss_threhold:
                    break

            self.updateUVE()
            self.train_set.showE()
            self.showU()
            average_epoch += epoch
            self.train_set.swapFC()

            if step >= self.train_test:
                break

        out = open(self.outName, 'a')
        print >> out, 'Average edge:', edge_counter*1.0/step, '.',
        print >> out, 'Average epoch:', average_epoch*1.0/step, 'for',
        print >> out, 'Random' if edge_initial else 'IoU'
        out.close()

    def saveModel(self):
        print 'Saving the Uphi model...'
        torch.save(self.Uphi, t_dir+'uphi_%d.pth'%self.seq_index)
        print 'Saving the Vphi model...'
        torch.save(self.Vphi, t_dir+'vphi_%d.pth'%self.seq_index)
        print 'Saving the Ephi1 model...'
        torch.save(self.Ephi1, t_dir+'ephi1_%d.pth'%self.seq_index)
        print 'Saving the Ephi model...'
        torch.save(self.Ephi2, t_dir+'ephi2_%d.pth'%self.seq_index)
        print 'Saving the global variable u...'
        torch.save(self.u, t_dir+'u_%d.pth'%self.seq_index)
        print 'Done!'

    def updateUVE(self):
        with torch.no_grad():
            candidates = []
            E_CON, V_CON = [], []
            for edge in self.train_set:
                e, gt, vs_index, vr_index = edge
                e = e.view(1,-1).to(self.device)
                vs = self.train_set.getApp(1, vs_index)
                vr = self.train_set.getApp(0, vr_index)

                e1 = self.Ephi1(e, vs, vr, self.u)
                vr1 = self.Vphi(e1, vs, vr, self.u)
                candidates.append((e1, gt, vs, vr1, vs_index, vr_index))
                E_CON.append(e1)
                V_CON.append(vs)
                V_CON.append(vr1)

            E = self.train_set.aggregate(E_CON).view(1, -1)        # This part is the aggregation for Edge
            V = self.train_set.aggregate(V_CON).view(1, -1)        # This part is the aggregation for vertex
            u1 = self.Uphi(E, V, self.u)
            self.u = u1.data

            nxt = self.train_set.nxt
            for iteration in candidates:
                e1, gt, vs, vr1, vs_index, vr_index = iteration
                e2 = self.Ephi2(e1, vs, vr1, u1)
                if gt.item():
                    self.train_set.detections[nxt][vr_index][0] = vr1.data
                self.train_set.edges[vs_index][vr_index] = e2.data.view(-1)

    def update(self):
        start = time.time()
        self.updateNetwork()
        self.saveModel()
        out = open(self.outName, 'a')
        print >> out, 'The final time consuming:{}\n\n'.format((time.time()-start)/60)
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

        name_dir = '%s_4/'%app_dir
        t_dir = f_dir + name_dir
        start = time.time()
        print '     Starting Graph Network...'
        gn = GN()
        print 'Time consuming:', (time.time()-start)/60
        # if not os.path.exists(t_dir):
        #     os.mkdir(t_dir)
        #     start = time.time()
        #     print '     Starting Graph Network...'
        #     gn = GN()
        #     print 'Time consuming:', time.time()-start
        # else:
        #     # deleteDir(t_dir)
        #     # os.mkdir(t_dir)
        #     print 'The model has been here!'
    except KeyboardInterrupt:
        tmp_t = (time.time()-start)/60
        print 'Time consuming:', tmp_t
        print ''
        print '-'*90
        print 'Existing from training early.'
# ImportError:/libgomp.so.1: version 'GOMP_4.0' not found (required by torch/lib/libcaffe2.so)
#/usr/lib/x86_64-linux-gnu/libgomp.so.1
