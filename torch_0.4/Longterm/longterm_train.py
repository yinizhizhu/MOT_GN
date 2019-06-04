# from __future__ import print_function
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from longterm_train_dataset import DatasetFromFolder
import time, random, os, shutil
from munkres import Munkres
from m_global_set import edge_initial, u_initial, debug
from m_mot_model import *

torch.manual_seed(123)
np.random.seed(123)

t_dir = ''  # the dir of the final level


def deleteDir(del_dir):
    shutil.rmtree(del_dir)


# 7 - training with all the sequences for final model
# 4 - training with four sequences for selecting best parameters
# 0 - training with all the sequences but only first 80% of sequence for training, and the rest for validation
train_set_num = 0


class GN():
    def __init__(self, lr=5e-4, batchs=40, cuda=True):
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
        self.Ephi = ephi().to(self.device)

        self.criterion = nn.MSELoss() if criterion_s else nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(self.device)

        self.optimizer = optim.Adam([
            {'params': self.Uphi.parameters()},
            {'params': self.Ephi.parameters()}],
            lr=lr)

        if train_set_num == 4:
            seqs = [2, 4, 5, 10]
            lengths = [600, 1050, 837, 654]
        else:
            seqs = [2, 4, 5, 9, 10, 11, 13]
            lengths = [600, 1050, 837, 525, 654, 900, 750]

        for i in xrange(len(seqs)):
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
            if train_set_num == 0:
                self.train_test = int(self.train_test * 0.8)

            self.tag = 0
            # self.loss_threhold = 0.3133
            self.loss_threhold = 0.03
            self.update()

            print '     Logging...'
            t_data = time.time() - start
            self.log(t_data)

    def log(self, t_data):
        out = open(self.outName, 'a')
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
        self.u = self.u.to(self.device)

    def updateNetwork(self):
        self.train_set.setBuffer(1)
        step = 1
        average_epoch = 0
        edge_counter = 0.0
        for head in xrange(1, self.train_test):
            self.train_set.loadNext()  # Get the next frame
            edge_counter += self.train_set.m * self.train_set.n
            start = time.time()
            show_name = 'LOSS_{}'.format(step)
            print '         Step -', step
            data_loader = DataLoader(dataset=self.train_set, num_workers=self.numWorker, batch_size=self.batchsize, shuffle=True)
            for epoch in xrange(1, self.nEpochs):
                num = 0
                epoch_loss = 0.0
                arpha_loss = 0.0
                for iteration in enumerate(data_loader, 1):
                    index, (e, gt, vs_index, vs_pos, vr_index) = iteration

                    # print e.size(), e
                    # print gt.size(), gt
                    # print vs_index.size(), vs_index
                    # print vs_pos.size(), vs_pos
                    # print vr_index.size(), vr_index

                    self.optimizer.zero_grad()

                    e = e.to(self.device)
                    gt = gt.to(self.device)
                    u_ = self.Uphi(self.train_set.E, self.train_set.V, self.u)
                    v1 = self.train_set.getMotion(1, vs_index, vs_pos)
                    v2 = self.train_set.getMotion(0, vr_index, vs_pos, vs_index)
                    # print v1
                    # print v2
                    # raw_input('Continue?')

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
                    arpha_loss += arpha.item()
                    arpha.backward(retain_graph=True)

                    #  The regular loss
                    loss = self.criterion(e_, gt.squeeze(1))
                    epoch_loss += loss.item()
                    loss.backward()

                    # update the network: Uphi and Ephi
                    self.optimizer.step()

                    #  Show the parameters of the Uphi and Ephi to check the process of optimiser
                    # print self.Uphi.features[0].weight.data
                    # print self.Ephi.features[0].weight.data
                    # raw_input('continue?')

                    num += vs_pos.numel()

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

            print '         Time consuming:{}\n\n'.format(time.time()-start)
            self.updateUE()
            self.train_set.showE()
            self.showU()
            average_epoch += epoch
            step += 1
            self.train_set.getVelocity()
        out = open(self.outName, 'a')
        print >> out, 'Average edge:', edge_counter*1.0/step, '.',
        print >> out, 'Average epoch:', average_epoch*1.0/step, 'for',
        print >> out, 'Random' if edge_initial else 'IoU'
        out.close()

    def saveModel(self):
        print 'Saving the Uphi model...'
        torch.save(self.Uphi, t_dir+'uphi_%02d.pth'%self.seq_index)
        print 'Saving the Ephi model...'
        torch.save(self.Ephi, t_dir+'ephi_%02d.pth'%self.seq_index)
        print 'Saving the global variable u...'
        torch.save(self.u, t_dir+'u_%02d.pth'%self.seq_index)
        print 'Done!'

    def updateUE(self):
        u_ = self.Uphi(self.train_set.E, self.train_set.V, self.u)

        self.u = u_.data

        # update the edges
        for edge in self.train_set:
            e, gt, vs_index, vs_pos, vr_index = edge
            e = e.to(self.device).view(1,-1)
            v1 = self.train_set.getMotion(1, vs_index, vs_pos)
            v2 = self.train_set.getMotion(0, vr_index, vs_pos, vs_index)
            e_ = self.Ephi(e, v1, v2, u_)
            self.train_set.edges[vs_pos][vs_index][vr_index] = e_.data.view(-1)

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
        elif edge_initial == 3:
            f_dir += 'Equal/'

        if not os.path.exists(f_dir):
            os.mkdir(f_dir)

        t_dir = f_dir + 'longterm_all_%d/'%train_set_num
        if debug:
            deleteDir(t_dir)
        if not os.path.exists(t_dir):
            os.mkdir(t_dir)
            start = time.time()
            print '     Starting Graph Network...'
            gn = GN()
            print 'Time consuming:', time.time()-start
        else:
            # deleteDir(t_dir)
            # os.mkdir(t_dir)
            print 'The model has been here!'

    except KeyboardInterrupt:
        print 'Time consuming:', time.time()-start
        print ''
        print '-'*90
        print 'Existing from training early.'
# ImportError:/libgomp.so.1: version 'GOMP_4.0' not found (required by torch/lib/libcaffe2.so)
#/usr/lib/x86_64-linux-gnu/libgomp.so.1
