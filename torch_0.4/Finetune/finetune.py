import time, os
import torch.nn as nn
import torch.optim as optim
import torch, torchvision
from torch.utils.data import DataLoader
from finetune_dataset import DatasetFromFolder

from tensorboardX import SummaryWriter


class appearance(nn.Module):
    def __init__(self):
        super(appearance, self).__init__()
        features = list(torchvision.models.resnet34(pretrained=True).children())[:-1]
        # print features
        self.features = nn.Sequential(*features)

    def forward(self, x):
        return self.features(x)


class finetuning():
    def __init__(self, cuda=True):
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.nEpochs = 10
        self.numWorker = 4
        self.batchsize = 8
        self.loadModel()
        self.optimizer = optim.Adam(self.Appearance.parameters(), lr=1e-5)
        self.train_set = DatasetFromFolder()
        self.data_loader = DataLoader(dataset=self.train_set, num_workers=self.numWorker, batch_size=self.batchsize,
                                 shuffle=True)

    def loadModel(self):
        self.Appearance = appearance()
        self.Appearance.to(self.device)
        self.Appearance.train()  # Training model

        f = open('Fine-tune/finetune.txt', 'w')
        print >> f, self.Appearance
        f.close()

    def mse(self, a, b):
        c = a - b
        ret = torch.mean(c*c)
        return ret

    def saveModel(self, epoch):
        self.writer.export_scalars_to_json('Fine-tune/scalars_%02d.json'%epoch)
        self.writer.close()

        print 'Saving the Appearance model...'
        torch.save(self.Appearance, 'Fine-tune/appearance_%02d.pth'%epoch)

    def finetune(self):
        step = 0
        head = time.time()
        for epoch in xrange(0, self.nEpochs):
            self.writer = SummaryWriter()
            start = time.time()
            num = 0
            epoch_loss = 0.0
            for iteration in enumerate(self.data_loader, 1):
                index, (anchor, positive, negative) = iteration

                a = self.Appearance(anchor.to(self.device))
                p = self.Appearance(positive.to(self.device))
                n = self.Appearance(negative.to(self.device))

                # Penalize the u to let its value not too big

                one = torch.FloatTensor([1.0]).to(self.device)
                zero = torch.FloatTensor([0.0]).to(self.device)

                self.optimizer.zero_grad()
                d_a_p = self.mse(a, p)
                loss1 = d_a_p.item()

                d_a_n = self.mse(a, n)
                loss2 = d_a_n.item()

                triplet_loss = one + d_a_p - d_a_n
                loss = torch.max(zero, triplet_loss)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalars('Step_loss', {'Loss1': loss1, 'Loss2': loss2}, step)

                num += self.batchsize
                step += 1
                # break

            print ' Time consuming:', time.time()-start
            epoch_loss /= num
            self.writer.add_scalars('Epoch_loss', {'Loss': epoch_loss}, epoch)
            self.saveModel(epoch)

        print 'Time consuming:', time.time() - head
try:
    if not os.path.exists('Fine-tune/'):
        os.mkdir('Fine-tune/')

    ft = finetuning()
    print '     Finetuning...'
    ft.finetune()
except KeyboardInterrupt:
    print ''
    print '-'*90
    print 'Existing from training early.'