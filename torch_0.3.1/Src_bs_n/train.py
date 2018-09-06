from __future__ import print_function
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import DatasetFromFolder

# from pycrayon import CrayonClient
# import time
#
# cc = CrayonClient(hostname="localhost", port=8889)

# cc.remove_experiment("train_loss")
class trainer:
    def __init__(self, typeDir, epochs, l, load, dir):
        print('===> Loading datasets')
        train_set = DatasetFromFolder(self.typeDir)
        self.training_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=4, shuffle=True)

    def train(self, epoch):
        epoch_loss = 0
        for iteration, batch in enumerate(self.training_data_loader, 1):
            input, target = Variable(batch[0]), Variable(batch[1])
            if self.cuda:
                input = input.cuda()
                target = target.cuda()

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(input), target)
            epoch_loss += loss.data[0]
            loss.backward()
            self.optimizer.step()

            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(self.training_data_loader), loss.data[0]))

            # self.trainL.add_scalar_value(self.trainLN, loss.data[0], time.time() - self.timeH)

        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(self.training_data_loader)))

    def training(self):
        for epoch in range(1, self.nEpochs + 1):
            self.train(epoch)
        # *.pth - Pretraining gblurConv with 1/3
        # *F.pth - Finetuning gblurConv with 1/3 on pretraining gblurConv
        # *F_ALL.pth - Finetuning gblurConv with 100% on pretraining gblurConv
        if (self.load == 2):
            torch.save(self.model, '%s/%sF_ALL_%d.pth' % (self.dir, self.typeDir, epoch))
            print("Checkpoint saved to %s/%sF_ALL_%d.pth" % (self.dir, self.typeDir, epoch))
        else:
            torch.save(self.model, '%s/%s_%d.pth' % (self.dir,self.typeDir, epoch))
            print("Checkpoint saved to %s/%s_%d.pth" % (self.dir,self.typeDir, epoch))
