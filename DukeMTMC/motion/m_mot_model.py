import torch, math
import torch.nn as nn
from m_global_set import criterion_s, u_s, e_s

v_num = 6  # Only take the appearance into consideration, and add velocity when basic model works
u_num = 100
e_num = 1 if criterion_s else 2

uphi_n = 256
ephi_n = 256

# uphi_n = 512
# ephi_n = 1024


class uphi(nn.Module):
    def __init__(self):
        super(uphi, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(u_num+v_num+e_num, uphi_n),
            nn.LeakyReLU(inplace=True),
            nn.Linear(uphi_n, u_num),
        )
        # self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                print m
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, e, v, u):
        """
        The network which updates the global variable u
        :param e: the aggregation of the probability
        :param v: the aggregation of the vertice
        :param u: global variable
        """
        # print 'U:', e.size(), v.size(), u.size()
        bs = e.size()[0]
        if u.size()[0] == 1:
            if bs == 1:
                tmp = u
            else:
                tmp = torch.cat((u, u), dim=0)
                for i in xrange(2, bs):
                    tmp=torch.cat((tmp, u), dim=0)
        else:
            tmp = u
        x = torch.cat((e, v), dim=1)
        x = torch.cat((x, tmp), dim=1)
        return self.features(x)


class ephi(nn.Module):
    def __init__(self):
        super(ephi, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(u_num+v_num*2+e_num, ephi_n),
            nn.LeakyReLU(inplace=True),
            nn.Linear(ephi_n, e_num),
        )
        # self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                print m
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, e, v1, v2, u):
        """
        The network which updates the probability e
        :param e: the probability between two detections
        :param v1: the sender
        :param v2: the receiver
        :param u: global variable
        """
        # print 'E:', e.size(), v1.size(), v2.size(), u.size()
        bs = e.size()[0]
        if u.size()[0] == 1:
            if bs == 1:
                tmp = u
            else:
                tmp = torch.cat((u, u), dim=0)
                for i in xrange(2, bs):
                    tmp=torch.cat((tmp, u), dim=0)
        else:
            tmp = u
        x = torch.cat((e, v1), dim=1)
        x = torch.cat((x, v2), dim=1)
        x = torch.cat((x, tmp), dim=1)
        return self.features(x)