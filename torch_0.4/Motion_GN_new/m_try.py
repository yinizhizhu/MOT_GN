import random, torch
from munkres import Munkres
import torch.nn as nn

def testIndex():
    a = [[random.random() for i in xrange(3)] for j in xrange(2)]

    b = Munkres()
    results = b.compute(a)
    print results


for i in xrange(0, 301):
    p = i/100.0
    ans = torch.tensor([[1-p, p]])
    print ans,

    gt = torch.LongTensor([1])
    print gt,

    criterion = nn.CrossEntropyLoss()

    loss = criterion(ans, gt)
    print loss
