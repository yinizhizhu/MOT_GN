import random, torch
from munkres import Munkres


def testIndex():
    a = [[random.random() for i in xrange(3)] for j in xrange(2)]

    b = Munkres()
    results = b.compute(a)
    print results

for i in xrange(6, 10, 1):
    print i


def testCat():
    a = torch.FloatTensor([[random.random() for i in xrange(5)]])
    b = torch.FloatTensor([[random.random() for i in xrange(5)]])
    c = torch.cat((a, b), dim = 0)
    print c
    d = torch.cat((a, b), dim = 1)
    print d

