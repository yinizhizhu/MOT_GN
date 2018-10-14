import random, torch
from munkres import Munkres


def testIndex():
    a = [[random.random() for i in xrange(3)] for j in xrange(2)]

    b = Munkres()
    results = b.compute(a)
    print results


a = torch.FloatTensor([random.random() for i in xrange(10)])
print a
print a[0], a[0].item()

index = [i for i in xrange(10)]
for i in xrange(len(index)-1, -1, -1):
    print index[i]