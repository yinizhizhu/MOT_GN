import random
import numpy as np
import torch.nn as nn
import torch, cv2, torchvision
from torch.autograd import Variable
import torch.nn.functional as F


def testPretrained():
    resnet = torchvision.models.resnet34(pretrained=True)
    print resnet


def testOperator():
    a = [[-random.random() for i in xrange(3)] for j in xrange(3)]
    print a[0]
    print a[1]

    b = torch.FloatTensor(a[0])
    b = Variable(b)
    torch.save(b, 'Model/u.pth')
    b = b.view(1, -1)
    print b
    print torch.sum(b)
    print torch.mean(b)
    print torch.max(torch.sum(b), torch.mean(b))


    c = torch.sum(torch.abs(b)).data.numpy()[0]
    print c

    c = torch.mul(b,b)
    print c
    print b, b.volatile
    c = Variable(c)

    print c, c.volatile

    d = torch.cat((b, c), dim=0)
    print d, d.volatile

    e = torch.cat((c, b), dim=0)
    print e, e.volatile


def testHungarian():
    from munkres import Munkres
    hungarian = Munkres()

    ret = []
    ret.append([0.5988732576370239, 0.593967080116272, 0.5874873399734497, 0.590605616569519, 0.5701925158500671, 0.5845481157302856, 0.5860251784324646])
    ret.append([0.5978249907493591, 0.5901663303375244, 0.5871194005012512, 0.5886968374252319, 0.5672130584716797, 0.5825384855270386, 0.5815722346305847])
    ret.append([0.5956549644470215, 0.5894451141357422, 0.5791335701942444, 0.5839282274246216, 0.5632365345954895, 0.577437698841095, 0.5764529705047607])
    ret.append([0.587761402130127, 0.5818054676055908, 0.575451135635376, 0.5763403177261353, 0.5562797784805298, 0.571980357170105, 0.5717030763626099])
    ret.append([0.6023207902908325, 0.5964986085891724, 0.5881332159042358, 0.5919026136398315, 0.5691371560096741, 0.5870761871337891, 0.5855976939201355])
    ret.append([0.5862066745758057, 0.580104410648346, 0.5727660059928894, 0.576642632484436, 0.5530561208724976, 0.5668273568153381, 0.5696781277656555])
    ret.append([0.5930306315422058, 0.5875985026359558, 0.5814967155456543, 0.5818882584571838, 0.5616766214370728, 0.5775641202926636, 0.5739971995353699])
    results = hungarian.compute(ret)
    print results

    ret = []
    ret.append([0.5978249907493591, 0.5901663303375244, 0.5871194005012512, 0.5886968374252319, 0.5672130584716797, 0.5825384855270386, 0.5815722346305847])
    ret.append([0.5988732576370239, 0.593967080116272, 0.5874873399734497, 0.590605616569519, 0.5701925158500671, 0.5845481157302856, 0.5860251784324646])
    ret.append([0.5956549644470215, 0.5894451141357422, 0.5791335701942444, 0.5839282274246216, 0.5632365345954895, 0.577437698841095, 0.5764529705047607])
    ret.append([0.587761402130127, 0.5818054676055908, 0.575451135635376, 0.5763403177261353, 0.5562797784805298, 0.571980357170105, 0.5717030763626099])
    ret.append([0.6023207902908325, 0.5964986085891724, 0.5881332159042358, 0.5919026136398315, 0.5691371560096741, 0.5870761871337891, 0.5855976939201355])
    ret.append([0.5862066745758057, 0.580104410648346, 0.5727660059928894, 0.576642632484436, 0.5530561208724976, 0.5668273568153381, 0.5696781277656555])
    ret.append([0.5930306315422058, 0.5875985026359558, 0.5814967155456543, 0.5818882584571838, 0.5616766214370728, 0.5775641202926636, 0.5739971995353699])
    results = hungarian.compute(ret)
    print results

    ret = []
    ret.append([0.5978249907493591, 0.5901663303375244, 0.5871194005012512, 0.5886968374252319, 0.5672130584716797, 0.5825384855270386, 0.5815722346305847])
    ret.append([0.5956549644470215, 0.5894451141357422, 0.5791335701942444, 0.5839282274246216, 0.5632365345954895, 0.577437698841095, 0.5764529705047607])
    ret.append([0.5988732576370239, 0.593967080116272, 0.5874873399734497, 0.590605616569519, 0.5701925158500671, 0.5845481157302856, 0.5860251784324646])
    ret.append([0.587761402130127, 0.5818054676055908, 0.575451135635376, 0.5763403177261353, 0.5562797784805298, 0.571980357170105, 0.5717030763626099])
    ret.append([0.6023207902908325, 0.5964986085891724, 0.5881332159042358, 0.5919026136398315, 0.5691371560096741, 0.5870761871337891, 0.5855976939201355])
    ret.append([0.5862066745758057, 0.580104410648346, 0.5727660059928894, 0.576642632484436, 0.5530561208724976, 0.5668273568153381, 0.5696781277656555])
    ret.append([0.5930306315422058, 0.5875985026359558, 0.5814967155456543, 0.5818882584571838, 0.5616766214370728, 0.5775641202926636, 0.5739971995353699])
    results = hungarian.compute(ret)
    print results

    ret = []
    ret.append([0.5978249907493591, 0.5901663303375244, 0.5871194005012512, 0.5886968374252319, 0.5672130584716797, 0.5825384855270386, 0.5815722346305847])
    ret.append([0.5956549644470215, 0.5894451141357422, 0.5791335701942444, 0.5839282274246216, 0.5632365345954895, 0.577437698841095, 0.5764529705047607])
    ret.append([0.587761402130127, 0.5818054676055908, 0.575451135635376, 0.5763403177261353, 0.5562797784805298, 0.571980357170105, 0.5717030763626099])
    ret.append([0.5988732576370239, 0.593967080116272, 0.5874873399734497, 0.590605616569519, 0.5701925158500671, 0.5845481157302856, 0.5860251784324646])
    ret.append([0.6023207902908325, 0.5964986085891724, 0.5881332159042358, 0.5919026136398315, 0.5691371560096741, 0.5870761871337891, 0.5855976939201355])
    ret.append([0.5862066745758057, 0.580104410648346, 0.5727660059928894, 0.576642632484436, 0.5530561208724976, 0.5668273568153381, 0.5696781277656555])
    ret.append([0.5930306315422058, 0.5875985026359558, 0.5814967155456543, 0.5818882584571838, 0.5616766214370728, 0.5775641202926636, 0.5739971995353699])
    results = hungarian.compute(ret)
    print results

    ret = []
    ret.append([0.5978249907493591, 0.5901663303375244, 0.5871194005012512, 0.5886968374252319, 0.5672130584716797, 0.5825384855270386, 0.5815722346305847])
    ret.append([0.5956549644470215, 0.5894451141357422, 0.5791335701942444, 0.5839282274246216, 0.5632365345954895, 0.577437698841095, 0.5764529705047607])
    ret.append([0.587761402130127, 0.5818054676055908, 0.575451135635376, 0.5763403177261353, 0.5562797784805298, 0.571980357170105, 0.5717030763626099])
    ret.append([0.6023207902908325, 0.5964986085891724, 0.5881332159042358, 0.5919026136398315, 0.5691371560096741, 0.5870761871337891, 0.5855976939201355])
    ret.append([0.5988732576370239, 0.593967080116272, 0.5874873399734497, 0.590605616569519, 0.5701925158500671, 0.5845481157302856, 0.5860251784324646])
    ret.append([0.5862066745758057, 0.580104410648346, 0.5727660059928894, 0.576642632484436, 0.5530561208724976, 0.5668273568153381, 0.5696781277656555])
    ret.append([0.5930306315422058, 0.5875985026359558, 0.5814967155456543, 0.5818882584571838, 0.5616766214370728, 0.5775641202926636, 0.5739971995353699])
    results = hungarian.compute(ret)
    print results

    ret = []
    ret.append([0.5978249907493591, 0.5901663303375244, 0.5871194005012512, 0.5886968374252319, 0.5672130584716797, 0.5825384855270386, 0.5815722346305847])
    ret.append([0.5956549644470215, 0.5894451141357422, 0.5791335701942444, 0.5839282274246216, 0.5632365345954895, 0.577437698841095, 0.5764529705047607])
    ret.append([0.587761402130127, 0.5818054676055908, 0.575451135635376, 0.5763403177261353, 0.5562797784805298, 0.571980357170105, 0.5717030763626099])
    ret.append([0.6023207902908325, 0.5964986085891724, 0.5881332159042358, 0.5919026136398315, 0.5691371560096741, 0.5870761871337891, 0.5855976939201355])
    ret.append([0.5862066745758057, 0.580104410648346, 0.5727660059928894, 0.576642632484436, 0.5530561208724976, 0.5668273568153381, 0.5696781277656555])
    ret.append([0.5988732576370239, 0.593967080116272, 0.5874873399734497, 0.590605616569519, 0.5701925158500671, 0.5845481157302856, 0.5860251784324646])
    ret.append([0.5930306315422058, 0.5875985026359558, 0.5814967155456543, 0.5818882584571838, 0.5616766214370728, 0.5775641202926636, 0.5739971995353699])
    results = hungarian.compute(ret)
    print results

    ret = []
    ret.append([0.5978249907493591, 0.5901663303375244, 0.5871194005012512, 0.5886968374252319, 0.5672130584716797, 0.5825384855270386, 0.5815722346305847])
    ret.append([0.5956549644470215, 0.5894451141357422, 0.5791335701942444, 0.5839282274246216, 0.5632365345954895, 0.577437698841095, 0.5764529705047607])
    ret.append([0.587761402130127, 0.5818054676055908, 0.575451135635376, 0.5763403177261353, 0.5562797784805298, 0.571980357170105, 0.5717030763626099])
    ret.append([0.6023207902908325, 0.5964986085891724, 0.5881332159042358, 0.5919026136398315, 0.5691371560096741, 0.5870761871337891, 0.5855976939201355])
    ret.append([0.5862066745758057, 0.580104410648346, 0.5727660059928894, 0.576642632484436, 0.5530561208724976, 0.5668273568153381, 0.5696781277656555])
    ret.append([0.5930306315422058, 0.5875985026359558, 0.5814967155456543, 0.5818882584571838, 0.5616766214370728, 0.5775641202926636, 0.5739971995353699])
    ret.append([0.5988732576370239, 0.593967080116272, 0.5874873399734497, 0.590605616569519, 0.5701925158500671, 0.5845481157302856, 0.5860251784324646])
    results = hungarian.compute(ret)
    print results

    ret = []
    ret.append([0.5956549644470215, 0.5894451141357422, 0.5791335701942444, 0.5839282274246216, 0.5632365345954895, 0.577437698841095, 0.5764529705047607])
    ret.append([0.5978249907493591, 0.5901663303375244, 0.5871194005012512, 0.5886968374252319, 0.5672130584716797, 0.5825384855270386, 0.5815722346305847])
    ret.append([0.587761402130127, 0.5818054676055908, 0.575451135635376, 0.5763403177261353, 0.5562797784805298, 0.571980357170105, 0.5717030763626099])
    ret.append([0.6023207902908325, 0.5964986085891724, 0.5881332159042358, 0.5919026136398315, 0.5691371560096741, 0.5870761871337891, 0.5855976939201355])
    ret.append([0.5862066745758057, 0.580104410648346, 0.5727660059928894, 0.576642632484436, 0.5530561208724976, 0.5668273568153381, 0.5696781277656555])
    ret.append([0.5930306315422058, 0.5875985026359558, 0.5814967155456543, 0.5818882584571838, 0.5616766214370728, 0.5775641202926636, 0.5739971995353699])
    ret.append([0.5988732576370239, 0.593967080116272, 0.5874873399734497, 0.590605616569519, 0.5701925158500671, 0.5845481157302856, 0.5860251784324646])
    results = hungarian.compute(ret)
    print results

    ret = [[random.random() for i in xrange(3)] for j in xrange(4)]
    print ret
    results = hungarian.compute(ret)
    print results

# testHungarian()


def showParameters():
    a = nn.Linear(10, 10)
    print a.weight.data
    print a.bias.data


def testRetainGraph():
    x = Variable(torch.randn(5, 5), requires_grad=True)
    y = Variable(torch.randn(5, 5), requires_grad=True)
    z = x*3 - y*4

    def print_grad(g):
        print g

    z.register_hook(print_grad)
    y.register_hook(print_grad)
    x.register_hook(print_grad)
    q = z.sum()*2
    q.backward()


def testRandomShuffle():
    a = [[i+j for i in xrange(3)] for j in xrange(3)]
    print a
    random.shuffle(a)
    print a


def swapNum(a, b):
    print a, b
    a = a^b # each bit indicate the same or not same
    b = a^b # get the a from the TAG
    a = a^b # get the b form the TAG
    print a, b


def torch4():
    cuda = True
    device = torch.device('cuda' if cuda else 'cpu')

    print torch.set_default_dtype(torch.float64)
    a = torch.tensor([random.random() for i in xrange(3)], device=device, requires_grad=True)
    print a
    print a.data
    print a.data.data.data
    print a.requires_grad

    print '-'*36
    c = a*2
    d = c.sum()
    d.backward()
    print a.grad

    print '-'*36
    e = torch.tensor(a.data, requires_grad=True)
    g = e + 1
    print a
    print e
    h = g.sum()
    h.backward()
    print e.grad
    print a.grad

    print '-'*36
    h = a.data
    h = h+1
    print a
    print h

# torch4()


def cat_4():
    a = torch.tensor([0, 1, 3, 2]).view(1,-1)
    b = torch.tensor([8, 4, 7, 8]).view(1,-1)
    c = torch.cat((a,b), dim=0)
    print c
    d = torch.cat((a,b), dim=1)
    print d

    e = torch.Tensor([[0], [0],[0],[0]])
    print e
    f = e.squeeze()
    print f

    f = torch.Tensor([[1]])
    print f
    print f.squeeze(dim=1)

# cat_4()


def loadJson():
    import json
    # reading data back
    with open('Json/all_scalars.json', 'r') as f:
        data = json.load(f)
        print data
        print len(data)
        for i in data:
            if 'LOSS_1/' in i:
                print i, data[i]

# loadJson()


def mk_del_dir():
    import shutil, os
    detsDir = '/dets/'
    if not os.path.exists(detsDir):
        # os.mkdir(detsDir)
        print '{} does not exist.'.format(detsDir)
    else:
        shutil.rmtree(detsDir)
    print detsDir, os.path.exists(detsDir)


def loadXls():
    from openpyxl import load_workbook

    f = open('out.txt', 'w')

    workbook = load_workbook('1.xlsx')
    sheets = workbook.get_sheet_names()
    booksheet = workbook.get_sheet_by_name(sheets[0])

    rows = booksheet.rows
    columns = booksheet.columns
    for row in rows:
        print row
        for col in row:
            print col
            word = col.value
            print word

            word = word.decode('gbk')
            print word

            print >> f, word.encode('utf8'), ' '
            print >> f, ''
    f.close()


def os_system():
    import commands
    out = open('finetune.txt', 'w')
    (status, output) = commands.getstatusoutput('python3 evaluation.py gt res')
    print >> out, output
    print status
    print output
    out.close()

# os_system()


def rename_dir():
    import os
    os.rename('Store_Exp/', 'Store_Exp1/')


def test_sort():
    a = [[random.random() for i in xrange(3)] for j in xrange(4)]
    for i in a:
        print i
    print '-'*90
    a = sorted(a, key=lambda b: b[0], reverse=True)
    for i in a:
        print i
    print '-'*90
    a = sorted(a, key=lambda b: b[1], reverse=False)
    for i in a:
        print i
    print '-'*90
    a = sorted(a, key=lambda b: b[2])
    for i in a:
        print i


import shutil
def deleteDir(del_dir):
    shutil.rmtree(del_dir)


def format():
    import os
    in_basic = 'Results/MOT16/'
    seqs = ['02', '04', '05', '09', '10', '11', '13']
    types = [['MotMetrics_IoU_cross_dets', 'MotMetrics_IoU_cross_gts'],
             ['MotMetrics_IoU_inner_dets', 'MotMetrics_IoU_inner_gts'],
             ['MotMetrics_IoU_inner_balanced_dets', 'MotMetrics_IoU_inner_balanced_gts'],
             ['MotMetrics_IoU_inner_balancedNearby_dets', 'MotMetrics_IoU_inner_balancedNearby_gts']]

    out_basic = in_basic+'MotMetrics/'
    if not os.path.exists(out_basic):
        os.mkdir(out_basic)
    else:
        deleteDir(out_basic)
        os.mkdir(out_basic)

    for type_2 in types:
        type, type2 = type_2
        in_dir = in_basic + type + '/'
        in_dir2 = in_basic + type2 + '/'

        print type
        tmp = type.split('_')[:-1]
        print tmp
        tmp = '_'.join(tmp)
        print tmp
        out_dir = out_basic + tmp + '/'
        print in_dir, out_dir
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        else:
            deleteDir(out_dir)
            os.mkdir(out_dir)
            print out_dir

        for seq in seqs:
            in_txt = in_dir + seq+'.txt'
            in_txt2 = in_dir2 + seq + '.txt'
            out_txt = out_dir +seq+'.txt'
            print ' ', in_txt, in_txt2, out_txt

            f = open(in_txt, 'r')
            f2 = open(in_txt2, 'r')
            out = open(out_txt, 'w')
            for line in f.readlines():
                tag_f2 = 0
                line = line.strip().split(' ')
                line2 = f2.readline().strip().split(' ')
                print '0.0', line
                print '0.0', line2
                if line[0] == 'IDF1':
                    print >> out, '\t',
                elif line[0] == 'OVERALL':
                    continue
                elif line[0] == 'Testing':
                    tag_f2 = 1
                    line[0] = type.split('_')[-1]
                    line2[0] = type2.split('_')[-1]
                elif line[0] == 'The':
                    tag_f2 = 1
                    line = line[2:]
                    line2 = line2[2:]
                elif '*' in line[0]:
                    line = [line[1]]

                # output the motmetrics of the detections
                for word in line[:-1]:
                    if len(word):
                        print >> out, '%s\t'%word,
                if len(line[-1]):
                    print >> out, line[-1]

                # output the motmetrics of the gts
                if tag_f2:
                    for word in line2[:-1]:
                        if len(word):
                            print >> out, '%s\t' % word,
                    if len(line2[-1]):
                        print >> out, line2[-1]
            out.close()
            f.close()
format()























