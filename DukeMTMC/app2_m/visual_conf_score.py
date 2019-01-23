import os, shutil
import matplotlib.pyplot as plt


def deleteDir(del_dir):
    shutil.rmtree(del_dir)


def getNum(str):
    str = str.split(')')[0].split('/')
    return [float(p.split(':')[1]) for p in str]


delta = 1  # 1: TP-FP-FN, 0:FP+FN


cf = open('Detections/conf_score.txt', 'r')
acc = [[[] for j in xrange(4)] for i in xrange(9)]
ALL = [[0.0 for j in xrange(3)] for i in xrange(9)]  # 0 - TP, 1 - DET, 2 - GT
step = 0
index = 0
for line in cf.readlines():
    if step:
        # print line.strip().split(' ')
        move = 0
        container = []
        parts = line.strip().split(' ')
        print parts
        # print parts[1], (0.3+(step-1)*0.05)
        for part in parts[2:]:
            part = part.split('(')
            num = getNum(part[1])
            container.append(num)
            acc[index][move].append(float(part[0]))
            move += 1
        print container
        if delta:
            acc[index][move].append((3*container[0][0] - container[0][1] - container[1][1])/container[1][1])  # TP - FP - FN
        else:
            acc[index][move].append((-2*container[0][0] + container[0][1] + container[1][1])/container[1][1])  # FP + FN
        ALL[step-1][0] += container[0][0]
        ALL[step-1][1] += container[0][1]
        ALL[step-1][2] += container[1][1]
    else:
        index += 1
    step = (step+1)%10
cf.close()

print ALL


def getALL():
    DELTA_GT = []
    TP_GT = []
    for i in xrange(9):
        if delta:
            DELTA_GT.append((3*ALL[i][0]-ALL[i][1]-ALL[i][2])/ALL[i][2])  # TP - FP - FN
        else:
            DELTA_GT.append((-2*ALL[i][0]+ALL[i][1]+ALL[i][2])/ALL[i][2])  # FP + FN
        TP_GT.append(ALL[i][0]/ALL[i][2])
    return DELTA_GT, TP_GT

DELTA_GT, TP_GT = getALL()

out = open('Detections/sorted.txt', 'w')
for i in xrange(9):
    for line in acc[i]:
        print >> out, line
    print >> out, ''
out.close()


def getMax(con):
    index, maxN = 0, con[0]
    for i in xrange(1, len(con)):

        if delta:
            judge = con[i] > maxN
        else:
            judge = con[i] < maxN
        if judge:
            maxN = con[i]
            index = i
    return index, maxN


def draw():
    out_dir = 'Detections/Pic/'
    if os.path.exists(out_dir):
        deleteDir(out_dir)
    os.mkdir(out_dir)

    x = [0.3+i*0.05 for i in xrange(9)]
    for i in xrange(1, 9):
        plt.figure()
        plt.plot()
        plt.xlabel('cfs')
        plt.ylabel('acc')
        # print conf_score[i]
        # l1, = plt.plot(x, acc[i][0], label='tp_det', color='sandybrown', linestyle='--')
        # l2, = plt.plot(x, acc[i][1], label='tp_gt', color='red', linestyle='--')
        # l3, = plt.plot(x, acc[i][2], label='TP_GT', color='slateblue', linestyle='--')
        # l4, = plt.plot(x, acc[i][3], label='delta_gt', color='blue', linestyle='--')
        # plt.legend(handles=[l1, l2, l3, l4], labels=['tp_det', 'tp_gt', 'TP_GT', 'delta_gt'], loc='best')


        l1, = plt.plot(x, acc[i][0], label='tp_det', color='sandybrown', linestyle='--')
        l2, = plt.plot(x, acc[i][1], label='tp_gt', color='red', linestyle='--')
        l3, = plt.plot(x, acc[i][3], label='delta_gt', color='blue', linestyle='--')

        index, maxN = getMax(acc[i][3])
        plt.plot([0.3+index*0.05, 0.3 + index*0.05], [maxN-0.1, maxN], color='blue')

        plt.legend(handles=[l1, l2, l3], labels=['tp_det', 'tp_gt', 'delta_gt'], loc='best')
        plt.savefig(out_dir + '%d.png'%i)
        plt.close()

    plt.figure()
    plt.plot()
    plt.xlabel('CFS')
    plt.ylabel('ACC')
    l1, = plt.plot(x, DELTA_GT, label='DELTA_GT', color='sandybrown', linestyle='--')

    index, maxN = getMax(DELTA_GT)
    plt.plot([0.3+index*0.05, 0.3 + index*0.05], [maxN-0.1, maxN], color='sandybrown')

    l2, = plt.plot(x, TP_GT, label='TP_GT', color='red', linestyle='--')
    plt.legend(handles=[l1, l2], labels=['DELTA_GT', 'TP_GT'], loc='best')
    plt.savefig(out_dir + 'ALL.png')
    plt.close()

draw()