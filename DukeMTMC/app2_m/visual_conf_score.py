import matplotlib.pyplot as plt

cf = open('Detections/conf_score.txt', 'r')
conf_score = [[[] for j in xrange(3)] for i in xrange(10)]
step = 0
for line in cf.readlines():
    if step:
        print line.strip().split(' ')
        move = 0
        for part in line.strip().split(' ')[2:]:
            part = part.split('(')
            print part
            conf_score[step][move].append(float(part[0]))
            move += 1
    step = (step+1)%10
cf.close()

out = open('Detections/sorted.txt', 'w')
for i in xrange(10):
    for line in conf_score[i]:
        print >> out, line
out.close()


x = [0.3+i*0.05 for i in xrange(9)]
for i in xrange(1, 10):
    plt.figure()
    plt.plot()
    plt.xlabel('cfs')
    plt.ylabel('ACC')
    print conf_score[i]
    l1, = plt.plot(x, conf_score[i][0], label='acc_det', color='sandybrown', linestyle='--')
    l2, = plt.plot(x, conf_score[i][1], label='acc_gt', color='cadetblue', linestyle='--')
    l3, = plt.plot(x, conf_score[i][2], label='allacc_gt', color='slateblue', linestyle='--')
    plt.legend(handles=[l1, l2, l3], labels=['acc_det', 'acc_gt', 'allacc_gt'], loc='best')
    plt.savefig('Detections/Pic/%.2f.png'%x[i-1])
    plt.close()