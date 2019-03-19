data = open('Compare/GN_VOT.txt', 'r')
out = open('Compare/gn_vot.txt', 'w')
counter = 1
for line in data.readlines():
    if counter % 3 == 0:
        line = line.strip().replace('|', '')
        attrs = line.split(' ')
        print attrs, '-', len(attrs)
        i = len(attrs)-1
        while i>=0:
            if len(attrs[i]) == 0:
                del attrs[i]
            i -= 1
        # 10, 11, 12, 14
        print >> out, '%d\t%d\t%d\t%.1f'%(int(attrs[10]), int(attrs[11]), int(attrs[12]), float(attrs[14]))
    counter += 1
out.close()
data.close()


data = open('Compare/GN_DIS.txt', 'r')
out = open('Compare/gn_dis.txt', 'w')
counter = 1
for line in data.readlines():
    if counter % 3 == 0:
        line = line.strip().replace('|', '')
        attrs = line.split(' ')
        print attrs, '-', len(attrs)
        i = len(attrs)-1
        while i>=0:
            if len(attrs[i]) == 0:
                del attrs[i]
            i -= 1
        # 10, 11, 12, 14
        print >> out, '%d\t%d\t%d\t%.1f'%(int(attrs[10]), int(attrs[11]), int(attrs[12]), float(attrs[14]))
    counter += 1
out.close()
data.close()
