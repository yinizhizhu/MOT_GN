# data = open('Compare/DMAN.txt', 'r')
# out = open('Compare/dman.txt', 'w')
# for line in data.readlines():
#     line = line.strip().replace(' ', '').replace(',', '')
#     attrs = line.split('\t')
#     print line, '-', len(attrs)
#     # 1, 8, 9, 10
#     print >> out, '%d\t%d\t%d\t%.1f'%(int(attrs[8]), int(attrs[9]), int(attrs[10]), float(attrs[1]))
# out.close()
# data.close()


data = open('Compare/M_A_r.txt', 'r')
out = open('Compare/m_a_r.txt', 'w')
counter = 1
for line in data.readlines():
    if counter % 3 == 0:
        line = line.strip().replace('|', '')
        attrs = line.split(' ')
        i = len(attrs) - 1
        while i >= 0:
            if attrs[i] == '':
                del attrs[i]
            i -= 1
        print len(attrs), '-', attrs
        # 10, 11, 12, 14
        print >> out, '%d\t%d\t%d\t%.1f'%(int(attrs[10]), int(attrs[11]), int(attrs[12]), float(attrs[14]))
    counter += 1
out.close()
data.close()