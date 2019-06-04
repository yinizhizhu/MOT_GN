# data = open('Compare/GN_DSEQ_FGAP16.txt', 'r')
# out = open('Compare/gn_dseq_fgap16.txt', 'w')
# data = open('Compare/VOT_DSEQ_FGAP.txt', 'r')
# out = open('Compare/VOT_dseq_fgap.txt', 'w')
# data = open('Compare/GN_FGAP16.txt', 'r')
# out = open('Compare/gn_fgap16.txt', 'w')
data = open('Compare/GN_VOT_NoRecover.txt', 'r')
out = open('Compare/gn_vot_norecover.txt', 'w')
counter = 1

pku = False

for line in data.readlines():
    tag = (counter % 4 == 3) if pku else (counter % 3 == 0)
    if tag:
        line = line.strip().replace('|', '')
        attrs = line.split(' ')
        print attrs, '-', len(attrs)
        i = len(attrs)-1
        while i>=0:
            if len(attrs[i]) == 0:
                del attrs[i]
            i -= 1
        # 10, 11, 12, 14
        print attrs
        print >> out, '%d\t%d\t%d\t%.1f'%(int(attrs[10]), int(attrs[11]), int(attrs[12]), float(attrs[14]))
    counter += 1
out.close()
data.close()


def getData(name):
    data = open('Compare/%s.txt'%name, 'r')
    out = open('Compare/%s.txt'%name.lower(), 'w')
    for line in data.readlines():
        line = line.strip().replace(' ', '').replace(',', '')
        attrs = line.split('\t')
        print line, '-', len(attrs)
        # 1, 8, 9, 10
        print >> out, '%d\t%d\t%d\t%.1f'%(int(attrs[8]), int(attrs[9]), int(attrs[10]), float(attrs[1]))
    out.close()
    data.close()

getData('MHT_bLSTM')
getData('NOMT')