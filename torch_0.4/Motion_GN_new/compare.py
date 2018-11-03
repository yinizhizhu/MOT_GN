data = open('Compare/DMAN.txt', 'r')
out = open('Compare/dman.txt', 'w')
for line in data.readlines():
    line = line.strip().replace(' ', '').replace(',', '')
    attrs = line.split('\t')
    print line, '-', len(attrs)
    # 1, 8, 9, 10
    print >> out, '%d\t%d\t%d\t%.1f'%(int(attrs[8]), int(attrs[9]), int(attrs[10]), float(attrs[1]))
out.close()
data.close()


data = open('Compare/MOTDT17.txt', 'r')
out = open('Compare/motdt17.txt', 'w')
for line in data.readlines():
    line = line.strip().replace(' ', '').replace(',', '')
    attrs = line.split('\t')
    print line, '-', len(attrs)
    # 1, 8, 9, 10
    print >> out, '%d\t%d\t%d\t%.1f'%(int(attrs[8]), int(attrs[9]), int(attrs[10]), float(attrs[1]))
out.close()
data.close()


data = open('Compare/PHD_GSDL17.txt', 'r')
out = open('Compare/phd_gsdl17.txt', 'w')
for line in data.readlines():
    line = line.strip().replace(' ', '').replace(',', '')
    attrs = line.split('\t')
    print line, '-', len(attrs)
    # 1, 8, 9, 10
    print >> out, '%d\t%d\t%d\t%.1f'%(int(attrs[8]), int(attrs[9]), int(attrs[10]), float(attrs[1]))
out.close()
data.close()


data = open('Compare/EAMTT.txt', 'r')
out = open('Compare/eamtt.txt', 'w')
for line in data.readlines():
    line = line.strip().replace(' ', '').replace(',', '')
    attrs = line.split('\t')
    print line, '-', len(attrs)
    # 1, 8, 9, 10
    print >> out, '%d\t%d\t%d\t%.1f'%(int(attrs[8]), int(attrs[9]), int(attrs[10]), float(attrs[1]))
out.close()
data.close()


data = open('Compare/GMPHD_KCF.txt', 'r')
out = open('Compare/gmphd_kcf.txt', 'w')
for line in data.readlines():
    line = line.strip().replace(' ', '').replace(',', '')
    attrs = line.split('\t')
    print line, '-', len(attrs)
    # 1, 8, 9, 10
    print >> out, '%d\t%d\t%d\t%.1f'%(int(attrs[8]), int(attrs[9]), int(attrs[10]), float(attrs[1]))
out.close()
data.close()


data = open('Compare/App.txt', 'r')
out = open('Compare/app.txt', 'w')
counter = 1
for line in data.readlines():
    if counter % 3 == 0:
        line = line.strip().replace('|', '')
        attrs = line.split(' ')
        print attrs, '-', len(attrs)
        # 10, 11, 12, 14
        print >> out, '%d\t%d\t%d\t%.1f'%(int(attrs[10]), int(attrs[11]), int(attrs[12]), float(attrs[14]))
    counter += 1
out.close()
data.close()


data = open('Compare/Motion.txt', 'r')
out = open('Compare/motion.txt', 'w')
counter = 1
for line in data.readlines():
    if counter % 3 == 0:
        line = line.strip().replace('|', '')
        attrs = line.split(' ')
        print attrs, '-', len(attrs)
        # 10, 11, 12, 14
        print >> out, '%d\t%d\t%d\t%.1f'%(int(attrs[10]), int(attrs[11]), int(attrs[12]), float(attrs[14]))
    counter += 1
out.close()
data.close()
