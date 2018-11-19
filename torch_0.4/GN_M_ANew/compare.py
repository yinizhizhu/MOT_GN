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

getData('DMAN')
getData('AM_ADM17')
getData('GMPHD_KCF')
getData('HAM_SADF17')
getData('PHD_GSDL17')
getData('EAMTT_17')

#
#
# data = open('Compare/MOTDT17.txt', 'r')
# out = open('Compare/motdt17.txt', 'w')
# for line in data.readlines():
#     line = line.strip().replace(' ', '').replace(',', '')
#     attrs = line.split('\t')
#     print line, '-', len(attrs)
#     # 1, 8, 9, 10
#     print >> out, '%d\t%d\t%d\t%.1f'%(int(attrs[8]), int(attrs[9]), int(attrs[10]), float(attrs[1]))
# out.close()
# data.close()
#
#
# data = open('Compare/PHD_GSDL17.txt', 'r')
# out = open('Compare/phd_gsdl17.txt', 'w')
# for line in data.readlines():
#     line = line.strip().replace(' ', '').replace(',', '')
#     attrs = line.split('\t')
#     print line, '-', len(attrs)
#     # 1, 8, 9, 10
#     print >> out, '%d\t%d\t%d\t%.1f'%(int(attrs[8]), int(attrs[9]), int(attrs[10]), float(attrs[1]))
# out.close()
# data.close()
#
#
# data = open('Compare/EAMTT.txt', 'r')
# out = open('Compare/eamtt.txt', 'w')
# for line in data.readlines():
#     line = line.strip().replace(' ', '').replace(',', '')
#     attrs = line.split('\t')
#     print line, '-', len(attrs)
#     # 1, 8, 9, 10
#     print >> out, '%d\t%d\t%d\t%.1f'%(int(attrs[8]), int(attrs[9]), int(attrs[10]), float(attrs[1]))
# out.close()
# data.close()
#
#
# data = open('Compare/GMPHD_KCF.txt', 'r')
# out = open('Compare/gmphd_kcf.txt', 'w')
# for line in data.readlines():
#     line = line.strip().replace(' ', '').replace(',', '')
#     attrs = line.split('\t')
#     print line, '-', len(attrs)
#     # 1, 8, 9, 10
#     print >> out, '%d\t%d\t%d\t%.1f'%(int(attrs[8]), int(attrs[9]), int(attrs[10]), float(attrs[1]))
# out.close()
# data.close()
#
#
# data = open('Compare/App.txt', 'r')
# out = open('Compare/app.txt', 'w')
# counter = 1
# for line in data.readlines():
#     if counter % 3 == 0:
#         line = line.strip().replace('|', '')
#         attrs = line.split(' ')
#         print attrs, '-', len(attrs)
#         # 10, 11, 12, 14
#         print >> out, '%d\t%d\t%d\t%.1f'%(int(attrs[10]), int(attrs[11]), int(attrs[12]), float(attrs[14]))
#     counter += 1
# out.close()
# data.close()


data = open('Compare/OLGN.txt', 'r')
out = open('Compare/olgn.txt', 'w')
counter = 1
for line in data.readlines():
    if counter % 4 == 3:
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


data = open('Compare/Final.txt', 'r')
out = open('Compare/final.txt', 'w')
counter = 1
for line in data.readlines():
    if counter % 4 == 3:
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


data = open('Compare/OnGN.txt', 'r')
out = open('Compare/ongn.txt', 'w')
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


data = open('Compare/GN_SOT.txt', 'r')
out = open('Compare/gn_sot.txt', 'w')
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


data = open('Compare/Validation.txt', 'r')
out = open('Compare/valdataion.txt', 'w')
for line in data.readlines():
    line = line.strip()
    attrs = line.split(' ')
    i = len(attrs)-1
    while i>=0:
        if len(attrs[i]) == 0:
            del attrs[i]
        i-=1
    print attrs
    # print >> out, '&',
    n = len(attrs)
    for i in xrange(n):
        if i < n-1:
            print >> out, attrs[i], '&',
        else:
            print >> out, attrs[i],
    print >> out, ' \\\\'
    print >> out, '\hline'
out.close()
data.close()