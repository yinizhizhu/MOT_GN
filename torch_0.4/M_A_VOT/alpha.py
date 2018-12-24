import matplotlib.pyplot as plt

alpha = open('Compare/alpha.txt', 'r')

DPM = []
FRCNN = []
SDP = []
ALL = []

for line in alpha.readlines():
    attrs = line.strip().replace('\t', '').split(' ')
    i = len(attrs)-1
    print attrs,
    while i>=0:
        if len(attrs[i]) == 0:
            del attrs[i]
        i -= 1
    print attrs
    DPM.append(float(attrs[1])/100.0)
    FRCNN.append(float(attrs[2])/100.0)
    SDP.append(float(attrs[3])/100.0)
    ALL.append(float(attrs[4])/100.0)

x = [i/10.0 for i in xrange(1, 10)]
print len(x), x
print len(DPM), DPM
print len(FRCNN), FRCNN
print len(SDP), SDP

alpha.close()

fig = plt.figure()

TL = fig.add_subplot(221)
TL.set_xlim(0.09, 0.93)
TL.set_ylim(0.338, 0.343)
TL.plot(x, DPM, 'sandybrown', label='DPM')
TL.plot([0.6, 0.6], [0.338, 0.342], color='sandybrown', linestyle='--')
TL.legend(loc='upper left')


TL = fig.add_subplot(222)
TL.set_xlim(0.09, 0.93)
TL.set_ylim(0.481, 0.497)
TL.plot(x, FRCNN, 'steelblue', label='FRCNN')
TL.plot([0.7, 0.7], [0.481, 0.496], color='steelblue', linestyle='--')
TL.legend(loc='best')


TL = fig.add_subplot(223)
TL.set_xlim(0.09, 0.93)
TL.set_ylim(0.554, 0.567)
TL.plot(x, SDP, 'cadetblue', label='SDP')
TL.plot([0.7, 0.7], [0.554, 0.566], color='cadetblue', linestyle='--')
TL.legend(loc='best')


TL = fig.add_subplot(224)
TL.set_xlim(0.09, 0.93)
TL.set_ylim(0.458, 0.469)
TL.plot(x, ALL, 'slateblue', label='OVERALL')
TL.plot([0.7, 0.7], [0.458, 0.468], color='slateblue', linestyle='--')
TL.legend(loc='best')


#
# TL = fig.add_subplot(221)
# TL.set_xlim(0.09, 0.93)
# TL.set_ylim(0.338, 0.343)
# TL.plot(x, DPM, 'teal', label='DPM')
# TL.plot([0.6, 0.6], [0.338, 0.342], color='teal', linestyle='--')
# TL.legend(loc='upper left')
#
#
# TL = fig.add_subplot(222)
# TL.set_xlim(0.09, 0.93)
# TL.set_ylim(0.481, 0.497)
# TL.plot(x, FRCNN, 'maroon', label='FRCNN')
# TL.plot([0.7, 0.7], [0.481, 0.496], color='maroon', linestyle='--')
# TL.legend(loc='best')
#
#
# TL = fig.add_subplot(223)
# TL.set_xlim(0.09, 0.93)
# TL.set_ylim(0.554, 0.567)
# TL.plot(x, SDP, 'darkgreen', label='SDP')
# TL.plot([0.7, 0.7], [0.554, 0.566], color='darkgreen', linestyle='--')
# TL.legend(loc='best')
#
#
#
# TL = fig.add_subplot(224)
# TL.set_xlim(0.09, 0.93)
# TL.set_ylim(0.458, 0.469)
# TL.plot(x, ALL, 'sienna', label='OVERALL')
# TL.plot([0.7, 0.7], [0.458, 0.468], color='sienna', linestyle='--')
# TL.legend(loc='best')


fig.subplots_adjust(wspace=0.3)
fig.savefig('Compare/alpha.pdf')