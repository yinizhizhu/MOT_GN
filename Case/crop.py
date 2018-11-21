import matplotlib.pyplot as plt
from PIL import Image

plt.axis('off')
types = ['appearance', 'motion']
for i in xrange(1, 3):
    fig = plt.figure()
    data = open('%d_bbx.txt'%i, 'r')
    first = data.readline().strip()
    second = data.readline().strip()
    attrs = data.readline().strip().split(',')
    bbx = [0.0 for j in xrange(4)]
    for j in xrange(4):
        bbx[j] = float(attrs[j])
    data.close()
    for k in xrange(2):
        t = types[k]
        img1 = Image.open('%d_%s/%s.jpg'%(i, t, first))
        img2 = Image.open('%d_%s/%s.jpg'%(i, t, second))
        crop1 = img1.crop([bbx[0]-bbx[2], bbx[1]-60, bbx[0]+2*bbx[2], bbx[1]+bbx[3]+60])
        crop2 = img2.crop([bbx[0]-bbx[2], bbx[1]-60, bbx[0]+2*bbx[2], bbx[1]+bbx[3]+60])

        TL = fig.add_subplot(2, 2, 1+k)
        TL.imshow(crop1)

        TL = fig.add_subplot(2, 2, 2+k)
        TL.imshow(crop2)


    fig.subplots_adjust(wspace=0.3)
    fig.savefig('%d.pdf'%i)
