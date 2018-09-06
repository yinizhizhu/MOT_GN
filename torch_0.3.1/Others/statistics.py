from PIL import Image


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class statistics():
    def __init__(self, cuda=True):
        self.trainDir = 'MOT16/train/'
        self.testDir = 'MOT16/test/'
        self.label = ['', 'Pedestrian', 'Person on vehicle',
                      'Car', 'Bicycle', 'Motorbike', 'Non motorized vehicle',
                      'Static person', 'Distractor', 'Occluder',
                      'Occluder on the ground', 'Occluder full', 'Reflection']

    def reset(self):
        self.visibility_ratio = [0, 0, 0]  # 0 - not visible heading, 1 - rest of zero visibility, 2 - non-zero
        self.bbx_size = [0, 0]  # 0 - small bounding box, 1 - suitable size
        self.cross_border = [0, 0]  # 0 - cross the border, 1 - within the image
        self.cross_border_size = [0, 0]  # 0 - small bounding box, 1 - suitable size

        self.confidence_score = [0, 0]

    def fixBB(self, x, y, w, h, size):
        width, height = size

        tag = False
        if x < 0 or y < 0:
            tag = True
        if w+x > width or h+y > height:
            tag = True

        w = min(w+x, width)
        h = min(h+y, height)
        x = max(x, 0)
        y = max(y, 0)
        w -= x
        h -= y
        return x, y, w, h, tag

    def preprocess(self, tag):
        self.reset()
        basis = self.trainDir if tag == 0 else self.testDir
        # trainList = os.listdir(basis)
        trainList = ['in_place', 'right_left']
        # trainList = ['MOT16-05', 'MOT16-10', 'MOT16-11', 'MOT16-13']
        # trainList = ['MOT16-05']
        for part in trainList:
            part = basis+part

            # get the length of the sequence
            info = part+'/seqinfo.ini'
            f = open(info, 'r')
            f.readline()
            for line in f.readlines():
                line = line.strip().split('=')
                if line[0] == 'seqLength':
                    seqL = int(line[1])
            f.close()

            # read the image
            imgs = [0] # store the sequence
            imgDir = part+'/img1/'
            for i in xrange(1, seqL+1):
                img = load_img(imgDir + '%06d.jpg'%i)
                imgs.append(img)

            print part

            if tag == 0:
                # get the gt
                gt = part + '/gt/gt.txt'
                f = open(gt, 'r')
                pre = -1
                for line in f.readlines():
                    line = line.strip().split(',')
                    if line[7] == '1':
                        """
                        Condition needed be taken into consideration:
                            x, y < 0 and x+w > W, y+h > H
                        """
                        index = int(line[0])
                        id = int(line[1])
                        x, y = int(line[2]), int(line[3])
                        w, h = int(line[4]), int(line[5])
                        l, vr = int(line[7]), float(line[8])

                        # sweep the invisible head-bbx from the training data
                        if pre != id and vr == 0:
                            self.visibility_ratio[0] += 1
                        else:
                            pre = id
                            if vr == 0:
                                self.visibility_ratio[1] += 1
                            else:
                                self.visibility_ratio[2] += 1

                        img = imgs[index]
                        x, y, w, h, cross_tag = self.fixBB(x, y, w, h, img.size)

                        if w < 25 or h < 50:
                            self.bbx_size[0] += 1
                        else:
                            self.bbx_size[1] += 1

                        if cross_tag:
                            self.cross_border[0] += 1
                            if w < 25 or h < 50:
                                self.cross_border_size[0] += 1
                            else:
                                self.cross_border_size[1] += 1
                        else:
                            self.cross_border[1] += 1
                f.close()
            else:
                # get the det
                det = part + '/det/det.txt'
                f = open(det, 'r')
                for line in f.readlines():
                    line = line.strip().split(',')
                    index = int(line[0])
                    x = int(float(line[2]))
                    y = int(float(line[3]))
                    w = int(float(line[4]))
                    h = int(float(line[5]))
                    cs = float(line[6])

                    img = imgs[index]
                    # x_, y_, w_, h_, (width, height) = x, y, w, h, img.size
                    x, y, w, h, cross_tag = self.fixBB(x, y, w, h, img.size)
                    # if cross_tag:
                    #     print x_, y_, w_+x_-width, h_+y_-height, width, height

                    if w < 25 or h < 50:
                        self.bbx_size[0] += 1
                    else:
                        self.bbx_size[1] += 1

                    if cross_tag:
                        self.cross_border[0] += 1
                        if w < 25 or h < 50:
                            self.cross_border_size[0] += 1
                        else:
                            self.cross_border_size[1] += 1
                    else:
                        self.cross_border[1] += 1

                    if cs < 0:
                        self.confidence_score[0] += 1
                    else:
                        self.confidence_score[1] += 1
                f.close()

    def show(self, tag):

        print '-'*45,
        print ('Test' if tag else 'Train'),
        print '-'*45
        print self.visibility_ratio
        print self.bbx_size
        print self.cross_border
        print self.cross_border_size
        print self.confidence_score

try:
    test = statistics()
    for i in xrange(0, 1):
        test.preprocess(i)
        test.show(i)
    # print ' In the mot_dataset.py...'
except KeyboardInterrupt:
    print ''
    print '-'*90
    print 'Existing from training early.'

# Statistics
# print '-'*45, 'Train', '-'*45
# vr = [1169.0+8893.0, 100345.0]
# vr_sum = sum(vr)
# print vr_sum
# print 'Visibility Ratio:', vr[0]/vr_sum, vr[1]/vr_sum  #, vr[2]/vr_sum
#
# bb = [13150.0, 97257.0]
# bb_sum = sum(bb)
# print bb_sum
# print 'The size of bounding box:', bb[0]/bb_sum, bb[1]/bb_sum
#
# csb = [16292.0, 94115.0]
# csb_sum = sum(csb)
# print csb_sum
# print 'Cross the Border:', csb[0]/csb_sum, csb[1]/csb_sum
#
# csbb = [2170.0, 14122.0]
# csbb_sum = sum(csbb)
# print csbb_sum
# print 'The size of the bbx which crosses the border:', csbb[0]/csbb_sum, csbb[1]/csbb_sum
#
# print '-'*45, 'Test', '-'*45
# bb = [3315.0, 132061.0]
# bb_sum = sum(bb)
# print bb_sum
# print 'The size of bounding box:', bb[0]/bb_sum, bb[1]/bb_sum
#
# csb = [4772.0, 130604.0]
# csb_sum = sum(csb)
# print csb_sum
# print 'Cross the border:', csb[0]/csb_sum, csb[1]/csb_sum
#
# csbb = [0.0, 4772.0]
# csbb_sum = sum(csbb)
# print csbb_sum
# print 'The size of the bbx which crosses the border:', csbb[0]/csbb_sum, csbb[1]/csbb_sum
#
# cs = [53175.0, 82201.0]
# cs_sum = sum(cs)
# print cs_sum
# print 'Confidence Score:', cs[0]/cs_sum, cs[1]/cs_sum
