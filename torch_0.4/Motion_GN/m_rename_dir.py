# from __future__ import print_function
import os
import numpy as np
from m_mot_model import *
from m_global_set import edge_initial

torch.manual_seed(123)
np.random.seed(123)

year = 16
t_dir = ''  # the dir of the final level
seqs = [2, 4, 5, 9, 10, 11, 13]  # the set of sequences
lengths = [600, 1050, 837, 525, 654, 900, 750]  # the length of the sequence

target = '20_edge'  # inner, cross, gts

type_dir = 'IoU' if edge_initial == 0 else 'Random'

metric_dir = 'Results/MOT%d/MotMetrics_%s/' % (year, type_dir)
target_metric = 'Results/MOT%d/MotMetrics_%s_%s' % (year, type_dir, target)
print metric_dir, target_metric
# os.rename(metric_dir, target_metric)


def rename():
    out_dir = t_dir + 'motmetrics/'
    target_dir = t_dir + 'motmetrics_{}/'.format(target)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print 'There is no dir:', out_dir
    print '     ', out_dir, target_dir
    os.rename(out_dir, target_dir)


if __name__ == '__main__':
    try:
        f_dir = 'Results/MOT%s/' % year
        if not os.path.exists(f_dir):
            os.mkdir(f_dir)

        if edge_initial == 1:
            f_dir += 'Random/'
        elif edge_initial == 0:
            f_dir += 'IoU/'

        if not os.path.exists(f_dir):
            print f_dir, 'does not exist!'

        for i in xrange(7):
            seq_index = seqs[i]
            tts = []
            # tts = [tt for tt in xrange(100, 600, 100)]
            length = lengths[i]
            tts.append(length)

            for tt in tts:
                tag = 1
                if tt*2 > length:
                    if tt == length:
                        tag = 0
                    else:
                        continue
                print 'The sequence:', seq_index, '- The length of the training data:', tt

                s_dir = f_dir + '%02d/' % seq_index
                if not os.path.exists(s_dir):
                    print s_dir, 'does not exist!'

                t_dir = s_dir + '%d/' % tt
                if not os.path.exists(t_dir):
                    print t_dir, 'does not exist!'

                rename()
    except KeyboardInterrupt:
        print ''
        print '-'*90
        print 'Existing from training early.'
