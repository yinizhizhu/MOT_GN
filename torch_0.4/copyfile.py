import os
import sys
import getpass
import shutil

name = 'motmetrics'
seqs = [2, 4, 5, 9, 10, 11, 13]  # the set of sequences
lengths = [600, 1050, 837, 525, 654, 900, 750]  # the length of the sequence

name2 = 'Test_motion2'
test_seqs = [1, 3, 6, 7, 8, 12, 14]

for i in xrange(len(seqs)):
    src_dir = 'Results/MOT16/IoU/%02d/%d/%s/res_training.txt'%(seqs[i], lengths[i], name)
    des_dir = '../Test/%s/MOT16-%02d.txt'%(name2, test_seqs[i])
    print src_dir
    print des_dir
    shutil.copyfile(src_dir, des_dir)