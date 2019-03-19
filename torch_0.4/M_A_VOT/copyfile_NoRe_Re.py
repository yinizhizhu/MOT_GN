import shutil, os
from global_set import edge_initial

if edge_initial:
    init_dir = 'Random'
else:
    init_dir = 'IoU'

year = 17
name = 'motmetrics'
seqs = [2, 4, 5, 9, 10, 11, 13]  # the set of sequences
lengths = [600, 1050, 837, 525, 654, 900, 750]  # the length of the sequence

name2 = 'MOT_M_ANew_dis'
test_seqs = [1, 3, 6, 7, 8, 12, 14]
test_lengths = [450, 1500, 1194, 500, 625, 900, 750]

# copy the results for testing sets

n = len(test_seqs)

types = ['DPM0', 'SDP']
for type in types:
    for i in xrange(n):
        src_dir = 'Results/MOT%d/%s/%02d/%d/%s_%s_7%s/res_training.txt'%(year,
                                                                         init_dir,
                                                                         test_seqs[i],
                                                                         test_lengths[i],
                                                                         name,
                                                                         type,
                                                                         '_0.7_decay_1.90_Recover_uupdate_vc_0.99_exp')

        des_d = '../Test%d/%s/'%(year, name2)
        if not os.path.exists(des_d):
            os.mkdir(des_d)
        if type == 'DPM0':
            t = 'DPM'
        else:
            t = type
        des_dir = des_d + 'MOT%d-%02d-%s.txt'%(year, test_seqs[i], t)

        print src_dir,
        print des_dir
        shutil.copyfile(src_dir, des_dir)

types = ['FRCNN']
for type in types:
    for i in xrange(len(test_seqs)):
        src_dir = 'Results/MOT%d/%s/%02d/%d/%s_%s_7%s/res_training.txt'%(year,
                                                                         init_dir,
                                                                         test_seqs[i],
                                                                         test_lengths[i],
                                                                         name,
                                                                         type,
                                                                         '_0.7_decay_1.90_NoRecover_uupdate_vc_0.99_exp')

        des_d = '../Test%d/%s/'%(year, name2)
        if not os.path.exists(des_d):
            os.mkdir(des_d)
        des_dir = des_d + 'MOT%d-%02d-%s.txt'%(year, test_seqs[i], type)

        print src_dir,
        print des_dir
        shutil.copyfile(src_dir, des_dir)

# copy the results for training sets
types = ['DPM', 'SDP', 'FRCNN']
for type in types:
    for i in xrange(n):
        src_dir = '../MOT/MOT%d/train/MOT%d-%02d-%s/gt/gt.txt'%(year, year, seqs[i], type)
        des_d = '../Test%d/%s/'%(year, name2)
        if not os.path.exists(des_d):
            os.mkdir(des_d)

        des_dir = des_d + 'MOT%d-%02d-%s.txt'%(year, seqs[i], type)
        print src_dir,
        print des_dir

        shutil.copyfile(src_dir, des_dir)