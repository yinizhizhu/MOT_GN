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

# name2 = 'GN_VOT_FR_dseq'
name2 = 'DPM'
test_seqs = [1, 3, 6, 7, 8, 12, 14]
test_lengths = [450, 1500, 1194, 500, 625, 900, 750]

# copy the results for testing sets

n = len(test_seqs)

types = ['DPM4-0.5']
for type in types:
    for i in xrange(n):
        src_dir = 'Results/MOT%d/%s/%02d/%d/%s_%s_7%s/res_training.txt'%(year,
                                                                         init_dir,
                                                                         test_seqs[i],
                                                                         test_lengths[i],
                                                                         name,
                                                                         type,
                                                                         '_cf_0.3_iou_0.5_vc_0.98__Recover_fgap_8_new')

        des_d = '../Test%d/%s/'%(year, name2)
        if not os.path.exists(des_d):
            os.mkdir(des_d)
        t = 'DPM'
        des_dir = des_d + 'MOT%d-%02d-%s.txt'%(year, test_seqs[i], t)

        print src_dir,
        print des_dir
        shutil.copyfile(src_dir, des_dir)
