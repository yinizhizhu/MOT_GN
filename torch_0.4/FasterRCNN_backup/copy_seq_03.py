import shutil, os
from global_set import edge_initial

if edge_initial:
    init_dir = 'Random'
else:
    init_dir = 'IoU'

year = 17
name = 'motmetrics'
name2 = 'FasterRCNN'
test_seqs = [3]
test_lengths = [1500]

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
                                                                         '_cf_0.1_iou_0.5_vc_0.99__Recover')

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
                                                                         '_cf_0.1_iou_0.5_vc_0.99__NoRecover')

        des_d = '../Test%d/%s/'%(year, name2)
        if not os.path.exists(des_d):
            os.mkdir(des_d)
        des_dir = des_d + 'MOT%d-%02d-%s.txt'%(year, test_seqs[i], type)

        print src_dir,
        print des_dir
        shutil.copyfile(src_dir, des_dir)

