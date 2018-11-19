import shutil, os
from global_set import gap_dir

year = 17
name = 'motmetrics'
types = ['DPM', 'SDP', 'FRCNN']
seqs = [2, 4, 5, 9, 10, 11, 13]  # the set of sequences
lengths = [600, 1050, 837, 525, 654, 900, 750]  # the length of the sequence

cover_sel = 'Recover'

name2 = 'M_A_r_%s'%cover_sel + gap_dir
test_seqs = [1, 3, 6, 7, 8, 12, 14]

# copy the results for testing sets
for type in types:
    for i in xrange(len(seqs)):
        src_dir = 'Results/MOT%d/Random/%02d/%d/%s_%s_7_%s%s/res_training.txt'%(year,
                                                                               seqs[i],
                                                                               lengths[i],
                                                                               name,
                                                                               type,
                                                                               cover_sel,
                                                                                gap_dir)

        des_d = '../Test%d/%s/'%(year, name2)
        if not os.path.exists(des_d):
            os.mkdir(des_d)
        des_dir = des_d + 'MOT%d-%02d-%s.txt'%(year, test_seqs[i], type)

        print src_dir
        print des_dir
        shutil.copyfile(src_dir, des_dir)


# copy the results for training sets
for type in types:
    for i in xrange(len(seqs)):
        src_dir = '../MOT/MOT%d/train/MOT%d-%02d-%s/gt/gt.txt'%(year, year, seqs[i], type)
        des_d = '../Test%d/%s/'%(year, name2)
        if not os.path.exists(des_d):
            os.mkdir(des_d)

        des_dir = des_d + 'MOT%d-%02d-%s.txt'%(year, seqs[i], type)
        print src_dir
        print des_dir

        shutil.copyfile(src_dir, des_dir)
