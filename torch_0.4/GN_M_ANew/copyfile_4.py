import shutil, os


year = 17
name = 'motmetrics'
types = ['DPM0', 'FRCNN', 'SDP']
seqs = [9, 11, 13]  # the set of sequences
lengths = [525, 900, 750]  # the length of the sequence

name2 = 'MOT_M_ANew_bb'
test_seqs = [9, 11, 13]

# copy the results for testing sets
for type in types:
    for i in xrange(len(seqs)):
        src_dir = 'Results/MOT%d/Random/%02d/%d/%s_%s_4_0.7_decay_1.90_Recover_uupdate_tau_tdir1.0/res_training.txt'%(year, seqs[i], lengths[i], name, type)

        des_d = '../Validation/%s/'%(name2)
        if not os.path.exists(des_d):
            os.mkdir(des_d)
        des_dir = des_d + 'MOT%d-%02d-%s.txt'%(year, test_seqs[i], type)

        print src_dir
        print des_dir
        shutil.copyfile(src_dir, des_dir)

